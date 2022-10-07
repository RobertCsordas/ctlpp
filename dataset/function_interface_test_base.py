from typing import Dict, Any, Tuple, List, Union, Optional, Set
import numpy as np
import torch.utils.data
import bisect
import string
from .helpers import CacheLoaderMixin
import framework
from .sequence import TextSequenceTestState, TextClassifierTestState
import re
from dataclasses import dataclass
from tqdm import tqdm
from dataset.helpers.rejection_sampler import rejection_sample
from framework.utils import parallel_map


@dataclass
class FunctionTrace:
    expr: str
    res: int
    depth: int
    vals: List[int]
    fns: List[int]

    def __hash__(self) -> int:
        return hash(self.expr)


@dataclass
class RestrictedEdge:
    allowed_symbols: Set[int]
    target_function: Optional[int]
    target_group: Optional[int]


@dataclass
class SplitPlan:
    tree: List
    counts: Dict[int, int]


class FunctionInterfaceTestBase(CacheLoaderMixin, torch.utils.data.Dataset):
    in_vocabulary = None

    def gen_trace(self, vals: List[int], fns: List[int]) -> FunctionTrace:
        all = [self.symbol_names[vals[0]]] + [self.fn_names[f] for f in fns]
        if self.reversed:
            all = reversed(all)

        return FunctionTrace(" ".join(all), vals[-1], len(fns), vals, fns)

    def trace_from_input(self, initial_value: int, fns: List[int]) -> FunctionTrace:
        vals = [initial_value]
        for f in fns:
            vals.append(self.functions[f][vals[-1]])

        return self.gen_trace(vals, fns)

    def equal_split(self, l: List[int], slices: int, allow_less: bool = False, seed: Optional[np.random.RandomState] = None) -> List[List[int]]:
        if allow_less and len(l) < slices:
            res = [[] for _ in range(slices)]
            if len(l) > 0:
                has_sample = seed.choice(slices, len(l), replace=False)
                for i in range(len(l)):
                    res[has_sample[i]] = [l[i]]

            return res

        assert slices <= len(l)
        block_size = (len(l) + slices - 1) // slices
        return [l[i: i + block_size] for i in range(0, len(l), block_size)]

    def generate_groups(self, schema: Dict[str, Any], seed: np.random.RandomState):
        # Example schema:
        # {
        #     "sets": {"a1", "b1", "a2", "o", "b2"}],
        #     "set_sizes": {"o": 3}
        # }

        all_fns = list(range(self.n_functions))
        next_fn = 0

        unassigned_sets = schema["sets"].copy()

        self.groups = {}

        # Find groups with defined size
        for s in sorted(schema.get("set_sizes", {}).keys()):
            c = schema["set_sizes"][s]
            assert self.n_functions - next_fn > c
            self.groups[s] = all_fns[next_fn: next_fn + c]
            next_fn += c
            unassigned_sets.remove(s)

        # Assign the rest of the functions
        for s, fns in zip(sorted(unassigned_sets), self.equal_split(all_fns[next_fn:], len(unassigned_sets))):
            self.groups[s] = fns

        # Index to easily find the group of any function
        self.fn_to_group = {}
        for g, flist in self.groups.items():
            for fn in flist:
                self.fn_to_group[fn] = g

    def generate_restricted_overlaps(self, src_sets: List[str], target_set: str, seed: np.random.RandomState, imbalance: int = 0) -> List[List]:
        # Restricted edge format [...group history, {allowed symbols}, target function]
        remap_every = (self.n_symbols + self.symbol_overlap - 1) // self.symbol_overlap
        sym_rename = list(range(self.n_symbols))

        fn_overlap = list(sorted(self.groups[target_set]))

        edges = []

        for oi, f in enumerate(fn_overlap):
            overlapping_symbols = set(sym_rename[i % self.n_symbols] for i in range(oi * self.symbol_overlap, (oi + 1) * self.symbol_overlap))
            other_symbols = [s for s in range(self.n_symbols) if s not in overlapping_symbols]
            other_symbols_per_group = self.equal_split(other_symbols[imbalance:], len(src_sets), allow_less=True, seed=seed)
            other_symbols_per_group[0] += other_symbols[:imbalance]

            if oi % remap_every == (remap_every - 1):
                # If all symbols are covered by a pass, reshuffle
                sym_rename = seed.permutation(self.n_symbols).tolist()

            for sn, sg in zip(src_sets, other_symbols_per_group):
                symbols = overlapping_symbols.union(set(sg))
                edges.append([sn, symbols, f])

        return edges

    def invert_restricted_overlaps(self, restricted_overlaps: List[List]) -> List[List]:
        all_symbols = set(range(self.n_symbols))
        return [[*r[:-2], all_symbols - r[-2], r[-1]] for r in restricted_overlaps]

    def expand_edges(self, edges: List[List[Union[Set[str], str]]]) -> List[List[str]]:
        def expand_sets(s: List[Union[Set[str], str]]) -> List[List[str]]:
            prefix: List[List[str]] = [[]]
            for v in s:
                if isinstance(v, str):
                    for p in prefix:
                        p.append(v)
                else:
                    new_prefix = []
                    for val in v:
                        for p in prefix:
                            new_prefix.append(p + [val])
                    prefix = new_prefix
            return prefix

        res = []
        for e in edges:
            res += expand_sets(e)

        return res

    def generate_rollout_tree(self, schema: Dict[str, Any], depth_range: Tuple[int, int]) -> List:
        edges = self.expand_edges(schema["edges"])

        depth_range = (depth_range[0]+1, depth_range[1]+1)

        def step(history: List[str]) -> List:
            graph = []

            known = set()
            for e in edges:
                if e[-1] in known:
                    continue

                template = e[:-1]
                if history[-len(template):] == template:
                    if e[-1] == "OUT":
                        if len(history) >= depth_range[0]:
                            graph.append((e[-1], None))
                            known.add(e[-1])
                    elif len(history) < depth_range[1]:
                        nd = step(history + [e[-1]])
                        if nd:
                            graph.append((e[-1], nd))
                            known.add(e[-1])

            known_functions = set()
            for e in schema.get("restricted_edges", []):
                if e[-1] in known_functions:
                    continue

                template = e[:-2]
                if history[-len(template):] == template:
                    if e[-1] == "OUT":
                        if len(history) >= depth_range[0]:
                            graph.append((RestrictedEdge(e[-2], -1, "OUT"), None))
                            known_functions.add("OUT")
                    elif len(history) <= depth_range[1]:
                        group = self.fn_to_group[e[-1]]
                        nd = step(history + [group])
                        if nd:
                            graph.append((RestrictedEdge(e[-2], e[-1], group), nd))
                            known_functions.add(e[-1])

            return graph

        return step(["IN"])

    def length_count(self, tree) -> Dict[int, int]:
        res = {}

        def step(tree, allowed_symbols: set, l: int, nways: int):
            for edge in tree:
                if edge[0] == "OUT":
                    c = len(allowed_symbols) * nways
                    if c > 0:
                        res[l] = res.get(l, 0) + c
                elif isinstance(edge[0], str):
                    fns = self.groups[edge[0]]

                    if len(allowed_symbols) != self.n_symbols:
                        new_allowed_symbols = set()
                        for f in fns:
                            for s in allowed_symbols:
                                new_allowed_symbols.add(self.functions[f][s])
                    else:
                        new_allowed_symbols = allowed_symbols

                    step(edge[1], new_allowed_symbols, l + 1, nways * len(fns))
                else:
                    new_allowed_symbols = set()
                    for s in allowed_symbols.intersection(edge[0].allowed_symbols):
                        new_allowed_symbols.add(self.functions[edge[0].target_function][s])

                    step(edge[1], new_allowed_symbols, l + 1, nways)

        step(tree, set(range(self.n_symbols)), 0, 1)
        return res

    def create_plan(self, schema: Dict[str, Any], depth_range: Tuple[int, int]) -> SplitPlan:
        tree = self.generate_rollout_tree(schema, depth_range)
        return SplitPlan(tree, self.length_count(tree))

    def sample_one(self, plan: SplitPlan, depth: int, seed: np.random.RandomState) -> Optional[FunctionTrace]:
        fn_hist = []
        sym_hist = [seed.randint(self.n_symbols)]
        plan = plan.tree

        for i in range(depth):
            weights = [(len(self.groups[r[0]]) if r[0] != "OUT" else 0) if isinstance(r[0], str) else int(sym_hist[-1] in r[0].allowed_symbols) for r in plan]
            weights = np.cumsum(weights)
            if weights[-1] == 0:
                return None

            i = np.random.randint(weights[-1])

            rule_index = bisect.bisect(weights, i)
            fn_index = i - (weights[rule_index-1] if rule_index>0 else 0)

            if isinstance(plan[rule_index][0], str):
                fn = self.groups[plan[rule_index][0]][fn_index]
            else:
                fn = plan[rule_index][0].target_function

            plan = plan[rule_index][1]

            fn_hist.append(fn)
            sym_hist.append(self.functions[fn][sym_hist[-1]])

        for r in plan:
            if isinstance(r[0], RestrictedEdge) and r[0].target_group == "OUT":
                if sym_hist[-1] in r[0].allowed_symbols:
                    break
            elif r[0] == "OUT":
                break
        else:
            return None

        return self.gen_trace(sym_hist, fn_hist)

    def sample(self, plan: SplitPlan, n: Optional[int], depth: Tuple[int, int], seed: np.random.RandomState,
               fill_factor: Optional[float] = None) -> List[FunctionTrace]:
        valid_lengths = [l for l in sorted(plan.counts.keys()) if depth[0] <= l <= depth[1]]

        fill_factor = fill_factor or self.fill_factor
        res = []

        counts = {}
        csum = 0
        for i, l in enumerate(valid_lengths):
            lens_rm = len(valid_lengths) - i
            c = int(plan.counts[l] * fill_factor)
            if n is not None:
                c = min(c, (n - csum + lens_rm - 1) // lens_rm)
            counts[l] = c
            csum += c

        if n is None:
            n = sum(counts.values())
        else:
            assert sum(counts.values()) == n

        pbar = tqdm(total=n)
        for l, c in counts.items():
            def sampler(seed: np.random.RandomState) -> FunctionTrace:
                s = None
                while s is None:
                    s = self.sample_one(plan, l, seed)
                return s

            res = res + rejection_sample(c, seed, sampler, pbar=pbar)

        return res

    def random_split(self, l: List[Any], seed: np.random.RandomState, slices: List[int]):
        indices = seed.permutation(len(l))
        slices = slices + [len(l)]
        return [[l[i] for i in indices[0 if si == 0 else slices[si - 1]: s]] for si, s in enumerate(slices)]

    def __getitem__(self, item: int) -> Dict[str, Any]:
        table_index = bisect.bisect(self.split_offsets, item)
        relative_index = item - (self.split_offsets[table_index-1] if table_index>0 else 0)

        split = self.used_splits[table_index]
        in_seq = self.in_sequences[split][relative_index]

        return {
            "in": np.asarray(self.in_remap(in_seq), np.uint8),
            "out": np.asarray(self.out_remap([self.out_sequences[split][relative_index]]), np.uint8),
            "in_len": len(in_seq),
            "out_len": 1,
            "type": self.split_types[table_index]
        }

    def gen_function_names(self, n: int) -> List[str]:
        strs = string.ascii_letters + string.punctuation
        assert n < len(strs)
        return list(strs[:n])

    def gen_symbol_names(self, n: int) -> List[str]:
        return [str(i) for i in range(n)]

    def gen_functions(self, seed: np.random.RandomState, n_functions: int, n_symbols: int) -> List[Tuple[int]]:
        # Sample functions with diverse set of outputs. Ensure that each output symbol is produced equal number of times
        known = set()
        while len(known) < n_functions:
            known.add(tuple(seed.permutation(int(n_symbols))))

        return [tuple(i % n_symbols for i in p) for p in known]

    def generate_splits(self) -> Dict[str, List[FunctionTrace]]:
        raise NotImplementedError()

    def generate_dataset(self) -> Tuple[List[int], List[int]]:
        in_sentences = []
        out_sentences = []

        splits = self.generate_splits()

        print("Postprocessing...")

        in_sentences = {k: parallel_map(v, lambda s: self.in_vocabulary(s.expr)) for k, v in splits.items()}
        out_sentences = {k: [a.res for a in v] for k, v in splits.items()}
        print("Done.")

        return in_sentences, out_sentences

    def init_data_tables(self, seed: np.random.RandomState):
        pass

    def __init__(self, split: Union[str, List[str]], depth: Tuple[int, int], n_symbols: int, n_functions: int,
                       N: int = 100000, reversed: bool = False, shared_vocabulary: bool = False, seed: int = 0,
                       cache_dir: str = "./cache/"):

        self.cache_dir = cache_dir
        self.N = N
        self.depth = depth
        self.n_symbols = n_symbols
        self.n_functions = n_functions
        self.reversed = reversed

        self.seed = seed
        seed = np.random.RandomState(seed)
        self.fn_names = self.gen_function_names(n_functions)
        self.symbol_names = self.gen_symbol_names(n_symbols)
        self.functions = self.gen_functions(seed, n_functions, n_symbols)
        self.init_data_tables(seed)

        if self.in_vocabulary is None:
            self.in_vocabulary = framework.data_structures.WordVocabulary(self.fn_names+self.symbol_names)
            self.out_vocabulary = framework.data_structures.WordVocabulary(self.symbol_names)

        if isinstance(split, str):
            split = [split]

        self.in_sequences, self.out_sequences = self.load_cache()
        self.all_splits = list(sorted(self.in_sequences.keys()))
        self.used_splits = []
        for s in split:
            for name in self.in_sequences.keys():
                if re.match(f"^{s}$", name):
                    self.used_splits.append(name)
        self.split_offsets = np.cumsum([len(self.in_sequences[s]) for s in self.used_splits])
        self.split_types = [self.all_splits.index(s) for s in self.used_splits]

        self.max_in_len = max((max(len(i) for i in s) if s else 0) for s in self.in_sequences.values())

        if shared_vocabulary:
            iv = self.in_vocabulary
            ov = self.out_vocabulary
            self.in_vocabulary = iv + ov
            self.out_vocabulary = self.in_vocabulary
            imap = self.in_vocabulary.mapfrom(iv)
            omap = self.out_vocabulary.mapfrom(ov)
            self.in_remap = lambda s: [imap[x] for x in s]
            self.out_remap = lambda s: [omap[x] for x in s]
        else:
            self.in_remap = lambda x: x
            self.out_remap = lambda x: x


    def get_output_size(self) -> int:
        return len(self.out_vocabulary)

    def get_input_size(self) -> int:
        return len(self.in_vocabulary)

    @property
    def max_out_len(self) -> int:
        return 1

    def __len__(self) -> int:
        return self.split_offsets[-1]

    def start_test(self) -> TextSequenceTestState:
        return TextSequenceTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                     lambda x: " ".join(self.out_vocabulary(x)), type_names=self.all_splits)


class FunctionInterfaceTestClassificationMixin:
    def __getitem__(self, item: int) -> Dict[str, Any]:
        res = super().__getitem__(item)
        res["out"] = res["out"][0].item()
        return res

    def start_test(self) -> TextClassifierTestState:
        return TextClassifierTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                        lambda x: self.out_vocabulary([x])[0],
                                        type_names = self.all_splits, max_good_samples=100, max_bad_samples=100)
