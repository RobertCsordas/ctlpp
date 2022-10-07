from typing import Tuple, List, Union, Optional, Dict
import numpy as np
from .function_interface_test_base import FunctionInterfaceTestBase, FunctionInterfaceTestClassificationMixin, FunctionTrace


class FunctionInterfaceTestParallelBranch2Stage(FunctionInterfaceTestBase):
    VERSION = 6

    def config_id(self) -> str:
        return f"{self.N}_{self.depth[0]}_{self.depth[1]}_{self.seed}_{self.n_symbols}_{self.n_functions}_{self.reversed}_{self.overlap}_{self.symbol_overlap}_{self.share_imbalance}"

    @FunctionInterfaceTestBase.once_per_config
    def print_info(self):
        print("Restricted overlaps:")
        for o in self.restricted_overlaps:
            print(f"     {o[0]}, {o[1]}, {self.fn_names[o[2]]}")

        print("Groups")
        for gname, g in self.groups.items():
            print(f"     {gname}: {[self.fn_names[i] for i in g]}")
        print(f"Train set possible sample counts: {self.train_plan.counts}")

    def init_data_tables(self, seed: np.random.RandomState):
        self.generate_groups(
            {
                "sets": {"a1", "b1", "a2", "o", "b2"},
                "set_sizes": {"o": self.overlap}
            }, seed
        )

        self.restricted_overlaps = self.generate_restricted_overlaps(["a1", "b1"], "o", seed, imbalance=self.share_imbalance)

        self.train_plan = self.create_plan({
            "edges": [
                ["IN", {"a1", "b1"}],
                ["a1", "a2"],
                ["b1", "b2"],
                [{"a2", "b2", "o"}, {"a1", "b1", "OUT"}]
            ],

            "restricted_edges": self.restricted_overlaps
        }, self.depth)

        self.print_info()

    def generate_splits(self) -> Dict[str, List[FunctionTrace]]:
        splits = {}

        test_depth = (max(2, self.depth[0]), self.depth[1])

        print(f"Generating {self.__class__.__name__} with {self.N} elements")

        seed = np.random.RandomState(self.seed)

        samples = self.sample(self.train_plan, self.N + 1000, self.depth, seed)
        splits["iid"], splits["train"] = self.random_split(samples, seed, [1000])

        splits["test_group_alternate"] = self.sample(self.create_plan({
            "edges": [
                ["IN", {"a1", "b1"}],
                ["a1", "b2"],
                ["b1", "a2"],
                [{"a2", "b2"}, {"a1", "b1", "OUT"}]
            ]
        }, test_depth), 1000, test_depth, seed, fill_factor=1)


        splits["test_self_loop"] = self.sample(self.create_plan({
            "edges": [
                ["IN", {"a1", "b1", "a2", "b2", "o"}],
                ["a1", "a1"],
                ["b1", "b1"],
                ["a2", "a2"],
                ["b2", "b2"],
                ["o", "o"],
                [{"a1", "b1", "a2", "b2", "o"}, "OUT"],
            ]
        }, test_depth), 1000, test_depth, seed, fill_factor=1)

        splits["test_all"] = self.sample(self.create_plan({
            "edges": [
                [{"IN", "a1", "b1", "a2", "b2", "o"}, {"a1", "b1", "a2", "b2", "o", "OUT"}],
            ]
        }, test_depth), 1000, test_depth, seed, fill_factor=1)

        if self.overlap > 0:
            splits["test_all_overlap"] = self.sample(self.create_plan({
                "edges": [
                    ["IN", {"a1", "b1"}],
                    [{"a1", "b1"}, "o"],
                    ["o", {"a1", "b1", "OUT"}]
                ]
            }, test_depth), 1000, test_depth, seed, fill_factor=1)

            splits["test_cross_unseen_symbols"] = self.sample(self.create_plan({
                "edges": [
                    ["IN", {"a1", "b1"}],
                    ["o", "OUT"]
                ],
                "restricted_edges": self.invert_restricted_overlaps(self.restricted_overlaps)
            }, (2,2)), None, (2,2), seed, fill_factor=1)

        splits["test_unseen_pairs"] = self.sample(self.create_plan({
             "edges": [
                ["IN", {"a1", "b1"}],
                ["a1", "b2"],
                ["b1", "a2"],
                [{"a2", "b2", "o"}, {"OUT"}]
            ],
            "restricted_edges": self.invert_restricted_overlaps(self.restricted_overlaps)
        }, (2,2)), None, (2,2), seed, fill_factor=1)

        splits["pairs"] = self.sample(self.train_plan, None, (2,2), seed, fill_factor=1)
        splits["test_all_pairs"] = self.sample(self.create_plan({
            "edges": [
                ["IN", {"a1", "b1"}],
                [{"a1", "b1"}, {"a2", "b2", "o"}],
                [{"a2", "b2", "o"}, "OUT"]
            ]
        }, (2,2)), None, (2,2), seed, fill_factor=1)

        return splits

    def __init__(self, split: Union[str, List[str]], depth: Tuple[int, int], n_symbols: int, n_functions: int,
                       N: int = 100000, overlap: int = 0, symbol_overlap: Optional[int] = None,
                       reversed: bool = False, shared_vocabulary: bool = False, seed: int = 0, fill_factor: float = 0.5,
                       share_imbalance: int = 0, cache_dir: str = "./cache/"):

        self.overlap = overlap
        self.symbol_overlap = symbol_overlap or n_symbols
        self.fill_factor = fill_factor
        self.share_imbalance = share_imbalance

        super().__init__(split, depth, n_symbols, n_functions, N, reversed, shared_vocabulary, seed, cache_dir)


class FunctionInterfaceTestParallelBranch2StageClassification(FunctionInterfaceTestClassificationMixin, FunctionInterfaceTestParallelBranch2Stage):
    pass
