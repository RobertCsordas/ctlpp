from typing import Tuple, List, Union, Optional, Dict
import numpy as np
from .function_interface_test_base import FunctionInterfaceTestClassificationMixin, FunctionInterfaceTestBase, FunctionTrace


class FunctionInterfaceTestAlternate(FunctionInterfaceTestBase):
    VERSION = 3

    def config_id(self) -> str:
        return f"{self.N}_{self.depth[0]}_{self.depth[1]}_{self.seed}_{self.n_symbols}_{self.n_functions}_{self.reversed}"

    def init_data_tables(self, seed: np.random.RandomState):
        self.generate_groups(
            {
                "sets": {"a", "b"},
            }, seed
        )

        self.train_plan = self.create_plan({
            "edges": [
                ["IN", {"a", "b"}],
                ["a", "b"],
                ["b", "a"],
                [{"a", "b"}, "OUT"],
            ]
        }, self.depth)

    def generate_splits(self) -> Dict[str, List[FunctionTrace]]:
        splits = {}

        test_depth = (max(2, self.depth[0]), self.depth[1])

        print(f"Generating {self.__class__.__name__} with {self.N} elements")

        seed = np.random.RandomState(self.seed)

        samples = self.sample(self.train_plan, self.N + 1000, self.depth, seed)
        splits["iid"], splits["train"] = self.random_split(samples, seed, [1000])

        splits["test_self_loop"] = self.sample(self.create_plan({
            "edges": [
                ["IN", {"a", "b"}],
                ["a", "a"],
                ["b", "b"],
                [{"a", "b"}, "OUT"]
            ]
        }, test_depth), 1000, test_depth, seed, fill_factor=1)

        splits["test_all"] = self.sample(self.create_plan({
            "edges": [
                [{"a", "b", "IN"}, {"a", "b", "OUT"}]
            ]
        }, test_depth), 1000, test_depth, seed, fill_factor=1)

        splits["pairs"] = self.sample(self.train_plan, None, (2, 2), seed, fill_factor=1)
        splits["test_all_pairs"] = self.sample(self.create_plan({
            "edges": [
                [{"a", "b", "IN"}, {"a", "b", "OUT"}]
            ]
        }, (2, 2)), None, (2, 2), seed, fill_factor=1)

        return splits

    def __init__(self, split: Union[str, List[str]], depth: Tuple[int, int], n_symbols: int, n_functions: int,
                       N: int = 100000, reversed: bool = False, shared_vocabulary: bool = False, seed: int = 0,
                       fill_factor: float = 0.5, cache_dir: str = "./cache/"):

        self.fill_factor = fill_factor

        super().__init__(split, depth, n_symbols, n_functions, N, reversed, shared_vocabulary, seed, cache_dir)


class FunctionInterfaceTestAlternateClassification(FunctionInterfaceTestClassificationMixin, FunctionInterfaceTestAlternate):
    pass
