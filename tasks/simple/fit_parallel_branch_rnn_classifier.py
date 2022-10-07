from .simple_task import SimpleTask
from typing import List, Union
from .. import task, args
import framework
import dataset
from .sequence_classifier_mixin import SequenceClassifierMixin

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-fit.max_depth", default=6)
    parser.add_argument("-fit.n_symbols", default=8)
    parser.add_argument("-fit.n_functions", default=16)
    parser.add_argument("-fit.reversed", default=False)
    parser.add_argument("-fit.N", default=100000)
    parser.add_argument("-fit.overlap", default=0)
    parser.add_argument("-fit.symbol_overlap", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-fit.train_on_pairs", default=False)
    parser.add_argument("-fit.variant", default="parallel_branch", choice=["parallel_branch", "parallel_branch_2stage", "alternate"])
    parser.add_argument("-fit.imbalance", default=0)


@task(name="fit_parallel_branch_rnn_classifier")
class FITParallelBranchRNNClassifier(SequenceClassifierMixin, SimpleTask):
    VALID_NUM_WORKERS = 0

    def get_loader(self, split: Union[str, List[str]]):
        # return dataset.FunctionInterfaceTestParallelBranchClassification(split, (2, self.helper.args.fit.max_depth),
        #         n_symbols=self.helper.args.fit.n_symbols, n_functions=self.helper.args.fit.n_functions,
        #         N=self.helper.args.fit.N,
        #         reversed=self.helper.args.fit.reversed, overlap=self.helper.args.fit.overlap,
        #         symbol_overlap=self.helper.args.fit.symbol_overlap, invert_debug=self.helper.args.fit.invert_debug,
        #         exclude_self_loops=self.helper.args.fit.exclude_self_loops, n_3rd_group=self.helper.args.fit.n_3rd_group)

        if self.helper.args.fit.variant=="parallel_branch":
            return dataset.FunctionInterfaceTestParallelBranchClassification(split, (2, self.helper.args.fit.max_depth),
                n_symbols=self.helper.args.fit.n_symbols, n_functions=self.helper.args.fit.n_functions,
                N=self.helper.args.fit.N,
                reversed=self.helper.args.fit.reversed, overlap=self.helper.args.fit.overlap,
                symbol_overlap=self.helper.args.fit.symbol_overlap)
        elif self.helper.args.fit.variant=="parallel_branch_2stage":
            return dataset.FunctionInterfaceTestParallelBranch2StageClassification(split, (2, self.helper.args.fit.max_depth),
                n_symbols=self.helper.args.fit.n_symbols, n_functions=self.helper.args.fit.n_functions,
                N=self.helper.args.fit.N,
                reversed=self.helper.args.fit.reversed, overlap=self.helper.args.fit.overlap,
                symbol_overlap=self.helper.args.fit.symbol_overlap, share_imbalance=self.helper.args.fit.imbalance)
        elif self.helper.args.fit.variant=="alternate":
            return dataset.FunctionInterfaceTestAlternateClassification(split, (2, self.helper.args.fit.max_depth),
                n_symbols=self.helper.args.fit.n_symbols, n_functions=self.helper.args.fit.n_functions,
                N=self.helper.args.fit.N,
                reversed=self.helper.args.fit.reversed)
        else:
            assert False, "Invalid variant"


    def create_datasets(self):
        self.prepared = False
        self.batch_dim = 1
        self.train_set = self.get_loader(["train"] + (["pairs"] if self.helper.args.fit.train_on_pairs else []))
        self.valid_sets.test = self.get_loader(["test_.*", "pairs"])
        self.valid_sets.iid = self.get_loader("iid")
