from matplotlib.pyplot import plot
from .transformer_classifier_mixin import TransformerClassifierMixin
from .simple_task import SimpleTask
from typing import Dict, Any, List, Union, Optional
from .. import task, args
import torch
import framework
import dataset
import numpy as np
from sklearn.decomposition import PCA
from dataset.function_interface_test_base import FunctionTrace

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


@task(name="fit_parallel_branch_trafo_classifier")
class FITParallelBranchTrafoClassifier(TransformerClassifierMixin, SimpleTask):
    VALID_NUM_WORKERS = 0

    def get_loader(self, split: Union[str, List[str]]):
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

    def create_model(self) -> torch.nn.Module:
        model = super().create_model()
        self.record_raw_output = False
        self.raw_output_all_cols = None
        self.recorded_grads = None
        model.output_map.register_forward_hook(self.save_raw_representation_hook)
        model.trafo.layer.register_forward_hook(self.save_all_cols_hook)
        model.trafo.layer.register_full_backward_hook(self.save_all_col_grads)
        return model

    def save_all_col_grads(self, module, grad_input, grad_output):
        if self.record_raw_output:
            if self.gstep == 8:
                self.recorded_grads = grad_input[0]
            self.gstep -= 1

    def save_all_cols_hook(self, module, input, output):
        if self.record_raw_output:
            if self.record_curr_step == 8:
                self.raw_output_all_cols = input[0].detach()

            self.record_curr_step += 1

    def save_raw_representation_hook(self, module, input, output):
        if self.record_raw_output:
            self.raw_output = input[0].detach()

    def batch_from_traces(self, traces: List[FunctionTrace]) -> Dict[str, Any]:
        batch = [self.train_set.in_vocabulary(t.expr) for t in traces]
        res = [self.train_set.out_vocabulary(self.train_set.symbol_names[t.res]) for t in traces]
        batch = torch.tensor(batch, dtype=torch.long, device=self.helper.device)

        return {"in": batch.T,
            "in_len": torch.tensor([len(b) for b in batch], dtype=torch.long, device=self.helper.device),
            "out": torch.tensor(res, device=self.helper.device, dtype=torch.long)
        }

    def prepare_validation_data(self):
        if self.prepared:
            return

        self.prepared = True

        traces = []
        self.fn_test_index_table = {}
        fn_test_functions = []
        fn_test_symbols = []
        fn_test_in_symbols = []
        for f in range(self.train_set.n_functions):
            for s in range(self.train_set.n_symbols):
                traces.append(self.train_set.trace_from_input(s, [f]))
                self.fn_test_index_table[traces[-1].res, f] = len(traces) - 1
                fn_test_functions.append(f)
                fn_test_symbols.append(traces[-1].res)
                fn_test_in_symbols.append(s)

        self.fn_test_functions = torch.tensor(fn_test_functions, device = self.helper.device)
        self.fn_test_symbols = torch.tensor(fn_test_symbols, device = self.helper.device)
        self.fn_test_in_symbols = torch.tensor(fn_test_in_symbols, device = self.helper.device)

        self.fn_test_data = self.batch_from_traces(traces)


        self.double_fn_index = {}
        traces = []
        for s in range(self.train_set.n_symbols):
            for f1 in range(self.train_set.n_functions):
                for f2 in range(self.train_set.n_functions):
                    traces.append(self.train_set.trace_from_input(s, [f1, f2]))
                    self.double_fn_index[f1, f2, traces[-1].vals[1]] = len(traces) - 1

        self.double_fn_test_data = self.batch_from_traces(traces)

        self.fn_names_with_groupid = {f: self.train_set.fn_names[f] for f in range(self.train_set.n_functions)}
        for gi, g in self.train_set.groups.items():
            for f in g:
                self.fn_names_with_groupid[f] = f"{gi}{self.fn_names_with_groupid[f]}"


    def cos_dist(self, d: torch.Tensor) -> torch.Tensor:
        norm = d.norm(dim=-1).clamp(min=1e-6)
        return (d @ d.T) / norm.unsqueeze(-1) / norm

    def l2_dist(self, d: torch.Tensor) -> torch.Tensor:
        return (d.unsqueeze(-1) - d.T).norm(dim=1)

    def plot_symbol_similarity_matrix(self, symbol: int, raw_data: torch.Tensor, fn_baselines: Optional[torch.Tensor] = None) -> framework.visualize.plot.Clustermap:
        reps = torch.stack([raw_data[self.fn_test_index_table[symbol, f]] - (0 if fn_baselines is None else fn_baselines[f]) for f in range(self.train_set.n_functions)], dim=0)
        marks = [self.fn_names_with_groupid[f] for f in range(self.train_set.n_functions)]

        cosdist = self.cos_dist(reps)

        return framework.visualize.plot.Clustermap(cosdist, None, marks=marks, textval=False, range=(0,1), show_clusters=True)


    def plot_fn_variation_removed(self, data: torch.Tensor, grad: torch.Tensor):
        res = {}
        all_pairs = []
        for f in range(self.train_set.n_functions):
            syms_per_fn = data[self.fn_test_functions == f]

            pairs = syms_per_fn.unsqueeze(-1) - syms_per_fn.T
            pairs = pairs.transpose(1, 2).flatten(end_dim=1)

            all_pairs.append(pairs)

        data = data.cpu()

        all_pairs = torch.cat(all_pairs, 0)
        pca = PCA(32)
        pca.fit(grad.cpu())

        raw_lowd = pca.transform(data)

        marks = []
        for f, s, si in zip(self.fn_test_functions, self.fn_test_symbols, self.fn_test_in_symbols):
            marks.append(f"{self.fn_names_with_groupid[f.item()]}{self.train_set.symbol_names[si.item()]}->{self.train_set.symbol_names[s.item()]}")

        for i in {8}:
            lr = torch.tensor(raw_lowd[:, :i])
            res[f"lowd_cos_dim_{i}"] = framework.visualize.plot.Clustermap(self.cos_dist(lr), None, marks=marks, textval=False, figsize=(20,20), colorbar=False, ticksize=6, xmarks=False)

            for s in range(self.train_set.n_symbols):
                res[f"symbol_{self.train_set.symbol_names[s]}_dim_{i}"] = \
                    self.plot_symbol_similarity_matrix(s, lr)
        return res

    def analyze_cross_group_performance(self, symbol: int, double_fn_correct: torch.Tensor) -> framework.visualize.plot.Heatmap:
        double_fn_correct = double_fn_correct.cpu()

        res = {}
        gnames = set()
        for gname1, g1 in self.train_set.groups.items():
            if len(g1) == 0:
                continue

            gnames.add(gname1)

            for gname2, g2 in self.train_set.groups.items():
                if len(g2) == 0:
                    continue

                n_ok = 0
                n_total = 0

                for f1 in g1:
                    for f2 in g2:
                        n_ok = n_ok + double_fn_correct[self.double_fn_index[f1, f2, symbol]].item()
                        n_total += 1

                res[gname1, gname2] = n_ok / n_total
                

                # print(f"Perf for {s}, {c}, {gname} is {n_ok/n_total}, (total: {n_total})")

        gnames = list(sorted(gnames))
        mres = np.empty([len(gnames), len(gnames)], dtype=np.float32)

        for gi1, g1 in enumerate(gnames):
            for gi2, g2 in enumerate(gnames):
                mres[gi1, gi2] = res[g1, g2]

        return framework.visualize.plot.Heatmap(mres, x_marks=gnames, y_marks=gnames, textval=True, range=(0, 1), figsize=(2,2), round_decimals=2)

    def analyze_cluster_performance(self, symbol: int, double_fn_correct: torch.Tensor, cm: framework.visualize.plot.Clustermap) -> framework.visualize.plot.Heatmap:
        n_clusters = cm.idx_to_cluster.max()

        double_fn_correct = double_fn_correct.cpu()

        res = {}
        gnames = set()
        for c in range(n_clusters):
            for gname, g in self.train_set.groups.items():
                if len(g) == 0:
                    continue
                n_ok = 0
                n_total = 0

                for f1 in range(self.train_set.n_functions):
                    if cm.idx_to_cluster[f1] != c + 1:
                        continue

                    # f at this point is iterating over functions in cluster c
                    for f2 in g:
                        # f2 is iterating over all functions from group g
                        n_ok = n_ok + double_fn_correct[self.double_fn_index[f1, f2, symbol]].item()
                    n_total += len(g)

                res[c, gname] = n_ok / n_total
                gnames.add(gname)

        gnames = list(sorted(gnames))
        mres = np.empty([n_clusters, len(gnames)], dtype=np.float32)
        for c in range(n_clusters):
            for gi, g in enumerate(gnames):
                mres[c, gi] = res[c, g]


        return framework.visualize.plot.Heatmap(mres, x_marks=gnames, y_marks=[str(i) for i in range(n_clusters)], textval=True, range=(0, 1), figsize=(2,2), round_decimals=2)


    def plot_pair_compatibility(self, symbol: int, double_fn_correct: torch.Tensor) -> framework.visualize.plot.Heatmap:
        mres = np.empty([self.train_set.n_functions, self.train_set.n_functions], dtype=np.float32)
        for f1 in range(self.train_set.n_functions):
            for f2 in range(self.train_set.n_functions):
                mres[f1, f2] = double_fn_correct[self.double_fn_index[f1, f2, symbol]].item()
        
        return framework.visualize.plot.Heatmap(mres, x_marks=self.train_set.fn_names, y_marks=self.train_set.fn_names, textval=False, range=(0, 1))


    def validate(self) -> Dict[str, Any]:
        res = super().validate()

        plot_now = (self.helper.args.debug_plot_interval is not None) \
                   and ((self.helper.state.iter // self.helper.args.test_interval) % \
                        self.helper.args.debug_plot_interval == 0)

        if not plot_now:
            return res

        self.prepare_validation_data()

        self.model.eval()

        with torch.no_grad():
            d = self.prepare_data(self.double_fn_test_data)
            r, _ = self.run_model(d)
            double_fn_correct = (r.outputs.argmax(-1) == self.double_fn_test_data["out"][:,0]).float()

        allgrad = []
        self.record_raw_output = True
        self.record_curr_step = 0

        # with torch.no_grad():
        for s in range(self.train_set.n_symbols):
            self.gstep = len(self.model.trafo.layers) - 1
            d = self.fn_test_data.copy()
            d["out"] = torch.full_like(d["out"], s)
            d = self.prepare_data(d)
            r, _ = self.run_model(d)
            r.loss.backward()
            allgrad.append(self.recorded_grads)
            self.recorded_grads = None
            assert self.gstep == -1
         
        self.model.train()
        self.record_raw_output = False

        rawdata = self.model.get_result(self.raw_output_all_cols, d["in_len"], 1)
        grad = torch.cat([self.model.get_result(g, d["in_len"], 1) for g in allgrad], 0) if allgrad[0] is not None else None


        vproj = self.model.trafo.layer.att.multi_head_merge(self.model.trafo.layer.att.data_to_kv(rawdata)[:, rawdata.shape[-1]:]) if hasattr(self.model.trafo.layer, "att") else None

        for s in range(self.train_set.n_symbols):
            sm = res[f"representation_cos_distance/symbol_{self.train_set.symbol_names[s]}"] = \
                self.plot_symbol_similarity_matrix(s, self.raw_output)

            
            res[f"representation_cos_distance_cluster_error/symbol_{self.train_set.symbol_names[s]}"] = self.analyze_cluster_performance(s, double_fn_correct, sm)

            res[f"representation_cos_distance_cluster_error/symbol_{self.train_set.symbol_names[s]}"] = self.analyze_cluster_performance(s, double_fn_correct, sm)

            res[f"representation_cos_distance_group_error/symbol_{self.train_set.symbol_names[s]}"] = self.analyze_cross_group_performance(s, double_fn_correct)
                

            res[f"representation_cos_distance_col/symbol_{self.train_set.symbol_names[s]}"] = \
                self.plot_symbol_similarity_matrix(s, rawdata)

            if vproj is not None:
                res[f"representation_cos_distance_col_vproj/symbol_{self.train_set.symbol_names[s]}"] = \
                    self.plot_symbol_similarity_matrix(s, vproj)

        all_sim = self.cos_dist(rawdata)
        marks = []
        for f, s, si in zip(self.fn_test_functions, self.fn_test_symbols, self.fn_test_in_symbols):
            marks.append(f"{self.fn_names_with_groupid[f.item()]}{self.train_set.symbol_names[si.item()]}->{self.train_set.symbol_names[s.item()]}")

        res["all_sim/actual_col"] = framework.visualize.plot.Clustermap(all_sim, None, marks=marks, textval=False, range=(0,1), figsize=(20,20), colorbar=False, ticksize=6, xmarks=False)

        if vproj is not None:
            all_sim = self.cos_dist(vproj)
            res["all_sim/actual_vproj"] = framework.visualize.plot.Clustermap(all_sim, None, marks=marks, textval=False, range=(0,1), figsize=(20,20), colorbar=False, ticksize=6, xmarks=False)

        all_sim = self.cos_dist(self.raw_output)
        res["all_sim/out_col"] = framework.visualize.plot.Clustermap(all_sim, None, marks=marks, textval=False, range=(0,1), figsize=(20,20), colorbar=False, ticksize=6)

        if grad is not None:
            res.update({f"col_out_lowd/{k}": v for k, v in self.plot_fn_variation_removed(rawdata, grad).items()})

        self.raw_output = None
        self.raw_output_all_cols = None
        self.recorded_grads = None

        if self.raw_data_to_save is None:
            self.raw_data_to_save = {}

        for k, v in res.items():
            if not hasattr(v, "raw_state"):
                continue

            s = v.raw_state()
            if s:
                self.raw_data_to_save[k] = s


        return res
