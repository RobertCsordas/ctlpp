import torch.nn
from layers.transformer import RelativeTransformerEncoderLayer
from layers.transformer.rnn_columns import UniversalTransformerRandomLayerEncoderWithLayer
from layers.transformer.ndr_experimental import NDRResidualCoreInitGelu
from models import TransformerClassifierModel
from interfaces import TransformerClassifierInterface
from .. import args
import framework
import functools
from interfaces import Result
from layers import LayerVisualizer
from typing import Dict, List, Tuple, Any


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-transformer_classifier.result_column", default="first", choice=["first", "last"])
    parser.add_argument("-transformer_classifier.autoregressive", default=False)
    parser.add_argument("-ndr.scalar_gate", default=False)
    parser.add_argument("-debug_plot_interval", default="1", parser=parser.int_or_none_parser)
    parser.add_argument("-embedding_init", default="auto", choice=["auto", "kaiming", "xavier", "pytorch"])
    parser.add_argument("-geometric.has_range", default=False)
    parser.add_argument("-trafo_classifier.out_mode", default="linear", choice=["linear", "tied", "attention"])
    parser.add_argument("-trafo_classifier.norm_att", default=False)
    parser.add_argument("-universal.n_test_layers", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-ndr.drop_gate", default=0.0)
    parser.add_argument("-ndr.gate_size_multiplier", default=1.0)
    parser.add_argument("-universal.n_rand_layers", default=0)
    parser.add_argument("-ndr.exclude_diagonal", default=True)


class TransformerClassifierMixin:
    VIS_DATASET_FILTER = None

    def get_config(self):
        einit = None if self.helper.args.embedding_init == "auto" else self.helper.args.embedding_init
        rel_args = dict(pos_embeddig=(lambda x, offset: x), embedding_init=einit or "xavier")
        default_init = einit or "pytorch"

        def make_universal(layer):
            return functools.partial(UniversalTransformerRandomLayerEncoderWithLayer(layer),
                                    n_extra=self.helper.args.universal.n_rand_layers,
                                    n_test=self.helper.args.universal.n_test_layers)

        trafos = {
            "relative_universal": (make_universal(RelativeTransformerEncoderLayer), rel_args),
            "ndr_residual_core_gelu": (functools.partial(make_universal(NDRResidualCoreInitGelu), gate_size_multiplier= self.helper.args.ndr.gate_size_multiplier, normalize_score=self.helper.args.trafo_classifier.norm_att, scalar_gate = self.helper.args.ndr.scalar_gate, has_range = self.helper.args.geometric.has_range, p_gate_drop=self.helper.args.ndr.drop_gate, exclude_diagonal=self.helper.args.ndr.exclude_diagonal), dict(pos_embeddig=(lambda x, offset: x), embedding_init=default_init)),            
        }

        return trafos[self.helper.args.transformer.variant]

    def create_model(self) -> torch.nn.Module:
        constructor, args = self.get_config()
        model = TransformerClassifierModel(len(self.train_set.in_vocabulary),
                                      len(self.train_set.out_vocabulary), self.helper.args.state_size,
                                      nhead=self.helper.args.transformer.n_heads,
                                      n_layers=self.helper.args.transformer.encoder_n_layers,
                                      ff_multiplier=self.helper.args.transformer.ff_multiplier,
                                      transformer=constructor,
                                      result_column=self.helper.args.transformer_classifier.result_column,
                                      eos=self.helper.args.eos,
                                      sos=self.helper.args.sos,
                                      autoregressive=self.helper.args.transformer_classifier.autoregressive,
                                      dropout=self.helper.args.dropout,
                                      attention_dropout=self.helper.args.transformer.attention_dropout,
                                      out_mode=self.helper.args.trafo_classifier.out_mode,
                                      embedding_size=self.helper.args.embedding_size, **args)

        self.visualizer = LayerVisualizer(model, {"mha.plot_head_details": True})
        self.validation_started_on = None
        self.raw_data_to_save = None
        return model

    def create_model_interface(self):
        self.model_interface = TransformerClassifierInterface(self.model, label_smoothing=self.helper.args.label_smoothing)

    def get_steplabels(self, data: Dict[str, torch.Tensor]) -> List[str]:
        s = self.train_set.in_vocabulary(data["in"][:, 0].cpu().numpy().tolist())
        s = (["B"] if self.helper.args.sos else []) + list(s)

        if self.helper.args.eos:
            s.append("-")
            epos = data["in_len"][0].int().item() + int(self.helper.args.sos)
            s[epos] = "E"
            for p in range(epos+1, len(s)):
                s[p] = ""

        return s

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        if (self.VIS_DATASET_FILTER is None) or (name in self.VIS_DATASET_FILTER):
            self.validation_started_on = name
        return super().validate_on_name(name)

    def run_model(self, data: Dict[str, torch.Tensor]) -> Tuple[Result, Dict[str, Any]]:
        plots = {}
        plot_now = (self.helper.args.debug_plot_interval is not None) and (self.validation_started_on is not None) \
                   and ((self.helper.state.iter // self.helper.args.test_interval) % \
                        self.helper.args.debug_plot_interval == 0)

        if plot_now:
            s = self.get_steplabels(data)
            self.visualizer.prepare({"steplabel": s})

        res = self.model_interface(data)

        if plot_now:
            plots.update({f"activations/{k}": v for k, v in self.visualizer.plot().items()})

            self.raw_data_to_save = {n: o.map for n, o in plots.items()
                                    if isinstance(o, framework.visualize.plot.AnimatedHeatmap)}
            self.raw_data_to_save["steplabels"] = s

        if (self.validation_started_on is not None):
            plots = {f"validation_plots/{self.validation_started_on}/{k}": v for k, v in plots.items()}
            self.helper.log(plots)
            plots = {}
            self.validation_started_on = None

        return res, plots

    def finish(self):
        print("Saving raw plots")
        if self.raw_data_to_save is not None:
            for k, v in self.raw_data_to_save.items():
                print(f"   Saving {k}")
                self.helper.export_tensor(f"raw_plots/{k}", v)
        print("Done.")
