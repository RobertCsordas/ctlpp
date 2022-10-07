import torch
import torch.nn
from .transformer import TransformerEncoderLayer


class UniversalTransformerEncoder(torch.nn.Module):
    def __init__(self, layer, n_layers: int, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, **kwargs)
        self.set_n_layers(n_layers)

    def set_n_layers(self, n_layers: int):
        self.layers = [self.layer] * n_layers

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data


def UniversalTransformerEncoderWithLayer(layer=TransformerEncoderLayer):
    return lambda *args, **kwargs: UniversalTransformerEncoder(layer, *args, **kwargs)
