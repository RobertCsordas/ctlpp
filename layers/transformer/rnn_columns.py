from typing import Optional
import torch
import torch.nn
from .transformer import ActivationFunction
import numpy as np


class UniversalTransformerRandomLayerEncoder(torch.nn.Module):
    def __init__(self, layer, n_layers: int, n_extra: int, n_test: Optional[int] = None, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, **kwargs)
        self.n_extra = n_extra
        self.n_layers = n_layers
        self.n_test = n_test

    def set_n_layers(self, n_layers: int):
        self.layers = [self.layer] * n_layers

    def forward(self, data: torch.Tensor, *args, **kwargs):
        self.set_n_layers(np.random.randint(self.n_layers, self.n_extra + self.n_layers + 1) if self.training else \
                          (self.n_test or self.n_layers))
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data


def UniversalTransformerRandomLayerEncoderWithLayer(layer):
    return lambda *args, **kwargs: UniversalTransformerRandomLayerEncoder(layer, *args, **kwargs)
