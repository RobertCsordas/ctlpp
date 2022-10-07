import torch
import torch.nn
import torch.nn.functional as F
from .multi_head_attention import MultiHeadAttention, AttentionMask
from typing import Optional, Callable, Dict
from dataclasses import dataclass
# This file is based on PyTorch's internal implementation

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu,
                 attention_dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=attention_dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu') \
            if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, layer, n_layers: int, *args, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer(*args, **kwargs) for _ in range(n_layers)])

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data

def TransformerEncoderWithLayer(layer=TransformerEncoderLayer):
    return lambda *args, **kwargs: TransformerEncoder(layer, *args, **kwargs)
