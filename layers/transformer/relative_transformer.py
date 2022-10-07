
from typing import Optional
import torch
import torch.nn
import torch.nn.functional as F
from .transformer import ActivationFunction
from .multi_head_relative_pos_attention import FixedRelativeMultiheadAttention, AttentionMask


class RelativeTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu,
                 attention_dropout=0, test_pos_clamp: Optional[int] = None, drop_expand: bool = True):
        super().__init__()
        self.self_attn = FixedRelativeMultiheadAttention(d_model, nhead, dropout=attention_dropout,
                                                         test_pos_clamp=test_pos_clamp)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout) if drop_expand else lambda x: x
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None) -> torch.Tensor:
        src2 = self.self_attn(src, attend_to if attend_to is not None else src, mask, pos_offset=pos_offset)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu') \
           if self.activation is F.relu else 1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

