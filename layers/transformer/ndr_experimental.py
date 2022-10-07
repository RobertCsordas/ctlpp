import torch
import torch.nn
import torch.nn.functional as F
from .multi_head_relative_pos_attention import AttentionMask
from typing import Optional, Dict, Any
from .direction_sensitive_geometric import DirectionSensitiveGeometricAttention
from .transformer import ActivationFunction
from .ndr import NDRBase



class NDRResidualCoreInitGelu(NDRBase):
    def __init__(self, d_model: int, nhead: int, dropout: float, scalar_gate: bool = False, has_range: bool = False,
                 attention_dropout=0, p_gate_drop=0.05, dim_feedforward: Optional[int] = None,
                 normalize_score=True, gate_size_multiplier=1, exclude_diagonal=True, pos_encoding="direction", **kwargs):
        super().__init__()

        self.reg_loss = 0

        dim_feedforward = dim_feedforward or (4*d_model)
        self.att = DirectionSensitiveGeometricAttention(d_model, nhead, dropout=attention_dropout, range_select=has_range, normalize_score=normalize_score,
                                                             exclude_diagonal=exclude_diagonal, position=pos_encoding)

        self.p1 = torch.nn.Linear(d_model, dim_feedforward)
        self.p2 = torch.nn.Linear(dim_feedforward, d_model)

        self.g1 = torch.nn.Linear(d_model, int(d_model * gate_size_multiplier))
        self.g2 = torch.nn.Linear(int(d_model * gate_size_multiplier), 1 if scalar_gate else d_model)

        self.nmerge = torch.nn.LayerNorm(d_model)
        self.no = torch.nn.LayerNorm(d_model)

        self.drop = torch.nn.Dropout(dropout)

        self.g2.bias.data.fill_(-3)
        self.p_gate_drop = p_gate_drop

        self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None, attend_to: Optional[torch.Tensor] = None,
                pos_offset: Optional[int] = None, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        input = self.att(src, attend_to if attend_to is not None else src, mask, pos_offset=pos_offset)

        net = self.nmerge(src + self.drop(input))

        mid = F.gelu(self.p1(net))
        proj = self.p2(mid)
        proj = self.no(proj + net)

        gate = self.g2(F.gelu(self.g1(net)))
        bgate = torch.sigmoid(gate)

        if self.training and self.p_gate_drop>0:
            bgate = bgate.masked_fill(torch.rand(*bgate.shape[:-1], 1, device=bgate.device, dtype=bgate.dtype) < self.p_gate_drop, 0)

        if self.visualization_enabled:
            self.plot_cache.append(bgate[0])

        src = (src if residual is None else residual) * (1-bgate) + proj * bgate

        return src

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.p1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.p2.weight)

        torch.nn.init.xavier_uniform_(self.g1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.g2.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

