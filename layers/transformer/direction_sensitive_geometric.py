import torch
import torch.nn.functional as F
from .multi_head_attention import AttentionMask, AttentionMergeMixin
from typing import Optional
from .geometric_attention import geometric_attention_activation
import math
from .multi_head_relative_pos_attention import RelativeAttentionBase


class DirectionSensitiveGeometricAttention(AttentionMergeMixin, RelativeAttentionBase):
    def __init__(self, state_size: int, n_heads: int, dropout: float = 0.0,
                 input_size: Optional[int] = None, range_select: bool = False,
                 output_size: Optional[int] = None, normalize_score: bool = True, exclude_diagonal: bool = True,
                 position: str = "direction"):

        super(AttentionMergeMixin, self).__init__(state_size, n_heads, dropout)
        self.input_size = input_size or state_size

        assert position in ["direction", "none"]
        self.position = position

        self.exclude_diagonal = exclude_diagonal

        self.data_to_kv = torch.nn.Linear(state_size, 2 * n_heads * self.projection_size)
        self.data_to_q = torch.nn.Linear(self.input_size, n_heads * self.projection_size)
        if position == "direction":
            self.data_to_qp = torch.nn.Linear(self.input_size, n_heads * 2)

        self.data_to_range_switch = torch.nn.Linear(state_size, n_heads) if range_select else None

        # self.s_bias = torch.nn.Parameter(torch.full([1], -2.0))
        self.s_bias = torch.nn.Parameter(torch.full([1], 0.0))
        self.scale = torch.nn.Parameter(torch.full([1], 1.0 / math.sqrt(self.projection_size)))
        self.scale_pos = torch.nn.Parameter(torch.full([1], 1.0))
        self.normalize_score = normalize_score

        self.input_size = state_size if input_size is None else input_size


        print(f"DirectionSensitiveGeometricAttention: normalize score: {normalize_score}")

        super(DirectionSensitiveGeometricAttention, self).__init__(output_size)
        self.reset_parameters()

    def get_attention_scores(self, mask: Optional[torch.Tensor],
                   q_content: torch.Tensor, k_content: torch.Tensor,
                   q_pos: Optional[torch.Tensor],
                   pos_offset: int, range_switch: Optional[torch.Tensor]) -> torch.Tensor:

        # content-content addressing
        logits = torch.bmm(q_content, self.dropout(k_content).transpose(1, 2)) * self.scale

        if self.position == "direction":
            prefer_back, prefer_front = (q_pos * self.scale_pos).unsqueeze(-2).expand(-1,-1,logits.shape[-1],-1).unbind(-1)
            fpos = prefer_front.triu(1 + pos_offset) + prefer_back.tril(-1 + pos_offset)

            logits = logits + fpos + self.s_bias

        logits = self.apply_logit_masks(logits.view(logits.shape[0] // self.n_heads, self.n_heads, *logits.shape[1:]), mask).flatten(0,1)

        if self.exclude_diagonal:
            logits.masked_fill_(torch.eye(logits.shape[-1], device=logits.device, dtype=torch.bool)[pos_offset : pos_offset + logits.shape[-2]], float("-inf"))

        return geometric_attention_activation(logits, pos_offset, range_switch, normalize=self.normalize_score)

    def _attention(self, mask: Optional[torch.Tensor],
                   q_content: torch.Tensor, k_content: torch.Tensor,
                   q_pos: Optional[torch.Tensor],
                   v: torch.Tensor, pos_offset: int, range_switch: Optional[torch.Tensor]) -> [torch.Tensor, torch.Tensor]:

        scores = self.get_attention_scores(mask, q_content, k_content, q_pos, pos_offset, range_switch)

        # Scores shape: [n_batch * n_heads, n_out, n_in]
        return self._attention_read(mask, scores, v)

    def forward(self, curr_state: torch.Tensor, attend_to: torch.Tensor, mask: Optional[AttentionMask],
                pos_offset: Optional[int] = None, need_weights: bool = False):
        # curr_state: [batch_size, out_len, c]
        # attend_to: [batch_size, in_len, c]
        batch_size, in_len = attend_to.shape[0:2]
        out_len = curr_state.shape[1]

        if pos_offset is None:
            assert curr_state.shape[1] == attend_to.shape[1], "If attend_to has different shape than curr_state, pos_offset should be provided"
            pos_offset = 0

        k_content, v = self.transform_data(attend_to, self.data_to_kv, 2)
        q_content, = self.transform_data(curr_state, self.data_to_q, 1)
        if self.position == "direction":
            q_pos, = self.transform_data(curr_state, self.data_to_qp, 1)
        else:
            q_pos = None

        rs = self.transform_data(curr_state, self.data_to_range_switch, 1)[0].sigmoid()\
             if self.data_to_range_switch else None

        data, scores = self.merged_attention(batch_size, out_len, mask, q_content, k_content, q_pos, v,
                                             pos_offset, rs, need_weights=need_weights)

        if need_weights:
            return data, scores
        else:
            return data

    def xavier_manual_(self, tensor: torch.Tensor, fan_in: int, fan_out: int, gain: float = 1) -> torch.Tensor:
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

        return torch.nn.init._no_grad_uniform_(tensor, -a, a)

    def reset_parameters(self):
        self.xavier_manual_(self.data_to_q.weight, self.input_size, self.projection_size)
        self.xavier_manual_(self.data_to_kv.weight[:self.n_heads * self.projection_size], self.state_size, self.projection_size)
        self.xavier_manual_(self.data_to_kv.weight[self.n_heads * self.projection_size:], self.state_size, self.projection_size * self.n_heads)
        self.xavier_manual_(self.multi_head_merge.weight, self.projection_size * self.n_heads, self.output_size)