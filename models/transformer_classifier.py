import torch
import torch.nn
from layers import LowDimensionalEmbedding
from layers.transformer import TransformerEncoderWithLayer, AttentionMask
from typing import Callable, Optional
import math
from layers.transformer.multi_head_attention import MultiHeadAttention
from .transformer_encoder import TransformerEncoderBase


class TransformerClassifierModel(TransformerEncoderBase):
    def __init__(self, n_input_tokens: int, n_out_tokens: int, state_size: int = 512, ff_multiplier: float = 4,
                 max_len=5000, transformer=TransformerEncoderWithLayer(),
                 pos_embeddig: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
                 out_mode: str = "linear", embedding_init: str = "pytorch", scale_mode: str = "none",
                 result_column: str = "first", sos: bool = True, eos: bool = True,
                 autoregressive: bool = False, embedding_size: Optional[None] = None, **kwargs):

        assert scale_mode in ["none", "opennmt", "down"]

        assert (out_mode != "tied") or (n_input_tokens == n_out_tokens)
        assert out_mode in ["tied", "linear", "attention"]

        self.n_out_tokens = n_out_tokens
        self.out_mode = out_mode
        self.result_column = result_column
        assert self.result_column in ["first", "last"]

        super().__init__(n_input_tokens, state_size, ff_multiplier, max_len, transformer, pos_embeddig, embedding_init,
                         scale_mode, sos, eos, autoregressive, embedding_size, **kwargs)

    def construct(self, transformer, **kwargs):
        self.encoder_pad = 0
        self.embedding = LowDimensionalEmbedding(self.n_input_tokens + 2, self.state_size, self.embedding_size,
                                                 backproject=self.out_mode == "tied")

        if self.out_mode == "tied":
            self.output_map = self.embedding.back_project
        elif self.out_mode == "linear":
            self.output_map = torch.nn.Linear(self.state_size, self.n_out_tokens)
        elif self.out_mode == "attention":
            self.output_map = MultiHeadAttention(self.state_size, 1, out_size=self.n_out_tokens)
            self.out_query = torch.nn.Parameter(torch.randn([1, self.state_size]) / math.sqrt(self.state_size))

        self.trafo = transformer(d_model=self.state_size, dim_feedforward=int(self.ff_multiplier * self.state_size),
                                 **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        if self.output_map == "linear":
            torch.nn.init.xavier_uniform_(self.output_map.weight)

    def get_result(self, res: torch.Tensor, src_len: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if self.result_column == "first":
            return res[:, 0 + offset]
        elif self.result_column == "last":
            return res.gather(1, src_len.view([src_len.shape[0], 1, 1]).expand(-1, -1, res.shape[-1]) - \
                              (1 + offset)).squeeze(1)
        else:
            assert False

    def mask_input(self, src: torch.Tensor, in_len_mask: torch.Tensor) -> torch.Tensor:
        return src

    def trafo_forward(self, src: torch.Tensor, src_len: torch.Tensor, tgt_len: Optional[torch.Tensor],
                      mask: AttentionMask) -> torch.Tensor:
        res = self.trafo(src, mask)

        if self.out_mode == "attention":
            return self.output_map(self.out_query.expand(src.shape[0], -1, -1), res,
                                   mask=AttentionMask(mask.src_length_mask, None)).squeeze(1)
        else:
            return self.output_map(self.get_result(res, src_len))
