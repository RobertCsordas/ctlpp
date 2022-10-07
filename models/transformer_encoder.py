import torch
import torch.nn
import torch.nn.functional as F
import framework
from layers.transformer import TransformerEncoderWithLayer, AttentionMask
from typing import Callable, Optional, Tuple, List
import math
from .encoder_decoder import add_eos
from layers import LowDimensionalEmbedding


class TransformerEncoderBase(torch.nn.Module):
    def __init__(self, n_input_tokens: int, state_size: int = 512, ff_multiplier: float = 4,
                 max_len=5000, transformer=TransformerEncoderWithLayer(),
                 pos_embeddig: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
                 embedding_init: str = "pytorch", scale_mode: str = "none",
                 sos: bool = True, eos: bool = True, autoregressive: bool = False,
                 embedding_size: Optional[None] = None, tied_embedding: bool = False, 
                 extra_lengt_prop: List[float] = [0.0, 0.0], **kwargs):
        super().__init__()

        assert scale_mode in ["none", "opennmt", "down"]
        assert embedding_init in ["pytorch", "xavier", "kaiming"]

        self.encoder_eos = n_input_tokens
        self.encoder_sos = n_input_tokens + 1
        self.encoder_pad = n_input_tokens + 2
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.embedding_init = embedding_init
        self.ff_multiplier = ff_multiplier
        self.n_input_tokens = n_input_tokens
        self.scale_mode = scale_mode
        self.autoregressive = autoregressive
        self.tied_embedding = tied_embedding
        self.extra_lengt_prop = extra_lengt_prop
        self.eos = eos
        self.sos = sos
        self.pos = pos_embeddig or framework.layers.PositionalEncoding(state_size, max_len=max_len, batch_first=True,
                                        scale=(1.0 / math.sqrt(state_size)) if scale_mode == "down" else 1.0)

        self.register_buffer('int_seq', torch.arange(max_len, dtype=torch.long))
        self.construct(transformer, **kwargs)
        self.reset_parameters()

    def pos_embed(self, t: torch.Tensor, offset: int, scale_offset: int) -> torch.Tensor:
        if self.scale_mode == "opennmt":
            t = t * math.sqrt(t.shape[-1])

        return self.pos(t, offset)

    def construct(self, transformer, **kwargs):
        self.embedding = LowDimensionalEmbedding(self.n_input_tokens + 3, self.state_size, self.embedding_size,
                                                 backproject=self.tied_embedding)
        self.trafo = transformer(d_model=self.state_size, dim_feedforward=int(self.ff_multiplier * self.state_size),
                                 **kwargs)

    def reset_parameters(self):
        if self.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        elif self.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.embedding.weight)

        # with torch.no_grad():
        #     self.embedding.weight.mul_(0.1)

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[: max_len] >= len.unsqueeze(-1)

    def trafo_forward(self, src: torch.Tensor, src_len: torch.Tensor, tgt_len: Optional[torch.Tensor],
                      mask: AttentionMask) -> torch.Tensor:
        return self.trafo(src, mask)

    def mask_input(self, src: torch.Tensor, in_len_mask: torch.Tensor) -> torch.Tensor:
        src[in_len_mask] = self.encoder_pad
        return src

    def forward(self, src: torch.Tensor, src_len: torch.Tensor, tgt_len: Optional[torch.Tensor] = None,
                max_out_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
       
        if self.eos is not None:
            src = add_eos(src, src_len, self.encoder_eos, batch_dim=0, pad_val=self.encoder_pad)
            src_len = src_len + 1

        if self.sos is not None:
            src = F.pad(src, (1, 0), value=self.encoder_sos)
            src_len = src_len + 1

        if tgt_len is not None and max_out_len is None:
            max_out_len = tgt_len.max().item()

        if max_out_len is not None:
            # If the output sequence is longer than the input, pad the input
            n_more_out = max_out_len - src.shape[1]
            if n_more_out > 0:
                src = F.pad(src, (0, n_more_out), value=self.encoder_pad)

        if tgt_len is None:
            ncols = src_len 
        else:
            ncols = torch.max(src_len, tgt_len)

        if self.extra_lengt_prop[1] > 0:
            l = torch.rand_like(ncols, dtype=torch.float) * (self.extra_lengt_prop[1] - self.extra_lengt_prop[0]) + \
                                self.extra_lengt_prop[0]
            n_extra = (ncols * l).ceil().int()
            src = F.pad(src, (0, n_extra.max().item()), value=self.encoder_pad)
            ncols += n_extra

        in_len_mask = self.generate_len_mask(src.shape[1], ncols)
        src = self.mask_input(src, in_len_mask)
        src = self.pos_embed(self.embedding(src.long()), 0, 0)

        causality_mask = Transformer.generate_square_subsequent_mask(src.shape[1], src.device) \
                         if self.autoregressive else None

        return self.trafo_forward(src, src_len, tgt_len, AttentionMask(in_len_mask, causality_mask)), ncols
