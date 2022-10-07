import torch
from typing import Optional
import torch.nn.functional as F


class LowDimensionalEmbedding(torch.nn.Module):
    def __init__(self, n_tokens: int, n_channels: int, embedding_size: Optional[int] = None, backproject: bool = False):
        super().__init__()

        self.embedding = torch.nn.Embedding(n_tokens, embedding_size or n_channels)
        self.is_low_d = embedding_size is not None and n_channels != embedding_size
        if self.is_low_d:
            self.out_map = torch.nn.Linear(embedding_size, n_channels)
            if backproject:
                self.bp_map = torch.nn.Linear(n_channels, embedding_size)
        else:
            self.out_map = lambda x: x
            self.bp_map = lambda x: x

        if backproject:
            self.bp_bias = torch.nn.Parameter(torch.zeros(n_tokens))

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_map(self.embedding(x))

    def back_project(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bp_map(x)
        return F.linear(x, self.embedding.weight, self.bp_bias)
