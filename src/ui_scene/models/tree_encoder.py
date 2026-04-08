from __future__ import annotations

import torch
from torch import nn


class TreeEncoder(nn.Module):
    """Minimal tree/text encoder placeholder."""

    def __init__(self, vocab_dim: int = 512, embedding_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(vocab_dim, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)

