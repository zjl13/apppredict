from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """Minimal image encoder wrapper."""

    def __init__(self, input_dim: int = 512, embedding_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)

