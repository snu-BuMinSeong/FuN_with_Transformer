from __future__ import annotations

import torch
from torch import nn


class ObservationEncoder(nn.Module):
    """Encode MiniGrid image observations into configurable embeddings."""

    def __init__(self, embedding_dim: int = 64) -> None:
        """Initialize a small convolutional encoder for 7x7 observations."""
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, self.embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an embedding tensor with shape ``(B, embedding_dim)``."""
        if x.ndim != 4:
            raise ValueError(f"Expected input with 4 dimensions (B, 3, 7, 7), got shape {tuple(x.shape)}.")

        expected_shape = (3, 7, 7)
        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(f"Expected input shape (B, 3, 7, 7), got shape {tuple(x.shape)}.")

        return self.net(x)
