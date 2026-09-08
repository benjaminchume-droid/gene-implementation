from __future__ import annotations

import torch
from torch import nn


class VisionProjector(nn.Module):

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        output_dim: int = 864,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),

            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:

        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)

        if embedding.shape[-1] != self.input_dim:
            raise ValueError(
                "Expected final dimension "
                f"{self.input_dim}, got "
                f"{embedding.shape[-1]}"
            )

        return self.network(
            embedding
        )
