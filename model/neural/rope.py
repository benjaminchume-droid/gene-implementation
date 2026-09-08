from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for rotary embeddings."
            )

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        inverse_frequency = 1.0 / (
            theta
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
    ):
        sequence_length = x.shape[-2]

        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            dtype=torch.float32,
            device=x.device,
        )

        inverse_frequency = (
            self.inverse_frequency
            .to(device=x.device)
        )

        frequencies = torch.outer(
            positions,
            inverse_frequency,
        )

        cos = frequencies.cos()[
            None,
            None,
            :,
            :
        ]

        sin = frequencies.sin()[
            None,
            None,
            :,
            :
        ]

        return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    rotated = torch.stack(
        (-x2, x1),
        dim=-1,
    )

    rotated = rotated.flatten(
        start_dim=-2
    )

    cos_full = torch.repeat_interleave(
        cos,
        2,
        dim=-1,
    )

    sin_full = torch.repeat_interleave(
        sin,
        2,
        dim=-1,
    )

    return (
        x * cos_full
        + rotated * sin_full
    )
