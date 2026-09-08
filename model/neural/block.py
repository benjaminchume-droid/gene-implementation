from __future__ import annotations

import torch
from torch import nn

from .attention import (
    CausalSelfAttention,
)

from .mlp import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        max_position_embeddings: int,
        attention_window: int = 8192,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ) -> None:

        super().__init__()

        self.norm1 = nn.RMSNorm(
            hidden_size
        )

        self.attention = (
            CausalSelfAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                max_position_embeddings=
                    max_position_embeddings,
                attention_window=
                    attention_window,
                dropout=dropout,
                rope_theta=rope_theta,
            )
        )

        self.norm2 = nn.RMSNorm(
            hidden_size
        )

        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=
                intermediate_size,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        use_cache: bool = False,
        position_offset: int = 0,
    ):

        attention_output, present = (
            self.attention(
                self.norm1(
                    hidden_states
                ),
                attention_mask=
                    attention_mask,
                past_key_value=
                    past_key_value,
                use_cache=
                    use_cache,
                position_offset=
                    position_offset,
            )
        )

        hidden_states = (
            hidden_states
            + attention_output
        )

        hidden_states = (
            hidden_states
            + self.mlp(
                self.norm2(
                    hidden_states
                )
            )
        )

        return (
            hidden_states,
            present,
        )
