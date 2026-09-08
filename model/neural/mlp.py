from __future__ import annotations

import torch
from torch import nn


class SwiGLU(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        self.gate = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.up = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        gated = torch.nn.functional.silu(
            self.gate(hidden_states)
        )

        hidden = gated * self.up(
            hidden_states
        )

        return self.dropout(
            self.down(hidden)
        )
