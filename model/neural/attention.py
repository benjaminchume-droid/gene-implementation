from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .rope import (
    RotaryEmbedding,
    apply_rope,
)

try:
    from torch.nn.attention.bias import (
        causal_lower_right,
    )
except ImportError:
    causal_lower_right = None


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        attention_window: int = 8192,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ) -> None:

        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads."
            )

        if attention_window <= 0:
            raise ValueError(
                "attention_window must be positive."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_window = attention_window
        self.dropout_probability = dropout

        self.qkv = nn.Linear(
            hidden_size,
            hidden_size * 3,
            bias=False,
        )

        self.output = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            theta=rope_theta,
        )

    def _fallback_causal_mask(
        self,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:

        past_length = key_length - query_length

        query_positions = torch.arange(
            query_length,
            device=device,
        ).unsqueeze(1)

        key_positions = torch.arange(
            key_length,
            device=device,
        ).unsqueeze(0)

        return (
            key_positions
            <= query_positions + past_length
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        use_cache: bool = False,
        position_offset: int = 0,
    ):

        batch, sequence, _ = hidden_states.shape

        if sequence > self.attention_window:
            raise ValueError(
                "Attention sequence exceeds the local "
                "attention window. Use chunked prefill."
            )

        qkv = self.qkv(
            hidden_states
        )

        qkv = qkv.view(
            batch,
            sequence,
            3,
            self.num_heads,
            self.head_dim,
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        query, key, value = qkv.unbind(
            dim=0
        )

        cos, sin = self.rope(
            query,
            position_offset=position_offset,
        )

        query = apply_rope(
            query,
            cos,
            sin,
        )

        key = apply_rope(
            key,
            cos,
            sin,
        )

        if past_key_value is not None:

            past_key, past_value = (
                past_key_value
            )

            keep_past = max(
                0,
                self.attention_window - sequence,
            )

            if keep_past > 0:
                past_key = past_key[
                    :,
                    :,
                    -keep_past:,
                    :
                ]

                past_value = past_value[
                    :,
                    :,
                    -keep_past:,
                    :
                ]
            else:
                past_key = past_key[
                    :,
                    :,
                    0:0,
                    :
                ]

                past_value = past_value[
                    :,
                    :,
                    0:0,
                    :
                ]

            key = torch.cat(
                [past_key, key],
                dim=2,
            )

            value = torch.cat(
                [past_value, value],
                dim=2,
            )

        key_length = key.shape[2]

        dropout_probability = (
            self.dropout_probability
            if self.training
            else 0.0
        )

        # ----------------------------------------------------
        # Fast causal path
        # ----------------------------------------------------

        if past_key_value is None:

            if attention_mask is None:

                output = (
                    F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        dropout_p=dropout_probability,
                        is_causal=True,
                    )
                )

            else:

                padding_mask = (
                    attention_mask[:, None, None, :]
                    .to(torch.bool)
                )

                causal_mask = (
                    torch.ones(
                        sequence,
                        key_length,
                        dtype=torch.bool,
                        device=hidden_states.device,
                    )
                    .tril()
                )

                combined_mask = (
                    causal_mask[None, None, :, :]
                    & padding_mask
                )

                output = (
                    F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=combined_mask,
                        dropout_p=dropout_probability,
                        is_causal=False,
                    )
                )

        # ----------------------------------------------------
        # Cached causal path
        # ----------------------------------------------------

        else:

            if attention_mask is None:

                if causal_lower_right is not None:

                    causal_bias = (
                        causal_lower_right(
                            sequence,
                            key_length,
                        )
                    )

                    output = (
                        F.scaled_dot_product_attention(
                            query,
                            key,
                            value,
                            attn_mask=causal_bias,
                            dropout_p=dropout_probability,
                            is_causal=False,
                        )
                    )

                else:

                    causal_mask = (
                        self._fallback_causal_mask(
                            sequence,
                            key_length,
                            hidden_states.device,
                        )
                    )

                    output = (
                        F.scaled_dot_product_attention(
                            query,
                            key,
                            value,
                            attn_mask=causal_mask[
                                None,
                                None,
                                :,
                                :
                            ],
                            dropout_p=dropout_probability,
                            is_causal=False,
                        )
                    )

            else:

                causal_mask = (
                    self._fallback_causal_mask(
                        sequence,
                        key_length,
                        hidden_states.device,
                    )
                )

                output_mask = causal_mask[
                    None,
                    None,
                    :,
                    :
                ]

                output = (
                    F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=output_mask,
                        dropout_p=dropout_probability,
                        is_causal=False,
                    )
                )

        output = output.transpose(
            1,
            2,
        ).contiguous()

        output = output.view(
            batch,
            sequence,
            self.hidden_size,
        )

        output = self.output(
            output
        )

        present_key_value = None

        if use_cache:

            cache_key = key[
                :,
                :,
                -self.attention_window:,
                :
            ].detach()

            cache_value = value[
                :,
                :,
                -self.attention_window:,
                :
            ].detach()

            present_key_value = (
                cache_key,
                cache_value,
            )

        return (
            output,
            present_key_value,
        )
