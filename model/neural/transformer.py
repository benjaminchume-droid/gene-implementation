from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .block import TransformerBlock
from .config import TransformerConfig


class GeneTransformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:

        super().__init__()

        config.validate()

        self.config = config

        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=
                    config.hidden_size,
                intermediate_size=
                    config.intermediate_size,
                num_heads=
                    config.num_heads,
                max_position_embeddings=
                    config.max_position_embeddings,
                attention_window=
                    config.attention_window,
                dropout=
                    config.dropout,
                rope_theta=
                    config.rope_theta,
            )
            for _ in range(
                config.num_layers
            )
        ])

        self.final_norm = nn.RMSNorm(
            config.hidden_size
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = (
                self.token_embeddings.weight
            )

        self.apply(
            self._initialize_weights
        )

    def _initialize_weights(
        self,
        module,
    ) -> None:

        if isinstance(
            module,
            nn.Linear,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def _forward_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        past_key_values,
        use_cache: bool,
        position_offset: int,
    ):

        hidden_states = self.token_embeddings(
            input_ids
        )

        hidden_states = self.dropout(
            hidden_states
        )

        present_key_values = []

        for index, layer in enumerate(
            self.layers
        ):

            past = None

            if past_key_values is not None:
                past = past_key_values[
                    index
                ]

            hidden_states, present = (
                layer(
                    hidden_states,
                    attention_mask=
                        attention_mask,
                    past_key_value=
                        past,
                    use_cache=
                        use_cache,
                    position_offset=
                        position_offset,
                )
            )

            if use_cache:
                present_key_values.append(
                    present
                )

        hidden_states = self.final_norm(
            hidden_states
        )

        logits = self.lm_head(
            hidden_states
        )

        loss = None

        if labels is not None:

            shift_logits = logits[
                :, :-1, :
            ].contiguous()

            shift_labels = labels[
                :, 1:
            ].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(
                    -1,
                    shift_logits.size(-1),
                ),
                shift_labels.view(
                    -1
                ),
                ignore_index=-100,
            )

        if use_cache:

            return {
                "logits": logits,
                "loss": loss,
                "past_key_values":
                    tuple(
                        present_key_values
                    ),
            }

        return {
            "logits": logits,
            "loss": loss,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> dict:

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]."
            )

        sequence_length = (
            input_ids.shape[1]
        )

        if sequence_length == 0:
            raise ValueError(
                "input_ids cannot be empty."
            )

        if (
            position_offset
            + sequence_length
            > self.config.max_position_embeddings
        ):
            raise ValueError(
                "Sequence exceeds model context."
            )

        # ----------------------------------------------------
        # Long prefill is processed in local-attention chunks.
        # ----------------------------------------------------

        if (
            use_cache
            and sequence_length
            > self.config.attention_window
        ):

            if labels is not None:
                raise ValueError(
                    "Chunked cached forward does not "
                    "support labels."
                )

            if attention_mask is not None:
                raise ValueError(
                    "Chunked cached forward does not "
                    "support attention_mask."
                )

            all_logits = []

            current_cache = (
                past_key_values
            )

            offset = position_offset

            for start in range(
                0,
                sequence_length,
                self.config.attention_window,
            ):

                end = min(
                    start
                    + self.config.attention_window,
                    sequence_length,
                )

                chunk = input_ids[
                    :,
                    start:end,
                ]

                result = self._forward_chunk(
                    chunk,
                    None,
                    None,
                    current_cache,
                    True,
                    offset,
                )

                all_logits.append(
                    result["logits"]
                )

                current_cache = (
                    result["past_key_values"]
                )

                offset += (
                    end - start
                )

            return {
                "logits": torch.cat(
                    all_logits,
                    dim=1,
                ),
                "loss": None,
                "past_key_values":
                    current_cache,
            }

        return self._forward_chunk(
            input_ids,
            attention_mask,
            labels,
            past_key_values,
            use_cache,
            position_offset,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:

        self.eval()

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]."
            )

        if input_ids.shape[1] == 0:
            raise ValueError(
                "input_ids cannot be empty."
            )

        if (
            input_ids.shape[1]
            > self.config.max_position_embeddings
        ):
            raise ValueError(
                "Input exceeds model context."
            )

        generated = input_ids

        # ----------------------------------------------------
        # PREFILL
        # ----------------------------------------------------

        outputs = self(
            generated,
            use_cache=True,
            position_offset=0,
        )

        logits = outputs[
            "logits"
        ]

        past_key_values = outputs[
            "past_key_values"
        ]

        current_position = (
            generated.shape[1]
        )

        # ----------------------------------------------------
        # CACHED DECODE
        # ----------------------------------------------------

        for _ in range(
            max_new_tokens
        ):

            next_token_logits = (
                logits[:, -1, :]
            )

            if temperature <= 0:

                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1,
                    keepdim=True,
                )

            else:

                next_token_logits = (
                    next_token_logits
                    / temperature
                )

                if top_k is not None:

                    k = min(
                        top_k,
                        next_token_logits.shape[-1],
                    )

                    values, _ = torch.topk(
                        next_token_logits,
                        k,
                    )

                    minimum = values[
                        :, -1
                    ].unsqueeze(-1)

                    next_token_logits = (
                        next_token_logits.masked_fill(
                            next_token_logits
                            < minimum,
                            torch.finfo(
                                next_token_logits.dtype
                            ).min,
                        )
                    )

                probabilities = (
                    torch.softmax(
                        next_token_logits,
                        dim=-1,
                    )
                )

                next_token = (
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                    )
                )

            generated = torch.cat(
                [
                    generated,
                    next_token,
                ],
                dim=1,
            )

            current_position += 1

            if (
                current_position
                >= self.config.max_position_embeddings
            ):
                break

            outputs = self(
                next_token,
                past_key_values=
                    past_key_values,
                use_cache=True,
                position_offset=
                    current_position - 1,
            )

            logits = outputs[
                "logits"
            ]

            past_key_values = outputs[
                "past_key_values"
            ]

        return generated

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
        )

    def trainable_parameter_count(
        self,
    ) -> int:

        return sum(
            parameter.numel()
            for parameter
            in self.parameters()
            if parameter.requires_grad
        )
