from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceLimits:
    context_length: int
    attention_window: int
    max_new_tokens: int

    def validate(self) -> None:

        if self.context_length <= 0:
            raise ValueError(
                "context_length must be positive."
            )

        if self.attention_window <= 0:
            raise ValueError(
                "attention_window must be positive."
            )

        if self.attention_window > self.context_length:
            raise ValueError(
                "attention_window cannot exceed context_length."
            )

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )

        if self.max_new_tokens > self.context_length:
            raise ValueError(
                "max_new_tokens cannot exceed context_length."
            )


def limits_from_config(
    config,
    max_new_tokens: int = 256,
) -> InferenceLimits:

    limits = InferenceLimits(
        context_length=
            config.max_position_embeddings,
        attention_window=
            config.attention_window,
        max_new_tokens=
            max_new_tokens,
    )

    limits.validate()

    return limits
