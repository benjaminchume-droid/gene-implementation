from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LongContextTrainingConfig:
    """
    Configuration contract for long-context training.

    The model supports a logical 112K context while attention
    remains bounded by an 8K local window.
    """

    context_length: int = 112000
    attention_window: int = 8192

    chunk_size: int = 8192

    overlap: int = 1024

    loss_mask_prefix: bool = False

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

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be positive."
            )

        if self.chunk_size > self.attention_window:
            raise ValueError(
                "chunk_size cannot exceed attention_window."
            )

        if self.overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        if self.overlap >= self.chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

    @property
    def stride(self) -> int:
        return self.chunk_size - self.overlap


def default_long_context_config() -> LongContextTrainingConfig:

    config = LongContextTrainingConfig()

    config.validate()

    return config
