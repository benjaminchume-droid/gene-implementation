from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_layers: int = 6
    num_heads: int = 8
    max_position_embeddings: int = 2048
    attention_window: int = 2048

    dropout: float = 0.0
    rope_theta: float = 10000.0

    tie_word_embeddings: bool = True

    model_name: str = "gene-transformer"

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        if self.intermediate_size <= 0:
            raise ValueError(
                "intermediate_size must be positive."
            )

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads."
            )

        if self.max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive."
            )

        if self.attention_window <= 0:
            raise ValueError(
                "attention_window must be positive."
            )

        if self.attention_window > self.max_position_embeddings:
            raise ValueError(
                "attention_window cannot exceed "
                "max_position_embeddings."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def to_dict(self) -> dict:
        return asdict(self)


