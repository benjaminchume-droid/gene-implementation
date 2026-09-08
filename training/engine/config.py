from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    output_dir: str = "gene/data/training/checkpoints"
    run_dir: str = "gene/data/training/runs"

    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 1

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100

    max_steps: int | None = None
    validation_interval: int = 100
    checkpoint_interval: int = 500

    max_grad_norm: float = 1.0
    mixed_precision: bool = True

    seed: int = 42
    max_sequence_length: int = 8192
    log_interval: int = 10

    resume_from: str | None = None

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive."
            )

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be positive."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prepare_dirs(self) -> None:
        Path(self.output_dir).mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(self.run_dir).mkdir(
            parents=True,
            exist_ok=True,
        )
