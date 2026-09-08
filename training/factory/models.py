from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrainingRecord:

    text: str

    source_type: str
    source_id: str

    quality: float = 1.0
    confidence: float = 1.0

    accepted: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return asdict(self)
