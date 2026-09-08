from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NormalizedTrainingRecord:

    text: str

    source_type: str
    source_id: str

    quality: float = 1.0
    confidence: float = 1.0

    competency: dict[str, Any] = field(
        default_factory=dict
    )

    provenance: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return asdict(self)
