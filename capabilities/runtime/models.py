from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeCapability:

    id: str
    name: str
    description: str
    kind: str

    procedure: list[dict[str, Any]] = field(
        default_factory=list
    )

    prerequisites: list[str] = field(
        default_factory=list
    )

    confidence: float = 0.0

    enabled: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )
