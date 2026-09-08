from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MultimodalObservation:

    visual: dict[str, Any] | None = None

    speech: str | None = None

    audio_metadata: dict[str, Any] = field(
        default_factory=dict
    )

    source: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )
