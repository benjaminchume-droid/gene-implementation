from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ScreenAnalysisRequest:
    image_path: str
    instruction: str = ""
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda:
        datetime.now(timezone.utc).isoformat()
    )


@dataclass
class VisionObservation:
    success: bool
    provider: str
    image_path: str

    description: str | None = None
    text: str | None = None

    regions: list[dict[str, Any]] = field(
        default_factory=list
    )

    structured: dict[str, Any] = field(
        default_factory=dict
    )

    error: str | None = None
