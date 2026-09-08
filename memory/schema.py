from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Memory:
    content: str
    source: str = "conversation"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
