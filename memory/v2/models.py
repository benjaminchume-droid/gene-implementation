from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    PREFERENCE = "preference"


@dataclass
class MemoryRecord:
    id: str
    memory_type: MemoryType
    content: str

    importance: float = 0.5
    confidence: float = 0.5

    source: str | None = None
    namespace: str = "default"

    tags: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    updated_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    access_count: int = 0
    last_accessed_at: str | None = None
