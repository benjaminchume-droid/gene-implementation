from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class KnowledgeRecord:
    id: int | None
    topic: str
    content: str
    source: str | None = None
    record_type: str = "fact"
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "content": self.content,
            "source": self.source,
            "record_type": self.record_type,
            "tags": self.tags,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
