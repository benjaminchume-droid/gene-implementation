from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class AgentTask:
    instruction: str
    id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    status: str = "created"
    route: str | None = None
    worker: str | None = None
    context: str = ""
    observations: list[dict] = field(
        default_factory=list
    )
    result: object = None
    error: str | None = None
    created_at: str = field(
        default_factory=lambda:
        datetime.now(timezone.utc).isoformat()
    )

    def update(self, status: str) -> None:
        self.status = status
