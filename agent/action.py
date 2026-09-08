from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class ActionRequest:
    action: str
    payload: dict = field(default_factory=dict)

    id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    requested_at: str = field(
        default_factory=lambda:
        datetime.now(timezone.utc).isoformat()
    )

    status: str = "requested"
    requires_confirmation: bool = False
    confirmation_reason: str | None = None

    def approve(self) -> None:
        self.status = "approved"

    def reject(self) -> None:
        self.status = "rejected"

    def complete(self) -> None:
        self.status = "completed"

    def fail(self, reason: str) -> None:
        self.status = "failed"
        self.confirmation_reason = reason