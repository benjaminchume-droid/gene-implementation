from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass
class Experience:

    id: str
    objective: str
    outcome: str
    success: bool

    observations: list[dict[str, Any]] = field(
        default_factory=list
    )

    actions: list[dict[str, Any]] = field(
        default_factory=list
    )

    errors: list[dict[str, Any]] = field(
        default_factory=list
    )

    plan_id: str | None = None

    capability_ids: list[str] = field(
        default_factory=list
    )

    evidence_ids: list[str] = field(
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

    @classmethod
    def create(
        cls,
        objective: str,
        outcome: str,
        *,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> "Experience":

        return cls(
            id=(
                "experience-"
                + uuid.uuid4().hex
            ),
            objective=objective,
            outcome=outcome,
            success=success,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict:
        return asdict(self)
