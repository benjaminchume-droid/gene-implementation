from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass
class PlanStep:

    id: str

    capability_id: str | None

    objective: str

    inputs: dict[str, Any] = field(
        default_factory=dict
    )

    expected_output: dict[str, Any] = field(
        default_factory=dict
    )

    dependencies: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Plan:

    id: str

    objective: str

    steps: list[PlanStep] = field(
        default_factory=list
    )

    status: str = "created"

    result: Any = None

    confidence: float = 0.0

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
        metadata: dict[str, Any] | None = None,
    ) -> "Plan":

        return cls(
            id=str(uuid.uuid4()),
            objective=objective,
            metadata=metadata or {},
        )
