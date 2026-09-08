from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass
class ExperimentAction:

    action: dict[str, Any]

    expected: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ExperimentObservation:

    action_index: int
    result: Any
    success: bool

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Experiment:

    id: str
    objective: str
    resource_id: str

    actions: list[ExperimentAction] = field(
        default_factory=list
    )

    observations: list[
        ExperimentObservation
    ] = field(
        default_factory=list
    )

    status: str = "created"
    attempts: int = 0

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @classmethod
    def create(
        cls,
        objective: str,
        resource_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> "Experiment":

        return cls(
            id=str(uuid.uuid4()),
            objective=objective,
            resource_id=resource_id,
            metadata=metadata or {},
        )
