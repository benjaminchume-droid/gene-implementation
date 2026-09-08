from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid


class MissionState(str, Enum):

    CREATED = "created"
    DISCOVERING = "discovering"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    LEARNING = "learning"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Mission:

    id: str
    objective: str

    state: MissionState = (
        MissionState.CREATED
    )

    context: dict[str, Any] = field(
        default_factory=dict
    )

    discovered: dict[str, Any] = field(
        default_factory=dict
    )

    plan: Any = None
    result: Any = None

    attempts: int = 0

    errors: list[dict[str, Any]] = field(
        default_factory=list
    )

    learning: dict[str, Any] = field(
        default_factory=dict
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

    @classmethod
    def create(
        cls,
        objective: str,
        *,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Mission":

        return cls(
            id=(
                "mission-"
                + uuid.uuid4().hex
            ),
            objective=objective,
            context=context or {},
            metadata=metadata or {},
        )

    def transition(
        self,
        state: MissionState,
    ) -> None:

        self.state = state

        self.updated_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
