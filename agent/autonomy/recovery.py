from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RecoveryDecision:

    retry: bool

    replanned_steps: list[str]

    reason: str

    metadata: dict[str, Any]


class RecoveryEngine:

    def __init__(
        self,
        planner: Callable[
            [str, dict[str, Any]],
            list[str],
        ] | None = None,
    ) -> None:

        self.planner = planner

    def recover(
        self,
        objective: str,
        failure: dict[str, Any],
    ) -> RecoveryDecision:

        if self.planner is None:
            return RecoveryDecision(
                retry=False,
                replanned_steps=[],
                reason=(
                    "No recovery planner "
                    "is available."
                ),
                metadata={},
            )

        try:

            steps = self.planner(
                objective,
                failure,
            )

            return RecoveryDecision(
                retry=bool(steps),
                replanned_steps=steps,
                reason=(
                    "Recovery planner "
                    "generated an alternative plan."
                ),
                metadata={},
            )

        except Exception as exc:

            return RecoveryDecision(
                retry=False,
                replanned_steps=[],
                reason=str(exc),
                metadata={
                    "error":
                        type(exc).__name__,
                },
            )
