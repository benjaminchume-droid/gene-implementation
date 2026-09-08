from __future__ import annotations

from ..models import Plan


class Replanner:

    def __init__(
        self,
        composer,
    ) -> None:
        self.composer = composer

    def replan(
        self,
        objective: str,
        failed_plan: Plan,
        failure: dict,
    ):

        remaining = [
            step.objective
            for step
            in failed_plan.steps
        ]

        return self.composer.compose(
            objective=objective,
            requested_steps=remaining,
        )
