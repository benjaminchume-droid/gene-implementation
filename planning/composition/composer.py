from __future__ import annotations

from dataclasses import dataclass

from gene.capabilities.runtime import (
    CapabilityRuntime,
    RuntimeCapability,
)

from ..models import (
    Plan,
    PlanStep,
)


@dataclass
class CompositionResult:

    success: bool
    plan: Plan
    capabilities_used: list[str]
    unresolved_steps: list[str]
    confidence: float


class CapabilityComposer:

    def __init__(
        self,
        runtime: CapabilityRuntime,
    ) -> None:
        self.runtime = runtime

    def find(
        self,
        description: str,
    ) -> list[RuntimeCapability]:

        return self.runtime.registry.find(
            description
        )

    def compose(
        self,
        objective: str,
        requested_steps: list[str],
    ) -> CompositionResult:

        plan = Plan.create(
            objective
        )

        capabilities_used = []
        unresolved = []

        previous = None

        for description in requested_steps:

            matches = self.find(
                description
            )

            capability = (
                matches[0]
                if matches
                else None
            )

            step = PlanStep(
                id=self._step_id(),
                capability_id=(
                    capability.id
                    if capability
                    else None
                ),
                objective=description,
                dependencies=(
                    [previous]
                    if previous
                    else []
                ),
            )

            plan.steps.append(step)

            if capability:
                capabilities_used.append(
                    capability.id
                )
                previous = step.id

            else:
                unresolved.append(
                    description
                )

        plan.confidence = (
            len(capabilities_used)
            / len(requested_steps)
            if requested_steps
            else 0.0
        )

        plan.status = (
            "ready"
            if not unresolved
            else "partially_resolved"
        )

        return CompositionResult(
            success=not unresolved,
            plan=plan,
            capabilities_used=
                capabilities_used,
            unresolved_steps=
                unresolved,
            confidence=
                plan.confidence,
        )

    @staticmethod
    def _step_id() -> str:
        import uuid
        return f"step-{uuid.uuid4()}"
