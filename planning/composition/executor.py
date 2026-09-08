from __future__ import annotations

from ..models import Plan


class PlanExecutor:

    def __init__(
        self,
        capability_runtime,
    ) -> None:
        self.runtime = capability_runtime

    def execute(
        self,
        plan: Plan,
        context: dict | None = None,
    ) -> Plan:

        context = context or {}

        if plan.status not in (
            "ready",
            "partially_resolved",
        ):
            raise ValueError(
                f"Plan cannot execute from status "
                f"{plan.status!r}."
            )

        results = []
        plan.status = "running"

        for step in plan.steps:

            if not step.capability_id:
                plan.status = "blocked"
                plan.result = {
                    "error":
                        "Unresolved capability.",
                    "step":
                        step.id,
                }
                return plan

            try:

                result = self.runtime.execute(
                    step.capability_id,
                    context={
                        **context,
                        "plan":
                            plan.id,
                        "step":
                            step.id,
                        "inputs":
                            step.inputs,
                    },
                )

                results.append(
                    {
                        "step":
                            step.id,
                        "capability":
                            step.capability_id,
                        "result":
                            result,
                    }
                )

                if not result.get(
                    "success",
                    False,
                ):
                    plan.status = "failed"
                    plan.result = {
                        "results":
                            results,
                        "failed_step":
                            step.id,
                    }
                    return plan

            except Exception as exc:

                plan.status = "failed"
                plan.result = {
                    "results":
                        results,
                    "error":
                        str(exc),
                    "failed_step":
                        step.id,
                }
                return plan

        plan.status = "completed"
        plan.result = {
            "results":
                results
        }

        return plan
