from __future__ import annotations

import time

from .models import (
    Experiment,
    ExperimentAction,
    ExperimentObservation,
)
from .verification.engine import (
    VerificationEngine,
)


class ExperimentRunner:

    def __init__(
        self,
        discovery_engine,
        verification: VerificationEngine | None = None,
    ) -> None:

        self.discovery = discovery_engine
        self.verification = (
            verification
            or VerificationEngine()
        )

    def run(
        self,
        experiment: Experiment,
        *,
        max_attempts: int = 1,
    ) -> Experiment:

        if max_attempts <= 0:
            raise ValueError(
                "max_attempts must be positive."
            )

        experiment.status = "running"

        for _ in range(max_attempts):

            experiment.attempts += 1
            all_passed = True

            for index, planned in enumerate(
                experiment.actions
            ):

                try:

                    resource = self._resource(
                        experiment.resource_id
                    )

                    started = time.time()

                    result = self.discovery.execute(
                        resource,
                        planned.action,
                    )

                    elapsed = (
                        time.time() - started
                    )

                    verification = (
                        self.verification.verify(
                            result,
                            planned.expected,
                        )
                    )

                    observation = ExperimentObservation(
                        action_index=index,
                        result={
                            "execution": result,
                            "verification": {
                                "passed":
                                    verification.passed,
                                "score":
                                    verification.score,
                                "checks":
                                    verification.checks,
                                "reasons":
                                    verification.reasons,
                            },
                            "elapsed_seconds":
                                elapsed,
                        },
                        success=verification.passed,
                    )

                    experiment.observations.append(
                        observation
                    )

                    if not verification.passed:
                        all_passed = False
                        break

                except Exception as exc:

                    experiment.observations.append(
                        ExperimentObservation(
                            action_index=index,
                            result={
                                "error":
                                    str(exc),
                            },
                            success=False,
                        )
                    )

                    all_passed = False
                    break

            if all_passed:
                experiment.status = "verified"
                return experiment

        experiment.status = "failed"
        return experiment

    def _resource(
        self,
        resource_id: str,
    ):

        for resource in (
            self.discovery.registry.discover_all()
        ):
            if resource.id == resource_id:
                return resource

        raise KeyError(
            f"Unknown resource: {resource_id}"
        )
