from __future__ import annotations

from gene.discovery.capabilities import (
    CapabilityStore,
)

from .models import Experiment


class ExperimentPromoter:

    def __init__(
        self,
        capabilities: CapabilityStore | None = None,
    ) -> None:

        self.capabilities = (
            capabilities
            or CapabilityStore()
        )

    def promote(
        self,
        experiment: Experiment,
        capability: str,
        *,
        reusable: bool = True,
        metadata: dict | None = None,
    ):

        if experiment.status != "verified":
            raise ValueError(
                "Only verified experiments can "
                "be promoted."
            )

        procedure = [
            {
                "action":
                    item.action,
                "expected":
                    item.expected,
            }
            for item in experiment.actions
        ]

        evidence = [
            observation.result
            for observation
            in experiment.observations
        ]

        return self.capabilities.add(
            objective=
                experiment.objective,
            resource_id=
                experiment.resource_id,
            resource_kind=
                "discovered",
            capability=
                capability,
            procedure=
                procedure,
            confidence=
                1.0,
            verified=True,
            reusable=reusable,
            metadata={
                "experiment_id":
                    experiment.id,
                "observations":
                    evidence,
                **(
                    metadata or {}
                ),
            },
        )
