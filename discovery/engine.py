from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .capabilities import CapabilityStore
from .models import Resource
from .registry import DiscoveryRegistry


@dataclass
class DiscoveryMission:

    objective: str

    constraints: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class DiscoveryEngine:

    def __init__(
        self,
        registry: DiscoveryRegistry | None = None,
        capabilities: CapabilityStore | None = None,
    ) -> None:

        self.registry = (
            registry
            or DiscoveryRegistry()
        )

        self.capabilities = (
            capabilities
            or CapabilityStore()
        )

    def discover(
        self,
        mission: DiscoveryMission,
    ) -> dict:

        resources = (
            self.registry.discover_all()
        )

        observations = []

        for resource in resources:

            for environment in (
                self.registry.environments_list()
            ):

                try:

                    observation = (
                        environment.observe(
                            resource
                        )
                    )

                    observations.append(
                        observation
                    )

                    break

                except Exception:
                    continue

        return {
            "objective":
                mission.objective,

            "resources":
                [
                    {
                        "id":
                            resource.id,
                        "kind":
                            resource.kind,
                        "name":
                            resource.name,
                        "description":
                            resource.description,
                        "actions":
                            resource.actions,
                    }
                    for resource
                    in resources
                ],

            "observations":
                [
                    {
                        "resource_id":
                            item.resource_id,
                        "source":
                            item.source,
                        "confidence":
                            item.confidence,
                        "data":
                            item.data,
                    }
                    for item
                    in observations
                ],
        }

    def execute(
        self,
        resource: Resource,
        action: dict[str, Any],
    ):

        errors = []

        for environment in (
            self.registry.environments_list()
        ):

            try:
                return environment.execute(
                    resource,
                    action,
                )

            except Exception as exc:
                errors.append(
                    str(exc)
                )

        raise RuntimeError(
            "No discovery environment could "
            "execute this action."
        )

    def acquire_capability(
        self,
        mission: DiscoveryMission,
        resource: Resource,
        capability: str,
        procedure,
        *,
        evidence_ids=None,
        confidence: float = 0.0,
        verified: bool = False,
        reusable: bool = False,
        metadata=None,
    ):

        return self.capabilities.add(
            objective=mission.objective,
            resource_id=resource.id,
            resource_kind=resource.kind,
            capability=capability,
            procedure=procedure,
            evidence_ids=evidence_ids,
            confidence=confidence,
            verified=verified,
            reusable=reusable,
            metadata=metadata,
        )

    def status(self) -> dict:

        return {
            "registry":
                self.registry.status(),
            "capabilities":
                self.capabilities.status(),
        }
