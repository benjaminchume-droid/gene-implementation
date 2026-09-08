from __future__ import annotations

from .models import RuntimeCapability
from .registry import CapabilityRuntimeRegistry


class CapabilityExecutor:

    def __init__(
        self,
        registry: CapabilityRuntimeRegistry,
        discovery_engine=None,
    ) -> None:

        self.registry = registry
        self.discovery = discovery_engine

    def execute(
        self,
        capability_id: str,
        *,
        context: dict | None = None,
    ) -> dict:

        capability = self.registry.get(
            capability_id
        )

        if not capability.enabled:
            raise RuntimeError(
                "Capability is disabled."
            )

        context = context or {}

        # A capability with no executable procedure is a
        # valid compositional/no-op capability. This is
        # especially useful for planning and higher-level
        # capabilities whose behavior is supplied elsewhere.
        if not capability.procedure:
            return {
                "success": True,
                "capability":
                    capability.id,
                "results": [],
                "mode":
                    "composite_or_noop",
            }

        if self.discovery is None:
            return {
                "success": False,
                "capability":
                    capability.id,
                "error":
                    "No discovery runtime is connected.",
            }

        results = []

        for step in capability.procedure:

            resource_id = (
                step.get(
                    "resource_id"
                )
                or context.get(
                    "resource_id"
                )
            )

            action = step.get(
                "action",
                {},
            )

            if not resource_id:
                raise ValueError(
                    "Capability procedure step "
                    "does not specify a resource."
                )

            resource = None

            for candidate in (
                self.discovery.registry.discover_all()
            ):

                if candidate.id == resource_id:
                    resource = candidate
                    break

            if resource is None:
                raise KeyError(
                    f"Runtime resource not found: "
                    f"{resource_id}"
                )

            result = self.discovery.execute(
                resource,
                action,
            )

            results.append(
                {
                    "step":
                        step,
                    "result":
                        result,
                }
            )

        return {
            "success": True,
            "capability":
                capability.id,
            "results":
                results,
        }
