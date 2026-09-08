from __future__ import annotations

from typing import Any

from .permissions import PermissionPolicy
from .registry import CapabilityRegistry
from .types import CapabilityResult


class CapabilityRouter:
    def __init__(
        self,
        registry: CapabilityRegistry,
        permissions: PermissionPolicy,
    ) -> None:
        self.registry = registry
        self.permissions = permissions

    def execute(
        self,
        capability_name: str,
        **arguments: Any,
    ) -> CapabilityResult:
        if not self.registry.has(capability_name):
            return CapabilityResult(
                success=False,
                capability=capability_name,
                error="Capability is not registered.",
            )

        capability = self.registry.get(capability_name)
        decision = self.permissions.check(
            capability.name,
            capability.risk,
        )

        if decision != "allowed":
            return CapabilityResult(
                success=False,
                capability=capability_name,
                error=f"Permission status: {decision}",
            )

        if capability.handler is None:
            return CapabilityResult(
                success=False,
                capability=capability_name,
                error="Capability has no execution handler.",
            )

        try:
            result = capability.handler(**arguments)
            return CapabilityResult(
                success=True,
                capability=capability_name,
                data=result,
            )
        except Exception as exc:
            return CapabilityResult(
                success=False,
                capability=capability_name,
                error=str(exc),
            )
