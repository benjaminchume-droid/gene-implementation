from __future__ import annotations

from dataclasses import dataclass, field

from .types import CapabilityRisk


@dataclass
class PermissionPolicy:
    """
    Central permission policy for Gene.

    Read-only and discovery capabilities are allowed.
    Mutating, system-level, browser-interaction, and external-tool
    capabilities require confirmation.
    """

    allowed: set[str] = field(default_factory=lambda: {
        "filesystem.read",
        "filesystem.list",
        "filesystem.search",
        "apps.find",
        "browser.navigate",
        "mcp.list",
    })

    confirmation_required: set[str] = field(default_factory=lambda: {
        "filesystem.write",
        "process.execute",
        "browser.execute",
        "mcp.execute",
    })

    denied: set[str] = field(default_factory=set)

    def check(
        self,
        capability: str,
        risk: CapabilityRisk | None = None,
    ) -> str:

        if capability in self.denied:
            return "denied"

        if capability in self.confirmation_required:
            return "confirmation_required"

        if capability in self.allowed:
            return "allowed"

        if risk in {
            CapabilityRisk.DESTRUCTIVE,
            CapabilityRisk.SYSTEM,
        }:
            return "confirmation_required"

        return "denied"
