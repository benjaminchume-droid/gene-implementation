from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PermissionDecision(str, Enum):
    ALLOWED = "allowed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    DENIED = "denied"


class CapabilityRisk(str, Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    SYSTEM = "system"
    NETWORK = "network"


@dataclass
class PermissionPolicy:
    allow_filesystem_read: bool = True
    allow_filesystem_write: bool = True
    allow_filesystem_delete: bool = False

    allow_process_execution: bool = True
    allow_app_discovery: bool = True

    allow_browser: bool = True
    allow_mcp: bool = True

    # Operations that can change the external environment
    confirmation_required: set[str] = field(default_factory=lambda: {
        "filesystem.write",
        "process.execute",
        "browser.execute",
        "mcp.execute",
    })

    denied: set[str] = field(default_factory=set)

    allowed_roots: list[str] = field(
        default_factory=lambda: ["gene"]
    )

    risk_map: dict[str, CapabilityRisk] = field(
        default_factory=lambda: {

            "filesystem.read":
                CapabilityRisk.READ,

            "filesystem.list":
                CapabilityRisk.READ,

            "filesystem.search":
                CapabilityRisk.READ,

            "filesystem.write":
                CapabilityRisk.WRITE,

            "filesystem.delete":
                CapabilityRisk.DESTRUCTIVE,

            "process.execute":
                CapabilityRisk.SYSTEM,

            "apps.find":
                CapabilityRisk.READ,

            "browser.navigate":
                CapabilityRisk.NETWORK,

            "browser.execute":
                CapabilityRisk.NETWORK,

            "mcp.register":
                CapabilityRisk.NETWORK,

            "mcp.list":
                CapabilityRisk.READ,

            "mcp.execute":
                CapabilityRisk.NETWORK,
        }
    )

    def check(
        self,
        capability: str,
    ) -> PermissionDecision:

        if capability in self.denied:
            return PermissionDecision.DENIED

        # Explicit confirmation takes priority.
        if capability in self.confirmation_required:
            return PermissionDecision.CONFIRMATION_REQUIRED

        mapping = {

            "filesystem.read":
                self.allow_filesystem_read,

            "filesystem.list":
                self.allow_filesystem_read,

            "filesystem.search":
                self.allow_filesystem_read,

            "filesystem.write":
                self.allow_filesystem_write,

            "filesystem.delete":
                self.allow_filesystem_delete,

            "process.execute":
                self.allow_process_execution,

            "apps.find":
                self.allow_app_discovery,

            "browser.navigate":
                self.allow_browser,

            "browser.execute":
                self.allow_browser,

            "mcp.register":
                self.allow_mcp,

            "mcp.list":
                self.allow_mcp,

            "mcp.execute":
                self.allow_mcp,
        }

        if not mapping.get(capability, False):
            return PermissionDecision.DENIED

        risk = self.risk_map.get(
            capability,
            CapabilityRisk.SYSTEM,
        )

        if risk in {
            CapabilityRisk.DESTRUCTIVE,
            CapabilityRisk.SYSTEM,
        }:
            return PermissionDecision.CONFIRMATION_REQUIRED

        return PermissionDecision.ALLOWED

    def is_allowed(self, capability: str) -> bool:
        return self.check(capability) == PermissionDecision.ALLOWED

    def requires_confirmation(self, capability: str) -> bool:
        return (
            self.check(capability)
            == PermissionDecision.CONFIRMATION_REQUIRED
        )

    def is_denied(self, capability: str) -> bool:
        return self.check(capability) == PermissionDecision.DENIED
