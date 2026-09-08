from __future__ import annotations

from .permissions import PermissionPolicy
from .registry import CapabilityRegistry
from .router import CapabilityRouter
from .types import Capability, CapabilityRisk
from ..filesystem.service import FileSystemService
from ..apps.service import ApplicationService
from ..browser.service import BrowserService


def create_capability_system() -> CapabilityRouter:
    filesystem = FileSystemService()
    applications = ApplicationService()
    browser = BrowserService()

    registry = CapabilityRegistry()

    registry.register(
        Capability(
            name="filesystem.read",
            description="Read a text file.",
            risk=CapabilityRisk.READ,
            handler=filesystem.read_file,
        )
    )

    registry.register(
        Capability(
            name="filesystem.write",
            description="Write a text file.",
            risk=CapabilityRisk.WRITE,
            handler=filesystem.write_file,
        )
    )

    registry.register(
        Capability(
            name="filesystem.list",
            description="List a directory.",
            risk=CapabilityRisk.READ,
            handler=filesystem.list_directory,
        )
    )

    registry.register(
        Capability(
            name="filesystem.search",
            description="Search files recursively.",
            risk=CapabilityRisk.READ,
            handler=filesystem.search,
        )
    )

    registry.register(
        Capability(
            name="apps.find",
            description="Find an installed executable.",
            risk=CapabilityRisk.READ,
            handler=applications.find,
        )
    )

    registry.register(
        Capability(
            name="browser.navigate",
            description="Navigate a browser.",
            risk=CapabilityRisk.NETWORK,
            handler=browser.navigate,
        )
    )

    permissions = PermissionPolicy(
        allowed={
            "filesystem.read",
            "filesystem.list",
            "filesystem.search",
            "apps.find",
        },
        confirmation_required={
            "filesystem.write",
            "browser.navigate",
        },
    )

    return CapabilityRouter(registry, permissions)
