from __future__ import annotations

from gene.discovery.capabilities import (
    CapabilityStore,
)

from .models import RuntimeCapability
from .persistence import (
    CapabilityPersistence,
)
from .registry import (
    CapabilityRuntimeRegistry,
)


class CapabilityRuntime:

    def __init__(
        self,
        discovery_engine=None,
        path: str = (
            "gene/data/capabilities/"
            "runtime.json"
        ),
    ) -> None:

        self.registry = (
            CapabilityRuntimeRegistry()
        )

        self.persistence = (
            CapabilityPersistence(
                path
            )
        )

        self.discovery = (
            discovery_engine
        )

        self.executor = None

        self.persistence.load(
            self.registry
        )

        self._refresh_executor()

    def _refresh_executor(
        self,
    ) -> None:

        from .executor import (
            CapabilityExecutor,
        )

        self.executor = CapabilityExecutor(
            registry=self.registry,
            discovery_engine=self.discovery,
        )

    def attach_discovery(
        self,
        discovery_engine,
    ) -> None:

        self.discovery = discovery_engine
        self._refresh_executor()

    def activate(
        self,
        capability: RuntimeCapability,
    ) -> None:

        capability.enabled = True

        self.registry.register(
            capability,
            replace=True,
        )

        self.persistence.save(
            self.registry
        )

    def deactivate(
        self,
        capability_id: str,
    ) -> None:

        capability = self.registry.get(
            capability_id
        )

        capability.enabled = False

        self.persistence.save(
            self.registry
        )

    def execute(
        self,
        capability_id: str,
        *,
        context: dict | None = None,
    ) -> dict:

        return self.executor.execute(
            capability_id,
            context=context,
        )

    def status(self) -> dict:

        return self.registry.status()
