from __future__ import annotations

from .models import RuntimeCapability


class CapabilityRuntimeRegistry:

    def __init__(self) -> None:

        self.capabilities: dict[
            str,
            RuntimeCapability,
        ] = {}

    def register(
        self,
        capability: RuntimeCapability,
        *,
        replace: bool = False,
    ) -> None:

        if (
            capability.id
            in self.capabilities
            and not replace
        ):
            raise ValueError(
                f"Capability already exists: "
                f"{capability.id}"
            )

        self.capabilities[
            capability.id
        ] = capability

    def unregister(
        self,
        capability_id: str,
    ) -> bool:

        return (
            self.capabilities.pop(
                capability_id,
                None,
            )
            is not None
        )

    def get(
        self,
        capability_id: str,
    ) -> RuntimeCapability:

        try:
            return self.capabilities[
                capability_id
            ]
        except KeyError:
            raise KeyError(
                f"Unknown runtime capability: "
                f"{capability_id}"
            ) from None

    def find(
        self,
        name: str,
    ) -> list[RuntimeCapability]:

        value = name.lower()

        return [
            capability
            for capability
            in self.capabilities.values()
            if capability.enabled
            and (
                value
                in capability.name.lower()
                or value
                in capability.description.lower()
            )
        ]

    def enabled(self) -> list[RuntimeCapability]:

        return [
            capability
            for capability
            in self.capabilities.values()
            if capability.enabled
        ]

    def list(self) -> list[RuntimeCapability]:

        return list(
            self.capabilities.values()
        )

    def status(self) -> dict:

        return {
            "total":
                len(self.capabilities),

            "enabled":
                len(self.enabled()),

            "capabilities": [
                {
                    "id":
                        capability.id,
                    "name":
                        capability.name,
                    "kind":
                        capability.kind,
                    "confidence":
                        capability.confidence,
                    "enabled":
                        capability.enabled,
                }
                for capability
                in self.capabilities.values()
            ],
        }
