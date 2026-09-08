from __future__ import annotations

from .types import Capability


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        if capability.name in self._capabilities:
            raise ValueError(
                f"Capability already registered: {capability.name}"
            )

        self._capabilities[capability.name] = capability

    def get(self, name: str) -> Capability:
        try:
            return self._capabilities[name]
        except KeyError:
            raise KeyError(f"Unknown capability: {name}") from None

    def has(self, name: str) -> bool:
        return name in self._capabilities

    def list(self) -> list[Capability]:
        return list(self._capabilities.values())
