from __future__ import annotations

from .models import SubAgentSpec


class SubAgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, SubAgentSpec] = {}

    def register(self, spec: SubAgentSpec) -> None:
        if not spec.role.strip():
            raise ValueError("Sub-agent role cannot be empty.")

        if spec.role in self._agents:
            raise ValueError(
                f"Sub-agent role already registered: {spec.role}"
            )

        self._agents[spec.role] = spec

    def get(self, role: str) -> SubAgentSpec | None:
        return self._agents.get(role)

    def exists(self, role: str) -> bool:
        return role in self._agents

    def all(self) -> list[SubAgentSpec]:
        return list(self._agents.values())

    def roles(self) -> list[str]:
        return list(self._agents.keys())

    def unregister(self, role: str) -> None:
        self._agents.pop(role, None)
