from .models import SubAgentSpec


class SubAgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, SubAgentSpec] = {}

    def register(self, spec: SubAgentSpec, *, replace: bool = False) -> None:
        name = spec.name.strip()

        if not name:
            raise ValueError("Sub-agent name cannot be empty.")

        if not replace and name in self._agents:
            raise ValueError(f"Sub-agent already registered: {name}")

        if spec.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        if spec.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        self._agents[name] = spec

    def get(self, name: str) -> SubAgentSpec | None:
        return self._agents.get(name)

    def exists(self, name: str) -> bool:
        return name in self._agents

    def remove(self, name: str) -> None:
        self._agents.pop(name, None)

    def all(self) -> list[SubAgentSpec]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents.keys())

    def clear(self) -> None:
        self._agents.clear()
