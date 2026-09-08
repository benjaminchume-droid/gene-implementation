from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class ToolDefinition:
    name: str
    description: str
    handler: Callable[..., Any]
    requires_confirmation: bool = False


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition):
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def describe(self) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "requires_confirmation": tool.requires_confirmation,
            }
            for tool in self._tools.values()
        ]
