from __future__ import annotations

from typing import Any


class PluginRegistry:
    def __init__(self) -> None:
        self.plugins: dict[str, Any] = {}

    def register(self, name: str, plugin: Any) -> None:
        if name in self.plugins:
            raise ValueError(f"Plugin already registered: {name}")

        self.plugins[name] = plugin

    def get(self, name: str) -> Any:
        return self.plugins[name]

    def list(self) -> list[str]:
        return list(self.plugins)
