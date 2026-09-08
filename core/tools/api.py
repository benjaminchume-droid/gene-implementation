from __future__ import annotations

from typing import Any


class APIProvider:
    def __init__(self) -> None:
        self.providers: dict[str, Any] = {}

    def register(self, name: str, provider: Any) -> None:
        self.providers[name] = provider

    def list_providers(self) -> list[str]:
        return list(self.providers)
