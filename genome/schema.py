from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenomeModule:
    name: str
    category: str
    enabled: bool = True
    priority: int = 0
    version: str = "1.0"
    configuration: dict[str, Any] = field(default_factory=dict)


@dataclass
class Genome:
    name: str
    version: str
    description: str
    modules: list[GenomeModule] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_module(self, module: GenomeModule) -> None:
        if any(existing.name == module.name for existing in self.modules):
            raise ValueError(f"Genome module already exists: {module.name}")

        self.modules.append(module)

    def get_module(self, name: str) -> GenomeModule | None:
        for module in self.modules:
            if module.name == name:
                return module
        return None

    def enabled_modules(self) -> list[GenomeModule]:
        return [
            module
            for module in self.modules
            if module.enabled
        ]
