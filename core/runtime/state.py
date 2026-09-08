from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GeneState:
    identity: dict[str, Any]
    active_sparks: list[str]
    genome: dict[str, Any]
    memory_status: dict[str, Any]
    knowledge_status: dict[str, Any]
    skill_status: dict[str, Any]
    context_config: dict[str, Any]

    def snapshot(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "active_sparks": self.active_sparks,
            "genome": self.genome,
            "memory": self.memory_status,
            "knowledge": self.knowledge_status,
            "skills": self.skill_status,
            "context": self.context_config,
        }
