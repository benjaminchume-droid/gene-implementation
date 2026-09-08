from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextItem:
    content: str
    source: str
    priority: float = 0.5
    token_estimate: int = 0
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ContextBundle:
    system: list[ContextItem] = field(
        default_factory=list
    )
    memories: list[ContextItem] = field(
        default_factory=list
    )
    knowledge: list[ContextItem] = field(
        default_factory=list
    )
    skills: list[ContextItem] = field(
        default_factory=list
    )
    tools: list[ContextItem] = field(
        default_factory=list
    )
    conversation: list[ContextItem] = field(
        default_factory=list
    )

    def all_items(self) -> list[ContextItem]:
        return (
            self.system
            + self.memories
            + self.knowledge
            + self.skills
            + self.tools
            + self.conversation
        )
