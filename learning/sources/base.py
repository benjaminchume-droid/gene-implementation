from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LearningSourceInfo:
    id: str
    kind: str
    name: str

    capabilities: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class LearningExchange:
    source_id: str
    request: str
    response: str

    observations: list[dict[str, Any]] = field(
        default_factory=list
    )

    actions: list[dict[str, Any]] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class LearningSource(ABC):

    @property
    @abstractmethod
    def info(self) -> LearningSourceInfo:
        raise NotImplementedError

    @abstractmethod
    def learn(
        self,
        request: str,
        context: dict[str, Any] | None = None,
    ) -> LearningExchange:
        raise NotImplementedError

    def available(self) -> bool:
        return True
