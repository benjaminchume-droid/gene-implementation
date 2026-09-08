from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Resource:
    id: str
    kind: str
    name: str
    description: str = ""

    actions: list[dict[str, Any]] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Observation:
    resource_id: str
    data: dict[str, Any]

    source: str = ""
    confidence: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class DiscoveryEnvironment(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def discover(self) -> list[Resource]:
        raise NotImplementedError

    @abstractmethod
    def observe(
        self,
        resource: Resource,
    ) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        resource: Resource,
        action: dict[str, Any],
    ) -> Any:
        raise NotImplementedError
