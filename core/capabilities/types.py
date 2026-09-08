from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class CapabilityRisk(str, Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    SYSTEM = "system"
    NETWORK = "network"


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    risk: CapabilityRisk = CapabilityRisk.READ
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Any] | None = None


@dataclass
class CapabilityResult:
    success: bool
    capability: str
    data: Any = None
    error: str | None = None
