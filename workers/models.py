from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from typing import Any


class WorkerState(str, Enum):
    REGISTERED = "registered"
    READY = "ready"
    RUNNING = "running"
    DISABLED = "disabled"
    FAILED = "failed"


class WorkerCapability(str, Enum):
    REASONING = "reasoning"
    CODING = "coding"
    RESEARCH = "research"
    VISION = "vision"
    MATH = "math"
    BROWSER = "browser"
    FILESYSTEM = "filesystem"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    CUSTOM = "custom"


@dataclass
class WorkerDefinition:
    name: str
    purpose: str

    capabilities: list[str] = field(
        default_factory=list
    )

    backend: str = "runtime"
    model: str | None = None

    enabled: bool = True
    state: WorkerState = WorkerState.REGISTERED

    reasoning_budget: int = 100
    context_budget: int = 16000
    memory_budget: int = 16
    tool_budget: int = 8

    parent: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )


@dataclass
class WorkerResult:
    success: bool
    worker: str

    output: Any = None

    error: str | None = None

    metrics: dict[str, Any] = field(
        default_factory=dict
    )
