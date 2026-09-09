from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SubAgentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubAgentSpec:
    name: str
    description: str = ""
    instructions: str = ""
    capabilities: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    model_id: str | None = None
    max_steps: int = 20
    timeout_seconds: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubAgentTask:
    id: str
    agent_name: str
    objective: str
    context: dict[str, Any] = field(default_factory=dict)
    instructions: str = ""
    tools: list[str] = field(default_factory=list)
    model_id: str | None = None
    max_steps: int = 20
    timeout_seconds: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubAgentResult:
    task_id: str
    status: SubAgentStatus
    output: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    steps: int = 0
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == SubAgentStatus.COMPLETED
