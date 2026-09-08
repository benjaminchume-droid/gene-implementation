from __future__ import annotations

from ..models import (
    WorkerDefinition,
    WorkerResult,
)
from .base import WorkerBackend


class RuntimeWorkerBackend(
    WorkerBackend
):

    name = "runtime"

    def execute(
        self,
        worker: WorkerDefinition,
        task: str,
        context: dict,
    ) -> WorkerResult:

        return WorkerResult(
            success=True,
            worker=worker.name,
            output={
                "status":
                    "worker_ready",

                "task":
                    task,

                "message":
                    (
                        "Worker execution is ready "
                        "for a model backend."
                    ),
            },
            metrics={
                "backend":
                    self.name,

                "reasoning_budget":
                    worker.reasoning_budget,

                "context_budget":
                    worker.context_budget,

                "tool_budget":
                    worker.tool_budget,
            },
        )
