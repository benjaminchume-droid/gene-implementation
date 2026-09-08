from __future__ import annotations

from dataclasses import replace

from .backends import (
    RuntimeWorkerBackend,
    WorkerBackend,
)
from .models import (
    WorkerDefinition,
    WorkerResult,
    WorkerState,
)
from .registry import WorkerRegistry


class WorkerRuntime:

    def __init__(
        self,
        registry: WorkerRegistry | None = None,
    ) -> None:

        self.registry = (
            registry
            or WorkerRegistry()
        )

        self.backends: dict[
            str,
            WorkerBackend,
        ] = {
            "runtime":
                RuntimeWorkerBackend(),
        }

    def register_backend(
        self,
        backend: WorkerBackend,
    ) -> None:

        self.backends[
            backend.name
        ] = backend

    def create(
        self,
        name: str,
        purpose: str,
        capabilities: list[str] | None = None,
        backend: str = "runtime",
        model: str | None = None,
        reasoning_budget: int = 100,
        context_budget: int = 16000,
        memory_budget: int = 16,
        tool_budget: int = 8,
        parent: str | None = None,
        metadata: dict | None = None,
    ) -> WorkerDefinition:

        if backend not in self.backends:
            raise ValueError(
                f"Unknown worker backend: "
                f"{backend}"
            )

        worker = WorkerDefinition(
            name=name,
            purpose=purpose,
            capabilities=capabilities or [],
            backend=backend,
            model=model,
            reasoning_budget=reasoning_budget,
            context_budget=context_budget,
            memory_budget=memory_budget,
            tool_budget=tool_budget,
            parent=parent,
            metadata=metadata or {},
            state=WorkerState.READY,
        )

        return self.registry.register(
            worker
        )

    def run(
        self,
        name: str,
        task: str,
        context: dict | None = None,
    ) -> WorkerResult:

        worker = self.registry.get(
            name
        )

        if not worker.enabled:
            return WorkerResult(
                success=False,
                worker=name,
                error=(
                    "Worker is disabled."
                ),
            )

        backend = self.backends.get(
            worker.backend
        )

        if backend is None:
            return WorkerResult(
                success=False,
                worker=name,
                error=(
                    f"Backend unavailable: "
                    f"{worker.backend}"
                ),
            )

        worker.state = WorkerState.RUNNING

        try:

            result = backend.execute(
                worker,
                task,
                context or {},
            )

            worker.state = (
                WorkerState.READY
                if result.success
                else WorkerState.FAILED
            )

            return result

        except Exception as exc:

            worker.state = (
                WorkerState.FAILED
            )

            return WorkerResult(
                success=False,
                worker=name,
                error=str(exc),
            )

    def disable(
        self,
        name: str,
    ) -> WorkerDefinition:

        return self.registry.disable(
            name
        )

    def enable(
        self,
        name: str,
    ) -> WorkerDefinition:

        return self.registry.enable(
            name
        )

    def remove(
        self,
        name: str,
    ) -> bool:

        return self.registry.remove(
            name
        )

    def compatible_workers(
        self,
        capability: str,
    ) -> list[WorkerDefinition]:

        return [
            worker
            for worker
            in self.registry.list(
                capability
            )
            if worker.enabled
        ]

    def status(self) -> dict:

        return self.registry.status()
