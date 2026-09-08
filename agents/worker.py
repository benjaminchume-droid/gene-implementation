from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkerBudget:
    reasoning_fraction: float = 0.25
    tool_fraction: float = 0.25
    context_fraction: float = 0.25
    memory_fraction: float = 0.25

    def validate(self) -> None:
        values = [
            self.reasoning_fraction,
            self.tool_fraction,
            self.context_fraction,
            self.memory_fraction,
        ]

        if any(v < 0 or v > 1 for v in values):
            raise ValueError(
                "Worker budget values must be between 0 and 1."
            )


@dataclass
class TaskWorker:
    name: str
    purpose: str
    budget: WorkerBudget

    def describe(self) -> dict:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "budget": {
                "reasoning": self.budget.reasoning_fraction,
                "tools": self.budget.tool_fraction,
                "context": self.budget.context_fraction,
                "memory": self.budget.memory_fraction,
            },
        }


class WorkerManager:

    def __init__(self):
        self.workers: dict[str, TaskWorker] = {}

    def register(self, worker: TaskWorker) -> None:
        worker.budget.validate()
        self.workers[worker.name] = worker

    def get(self, name: str) -> TaskWorker | None:
        return self.workers.get(name)

    def list(self) -> list[TaskWorker]:
        return list(self.workers.values())
