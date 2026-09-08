from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import (
    WorkerDefinition,
    WorkerResult,
)


class WorkerBackend(ABC):

    name = "base"

    @abstractmethod
    def execute(
        self,
        worker: WorkerDefinition,
        task: str,
        context: dict,
    ) -> WorkerResult:
        raise NotImplementedError
