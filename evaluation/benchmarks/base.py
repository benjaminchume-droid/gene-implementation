from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import (
    EvaluationExample,
    EvaluationResult,
)


class Benchmark(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        model,
        tokenizer,
        examples: list[EvaluationExample],
    ) -> EvaluationResult:
        raise NotImplementedError
