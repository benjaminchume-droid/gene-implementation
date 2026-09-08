from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationExample:

    prompt: str

    target: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class EvaluationResult:

    name: str

    score: float

    passed: bool

    examples: int

    metrics: dict[str, Any] = field(
        default_factory=dict
    )

    errors: list[str] = field(
        default_factory=list
    )


@dataclass
class BenchmarkReport:

    model_name: str

    results: list[EvaluationResult]

    aggregate_score: float

    passed: bool

    metadata: dict[str, Any] = field(
        default_factory=dict
    )
