from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class EvaluationCase:
    name: str
    input: Any
    expected: Any = None
    weight: float = 1.0
    evaluator: Callable[
        [Any, Any],
        float
    ] | None = None


@dataclass
class EvaluationResult:
    score: float
    passed: bool
    case_scores: dict[str, float] = field(
        default_factory=dict
    )
    failures: list[str] = field(
        default_factory=list
    )


class ParameterEvaluator:

    def __init__(
        self,
        runner: Callable[
            [dict[str, Any], Any],
            Any
        ],
    ) -> None:

        self.runner = runner

    @staticmethod
    def _default_score(
        actual: Any,
        expected: Any,
    ) -> float:

        if expected is None:
            return 1.0 if actual is not None else 0.0

        return (
            1.0
            if actual == expected
            else 0.0
        )

    def evaluate(
        self,
        parameters: dict[str, Any],
        cases: list[EvaluationCase],
        minimum_score: float = 0.80,
    ) -> EvaluationResult:

        if not cases:
            raise ValueError(
                "At least one evaluation case is required."
            )

        case_scores = {}
        failures = []

        total_weight = 0.0
        weighted_score = 0.0

        for case in cases:

            actual = self.runner(
                parameters,
                case.input,
            )

            scorer = (
                case.evaluator
                or self._default_score
            )

            score = float(
                scorer(
                    actual,
                    case.expected,
                )
            )

            score = max(
                0.0,
                min(1.0, score),
            )

            weight = max(
                0.0,
                case.weight,
            )

            case_scores[
                case.name
            ] = score

            total_weight += weight
            weighted_score += (
                score * weight
            )

            if score < minimum_score:
                failures.append(
                    case.name
                )

        final_score = (
            weighted_score / total_weight
            if total_weight > 0
            else 0.0
        )

        return EvaluationResult(
            score=final_score,
            passed=(
                final_score
                >= minimum_score
            ),
            case_scores=case_scores,
            failures=failures,
        )
