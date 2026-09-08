from __future__ import annotations

import json

from pathlib import Path

from .models import (
    BenchmarkReport,
    EvaluationExample,
)
from .benchmarks.registry import (
    BenchmarkRegistry,
)


class EvaluationRunner:

    def __init__(
        self,
        registry: BenchmarkRegistry | None = None,
    ) -> None:

        self.registry = (
            registry
            or BenchmarkRegistry()
        )

    def run(
        self,
        model,
        tokenizer,
        examples: list[EvaluationExample],
    ) -> BenchmarkReport:

        results = []

        for benchmark in (
            self.registry.list()
        ):

            results.append(
                benchmark.evaluate(
                    model,
                    tokenizer,
                    examples,
                )
            )

        aggregate = (
            sum(
                result.score
                for result
                in results
            )
            / len(results)
            if results
            else 0.0
        )

        return BenchmarkReport(
            model_name=(
                model.config.model_name
            ),
            results=results,
            aggregate_score=aggregate,
            passed=(
                bool(results)
                and all(
                    result.passed
                    for result
                    in results
                )
            ),
        )

    @staticmethod
    def save(
        report: BenchmarkReport,
        path: str,
    ) -> None:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model_name":
                report.model_name,
            "aggregate_score":
                report.aggregate_score,
            "passed":
                report.passed,
            "results": [
                {
                    "name":
                        result.name,
                    "score":
                        result.score,
                    "passed":
                        result.passed,
                    "examples":
                        result.examples,
                    "metrics":
                        result.metrics,
                    "errors":
                        result.errors,
                }
                for result
                in report.results
            ],
            "metadata":
                report.metadata,
        }

        target.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
