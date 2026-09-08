from __future__ import annotations

from .base import Benchmark
from ..models import (
    EvaluationExample,
    EvaluationResult,
)


class GenerationBenchmark(Benchmark):

    @property
    def name(self) -> str:
        return "generation"

    def evaluate(
        self,
        model,
        tokenizer,
        examples,
    ) -> EvaluationResult:

        successes = 0
        outputs = []

        for example in examples:

            try:

                ids = tokenizer.encode(
                    example.prompt
                )

                if not ids:
                    continue

                output = model.generate(
                    ids,
                    max_new_tokens=32,
                    temperature=0.0,
                )

                text = tokenizer.decode(
                    output
                )

                valid = (
                    isinstance(
                        text,
                        str,
                    )
                    and bool(
                        text.strip()
                    )
                )

                if (
                    valid
                    and example.target
                ):
                    valid = (
                        example.target
                        .lower()
                        in text.lower()
                    )

                if valid:
                    successes += 1

                outputs.append(
                    {
                        "prompt":
                            example.prompt,
                        "output":
                            text,
                        "passed":
                            valid,
                    }
                )

            except Exception as exc:

                outputs.append(
                    {
                        "prompt":
                            example.prompt,
                        "error":
                            str(exc),
                        "passed":
                            False,
                    }
                )

        score = (
            successes / len(examples)
            if examples
            else 0.0
        )

        return EvaluationResult(
            name=self.name,
            score=score,
            passed=score > 0.0,
            examples=len(examples),
            metrics={
                "success_rate":
                    score,
                "outputs":
                    outputs,
            },
        )
