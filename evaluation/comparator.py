from __future__ import annotations


class EvaluationComparator:

    def compare(
        self,
        baseline: dict,
        current: dict,
    ) -> dict:

        baseline_score = float(
            baseline.get(
                "aggregate_score",
                0.0,
            )
        )

        current_score = float(
            current.get(
                "aggregate_score",
                0.0,
            )
        )

        return {
            "baseline":
                baseline_score,
            "current":
                current_score,
            "delta":
                round(
                    current_score
                    - baseline_score,
                    12,
                ),
            "improved":
                current_score
                > baseline_score,
            "regressed":
                current_score
                < baseline_score,
        }
