from __future__ import annotations

from collections import Counter


class ExperienceAnalyzer:

    def analyze(
        self,
        experiences: list[dict],
    ) -> dict:

        if not experiences:
            return {
                "total": 0,
                "success_rate": 0.0,
                "patterns": [],
                "failure_patterns": [],
            }

        successful = [
            item
            for item in experiences
            if item.get("success") is True
        ]

        failed = [
            item
            for item in experiences
            if item.get("success") is False
        ]

        capability_counts = Counter()

        for item in successful:

            for capability in item.get(
                "capability_ids",
                [],
            ):
                capability_counts[capability] += 1

        error_counts = Counter()

        for item in failed:

            for error in item.get(
                "errors",
                [],
            ):
                if isinstance(error, dict):
                    key = str(
                        error.get(
                            "type",
                            "unknown",
                        )
                    )
                else:
                    key = str(error)

                error_counts[key] += 1

        return {
            "total":
                len(experiences),

            "success_rate":
                len(successful)
                / len(experiences),

            "patterns": [
                {
                    "capability":
                        name,
                    "uses":
                        count,
                }
                for name, count
                in capability_counts.most_common()
            ],

            "failure_patterns": [
                {
                    "error":
                        name,
                    "occurrences":
                        count,
                }
                for name, count
                in error_counts.most_common()
            ],
        }
