from __future__ import annotations

import json
import os

from pathlib import Path


def recommend(
    report_path: str = (
        "gene/data/training/resources/"
        "thread_scaling.json"
    ),
) -> dict:

    report = json.loads(
        Path(report_path).read_text(
            encoding="utf-8-sig"
        )
    )

    valid = [
        item
        for item in report["results"]
        if not item.get("failed")
    ]

    if not valid:
        raise RuntimeError(
            "No successful thread benchmark results."
        )

    # Highest measured throughput.
    best = max(
        valid,
        key=lambda item:
            item["tokens_per_second"],
    )

    cpu_count = (
        report.get(
            "logical_cpus"
        )
        or os.cpu_count()
        or 1
    )

    recommendation = {
        "logical_cpus":
            cpu_count,

        "threads":
            best["threads"],

        "context_length":
            best["context_length"],

        "measured_tokens_per_second":
            best["tokens_per_second"],

        "measured_step_seconds":
            best["total_seconds"],

        "memory_delta_mb":
            best["memory_delta_mb"],

        "policy": {
            "use_all_available_threads":
                True,

            "prefer_measured_configuration":
                True,

            "avoid_unverified_context":
                True,
        },
    }

    output = Path(
        "gene/data/training/resources/"
        "recommended_training_config.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            recommendation,
            indent=2,
        ),
        encoding="utf-8",
    )

    return recommendation


if __name__ == "__main__":

    result = recommend()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GENE RESOURCE RECOMMENDATION"
    )

    print(
        "=" * 72
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )
