from __future__ import annotations

import json

from dataclasses import asdict

from pathlib import Path

from .models import Plan


class PlanStore:

    def __init__(
        self,
        path: str = (
            "gene/data/planning/"
            "plans.jsonl"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        plan: Plan,
    ) -> None:

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    asdict(plan),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    def list(self) -> list[dict]:

        if not self.path.exists():
            return []

        records = []

        with self.path.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:

            for line in handle:

                if line.strip():
                    records.append(
                        json.loads(line)
                    )

        return records

    def status(self) -> dict:

        records = self.list()

        return {
            "total":
                len(records),

            "completed":
                sum(
                    1
                    for record
                    in records
                    if record.get(
                        "status"
                    )
                    == "completed"
                ),

            "failed":
                sum(
                    1
                    for record
                    in records
                    if record.get(
                        "status"
                    )
                    == "failed"
                ),

            "storage":
                str(self.path),
        }
