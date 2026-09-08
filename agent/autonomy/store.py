from __future__ import annotations

import json

from dataclasses import asdict
from pathlib import Path

from .models import Mission


class MissionStore:

    def __init__(
        self,
        path: str = (
            "gene/data/agent/"
            "missions/missions.jsonl"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        mission: Mission,
    ) -> None:

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    asdict(mission),
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
            "missions":
                len(records),

            "completed":
                sum(
                    1
                    for record in records
                    if record.get("state")
                    == "completed"
                ),

            "failed":
                sum(
                    1
                    for record in records
                    if record.get("state")
                    == "failed"
                ),

            "blocked":
                sum(
                    1
                    for record in records
                    if record.get("state")
                    == "blocked"
                ),

            "storage":
                str(self.path),
        }
