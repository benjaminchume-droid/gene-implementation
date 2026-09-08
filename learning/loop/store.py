from __future__ import annotations

import json
from pathlib import Path

from .models import Experience


class ExperienceStore:

    def __init__(
        self,
        path: str = (
            "gene/data/learning/"
            "experiences/experiences.jsonl"
        ),
    ) -> None:

        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        experience: Experience,
    ) -> None:

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    experience.to_dict(),
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

    def successful(self) -> list[dict]:

        return [
            record
            for record in self.list()
            if record.get("success") is True
        ]

    def failed(self) -> list[dict]:

        return [
            record
            for record in self.list()
            if record.get("success") is False
        ]

    def status(self) -> dict:

        records = self.list()

        return {
            "total": len(records),
            "successful":
                len(self.successful()),
            "failed":
                len(self.failed()),
            "storage":
                str(self.path),
        }
