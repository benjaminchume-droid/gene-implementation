from __future__ import annotations

import json

from dataclasses import asdict
from pathlib import Path


class PromotionStore:

    def __init__(
        self,
        root: str = (
            "gene/data/evolution/promotion"
        ),
    ) -> None:

        self.root = Path(root)

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _append(
        self,
        filename: str,
        value: dict,
    ) -> None:

        path = (
            self.root
            / filename
        )

        with path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                )
                + "\n"
            )

    def save_skill(
        self,
        candidate,
    ) -> None:

        self._append(
            "skills.jsonl",
            asdict(candidate),
        )

    def save_gene(
        self,
        gene,
    ) -> None:

        self._append(
            "genes.jsonl",
            asdict(gene),
        )

    def save_genome(
        self,
        genome,
    ) -> None:

        self._append(
            "genomes.jsonl",
            asdict(genome),
        )

    def _read(
        self,
        filename: str,
    ) -> list[dict]:

        path = (
            self.root
            / filename
        )

        if not path.exists():
            return []

        records = []

        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:

            for line in handle:

                if line.strip():
                    records.append(
                        json.loads(line)
                    )

        return records

    def skills(self):
        return self._read(
            "skills.jsonl"
        )

    def genes(self):
        return self._read(
            "genes.jsonl"
        )

    def genomes(self):
        return self._read(
            "genomes.jsonl"
        )

    def status(self) -> dict:

        return {
            "skills":
                len(self.skills()),
            "genes":
                len(self.genes()),
            "genomes":
                len(self.genomes()),
            "storage":
                str(self.root),
        }
