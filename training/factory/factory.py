from __future__ import annotations

import json

from dataclasses import asdict
from pathlib import Path

from .filter import TrainingRecordFilter
from .models import TrainingRecord


class TrainingDataFactory:

    def __init__(
        self,
        filter_engine: TrainingRecordFilter | None = None,
    ) -> None:

        self.filter = (
            filter_engine
            or TrainingRecordFilter()
        )

        self.records: list[
            TrainingRecord
        ] = []

        self.fingerprints: set[str] = set()

    def add(
        self,
        record: TrainingRecord,
    ) -> bool:

        if not self.filter.accept(record):
            return False

        fingerprint = self.filter.fingerprint(
            record.text
        )

        if fingerprint in self.fingerprints:
            return False

        self.fingerprints.add(fingerprint)
        self.records.append(record)

        return True

    def add_text(
        self,
        text: str,
        *,
        source_type: str,
        source_id: str,
        quality: float = 1.0,
        confidence: float = 1.0,
        metadata: dict | None = None,
    ) -> bool:

        return self.add(
            TrainingRecord(
                text=text,
                source_type=source_type,
                source_id=source_id,
                quality=quality,
                confidence=confidence,
                metadata=metadata or {},
            )
        )

    def source_counts(self) -> dict[str, int]:

        counts: dict[str, int] = {}

        for record in self.records:
            counts[record.source_type] = (
                counts.get(
                    record.source_type,
                    0,
                ) + 1
            )

        return counts

    def statistics(self) -> dict:

        return {
            "records":
                len(self.records),

            "sources":
                self.source_counts(),

            "average_quality":
                (
                    sum(
                        record.quality
                        for record in self.records
                    )
                    / len(self.records)
                    if self.records
                    else 0.0
                ),

            "average_confidence":
                (
                    sum(
                        record.confidence
                        for record in self.records
                    )
                    / len(self.records)
                    if self.records
                    else 0.0
                ),
        }

    def split(
        self,
        validation_ratio: float = 0.02,
    ) -> tuple[
        list[TrainingRecord],
        list[TrainingRecord],
    ]:

        if not 0.0 <= validation_ratio < 1.0:
            raise ValueError(
                "validation_ratio must be between 0 and 1."
            )

        total = len(self.records)
        validation_count = int(
            total * validation_ratio
        )

        validation = self.records[
            :validation_count
        ]

        train = self.records[
            validation_count:
        ]

        return train, validation

    def write_jsonl(
        self,
        records: list[TrainingRecord],
        path: str,
    ) -> None:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with target.open(
            "w",
            encoding="utf-8",
        ) as handle:

            for record in records:
                handle.write(
                    json.dumps(
                        asdict(record),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def build(
        self,
        train_path: str,
        validation_path: str,
        validation_ratio: float = 0.02,
    ) -> dict:

        train, validation = self.split(
            validation_ratio
        )

        self.write_jsonl(
            train,
            train_path,
        )

        self.write_jsonl(
            validation,
            validation_path,
        )

        return {
            "success": True,
            "train_records": len(train),
            "validation_records":
                len(validation),
            "statistics":
                self.statistics(),
            "train_path":
                train_path,
            "validation_path":
                validation_path,
        }
