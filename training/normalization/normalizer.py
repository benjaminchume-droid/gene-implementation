from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .competency import (
    PreliminaryCompetencyEstimator,
)
from .extractor import (
    GenericTextExtractor,
)
from .models import (
    NormalizedTrainingRecord,
)


class DatasetNormalizer:

    def __init__(
        self,
        extractor:
            GenericTextExtractor | None = None,
        competency:
            PreliminaryCompetencyEstimator | None = None,
    ) -> None:

        self.extractor = (
            extractor
            or GenericTextExtractor()
        )

        self.competency = (
            competency
            or PreliminaryCompetencyEstimator()
        )

    def normalize_record(
        self,
        record: dict,
        *,
        source_type: str,
        source_id: str,
        line_number: int,
    ) -> NormalizedTrainingRecord | None:

        text = self.extractor.extract(
            record
        )

        if not text:
            return None

        metadata = {
            key: value
            for key, value
            in record.items()
            if (
                not isinstance(
                    value,
                    (
                        dict,
                        list,
                    ),
                )
                and key
                not in self.extractor.text_fields
            )
        }

        competency = (
            self.competency.estimate(
                text,
                metadata=metadata,
            )
        )

        return NormalizedTrainingRecord(
            text=text,
            source_type=source_type,
            source_id=source_id,
            quality=1.0,
            confidence=(
                competency.get(
                    "confidence",
                    0.5,
                )
            ),
            competency=competency,
            provenance={
                "source_file":
                    source_id,
                "line":
                    line_number,
            },
            metadata=metadata,
        )

    def normalize_file(
        self,
        source_path: str,
        output_path: str,
        *,
        source_type: str = "dataset",
    ) -> dict:

        source = Path(
            source_path
        )

        output = Path(
            output_path
        )

        if not source.exists():
            raise FileNotFoundError(
                str(source)
            )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        seen = set()

        records_seen = 0
        records_written = 0
        records_skipped = 0
        duplicates = 0

        with (
            source.open(
                "r",
                encoding="utf-8-sig",
            ) as input_handle,
            output.open(
                "w",
                encoding="utf-8",
            ) as output_handle,
        ):

            for line_number, line in enumerate(
                input_handle,
                start=1,
            ):

                records_seen += 1

                if not line.strip():
                    continue

                try:
                    record = json.loads(
                        line
                    )
                except json.JSONDecodeError:
                    records_skipped += 1
                    continue

                if not isinstance(
                    record,
                    dict,
                ):
                    records_skipped += 1
                    continue

                normalized = (
                    self.normalize_record(
                        record,
                        source_type=
                            source_type,
                        source_id=
                            str(source),
                        line_number=
                            line_number,
                    )
                )

                if normalized is None:
                    records_skipped += 1
                    continue

                fingerprint = (
                    self._fingerprint(
                        normalized.text
                    )
                )

                if fingerprint in seen:
                    duplicates += 1
                    continue

                seen.add(
                    fingerprint
                )

                output_handle.write(
                    json.dumps(
                        normalized.to_dict(),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                records_written += 1

        return {
            "source":
                str(source),
            "output":
                str(output),
            "records_seen":
                records_seen,
            "records_written":
                records_written,
            "records_skipped":
                records_skipped,
            "duplicates":
                duplicates,
        }

    @staticmethod
    def _fingerprint(
        text: str,
    ) -> str:

        import hashlib

        normalized = " ".join(
            text.lower().split()
        )

        return hashlib.sha256(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()
