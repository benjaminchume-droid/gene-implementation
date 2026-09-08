from __future__ import annotations

import hashlib

from .models import TrainingRecord


class TrainingRecordFilter:

    def __init__(
        self,
        minimum_quality: float = 0.70,
        minimum_confidence: float = 0.60,
    ) -> None:

        self.minimum_quality = minimum_quality
        self.minimum_confidence = minimum_confidence

    def accept(
        self,
        record: TrainingRecord,
    ) -> bool:

        if not record.accepted:
            return False

        if record.quality < self.minimum_quality:
            return False

        if record.confidence < self.minimum_confidence:
            return False

        if not record.text.strip():
            return False

        return True

    @staticmethod
    def fingerprint(
        text: str,
    ) -> str:

        normalized = " ".join(
            text.lower().split()
        )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()
