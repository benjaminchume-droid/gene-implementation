from __future__ import annotations

import random


class DatasetSplitter:

    def __init__(
        self,
        validation_ratio: float = 0.02,
        seed: int = 42,
    ) -> None:

        if not 0 <= validation_ratio < 1:
            raise ValueError(
                "validation_ratio must be "
                "between 0 and 1."
            )

        self.validation_ratio = (
            validation_ratio
        )

        self.seed = seed

    def assign(
        self,
        record_id: str,
    ) -> str:

        # Stable across reruns.
        value = (
            sum(
                ord(char)
                for char
                in record_id
            )
            + self.seed
        )

        bucket = (
            value % 10000
        ) / 10000

        return (
            "validation"
            if bucket
            < self.validation_ratio
            else "train"
        )
