from __future__ import annotations

import hashlib


class Deduplicator:

    def __init__(self) -> None:
        self.seen: set[str] = set()

    @staticmethod
    def fingerprint(
        text: str,
    ) -> str:

        normalized = (
            " ".join(
                text.lower().split()
            )
        )

        return hashlib.sha256(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()

    def accept(
        self,
        text: str,
    ) -> bool:

        key = self.fingerprint(
            text
        )

        if key in self.seen:
            return False

        self.seen.add(key)

        return True

    def add(
        self,
        text: str,
    ) -> str:

        key = self.fingerprint(
            text
        )

        self.seen.add(key)

        return key

    @property
    def count(self) -> int:
        return len(self.seen)
