from __future__ import annotations

import math
import re

from dataclasses import dataclass

from .models import MemoryRecord


@dataclass
class RetrievalResult:
    record: MemoryRecord
    score: float
    lexical_score: float
    importance_score: float
    confidence_score: float


class MemoryRetriever:

    def __init__(
        self,
        store,
    ) -> None:
        self.store = store

    @staticmethod
    def _tokens(
        text: str,
    ) -> set[str]:

        return set(
            re.findall(
                r"\b\w+\b",
                text.lower(),
            )
        )

    @classmethod
    def _lexical_score(
        cls,
        query: str,
        content: str,
    ) -> float:

        q = cls._tokens(query)
        c = cls._tokens(content)

        if not q or not c:
            return 0.0

        overlap = len(
            q & c
        )

        return overlap / math.sqrt(
            len(q) * len(c)
        )

    def search(
        self,
        query: str,
        namespace: str | None = None,
        memory_type=None,
        limit: int = 10,
    ) -> list[RetrievalResult]:

        records = self.store.list(
            memory_type=memory_type,
            namespace=namespace,
        )

        results = []

        for record in records:

            lexical = (
                self._lexical_score(
                    query,
                    record.content,
                )
            )

            score = (
                lexical * 0.60
                + record.importance * 0.25
                + record.confidence * 0.15
            )

            results.append(
                RetrievalResult(
                    record=record,
                    score=score,
                    lexical_score=lexical,
                    importance_score=
                        record.importance,
                    confidence_score=
                        record.confidence,
                )
            )

        results.sort(
            key=lambda item:
                item.score,
            reverse=True,
        )

        return results[:limit]
