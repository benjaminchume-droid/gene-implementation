from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KnowledgeRecord:
    content: str
    source: str
    tags: list[str]


class KnowledgeStore:
    def __init__(self) -> None:
        self.records: list[KnowledgeRecord] = []

    def add(
        self,
        content: str,
        source: str,
        tags: list[str] | None = None,
    ) -> None:
        self.records.append(
            KnowledgeRecord(
                content=content,
                source=source,
                tags=tags or [],
            )
        )

    def search(self, query: str) -> list[KnowledgeRecord]:
        terms = query.lower().split()

        return [
            record
            for record in self.records
            if any(
                term in record.content.lower()
                for term in terms
            )
        ]
