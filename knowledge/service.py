from __future__ import annotations

from .database import KnowledgeDatabase


class KnowledgeService:

    def __init__(
        self,
        database: KnowledgeDatabase | None = None,
    ) -> None:

        self.database = (
            database
            or KnowledgeDatabase()
        )

    def remember_fact(
        self,
        topic: str,
        content: str,
        *,
        source: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.8,
    ):

        return self.database.add(
            topic=topic,
            content=content,
            source=source,
            record_type="fact",
            tags=tags,
            confidence=confidence,
        )

    def remember_procedure(
        self,
        topic: str,
        content: str,
        *,
        source: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.8,
    ):

        return self.database.add(
            topic=topic,
            content=content,
            source=source,
            record_type="procedure",
            tags=tags,
            confidence=confidence,
        )

    def remember_entity(
        self,
        topic: str,
        content: str,
        *,
        source: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.8,
    ):

        return self.database.add(
            topic=topic,
            content=content,
            source=source,
            record_type="entity",
            tags=tags,
            confidence=confidence,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        record_type: str | None = None,
    ):

        return self.database.search(
            query,
            limit=limit,
            record_type=record_type,
        )

    def status(self) -> dict:
        return {
            "records": self.database.count(),
            "database": str(
                self.database.path
            ),
        }
