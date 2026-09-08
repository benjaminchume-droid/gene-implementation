from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .records import KnowledgeRecord


class KnowledgeDatabase:

    def __init__(
        self,
        path: str = "gene/data/knowledge/gene.db",
    ) -> None:

        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.path
        )

        self.connection.row_factory = sqlite3.Row

        self._initialize()

    def _initialize(self) -> None:

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                record_type TEXT NOT NULL DEFAULT 'fact',
                tags TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_topic
            ON knowledge(topic)
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_type
            ON knowledge(record_type)
            """
        )

        self.connection.commit()

    def add(
        self,
        topic: str,
        content: str,
        *,
        source: str | None = None,
        record_type: str = "fact",
        tags: list[str] | None = None,
        confidence: float = 0.5,
    ) -> KnowledgeRecord:

        from datetime import datetime, timezone

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        tags = tags or []

        cursor = self.connection.execute(
            """
            INSERT INTO knowledge (
                topic,
                content,
                source,
                record_type,
                tags,
                confidence,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic,
                content,
                source,
                record_type,
                json.dumps(tags),
                max(0.0, min(1.0, confidence)),
                timestamp,
                timestamp,
            ),
        )

        self.connection.commit()

        return KnowledgeRecord(
            id=cursor.lastrowid,
            topic=topic,
            content=content,
            source=source,
            record_type=record_type,
            tags=tags,
            confidence=confidence,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def get(self, record_id: int) -> KnowledgeRecord | None:

        row = self.connection.execute(
            """
            SELECT *
            FROM knowledge
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        record_type: str | None = None,
    ) -> list[KnowledgeRecord]:

        query = query.strip()

        if not query:
            return []

        pattern = f"%{query}%"

        if record_type:
            rows = self.connection.execute(
                """
                SELECT *
                FROM knowledge
                WHERE record_type = ?
                  AND (
                      topic LIKE ?
                      OR content LIKE ?
                      OR tags LIKE ?
                  )
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (
                    record_type,
                    pattern,
                    pattern,
                    pattern,
                    limit,
                ),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT *
                FROM knowledge
                WHERE (
                    topic LIKE ?
                    OR content LIKE ?
                    OR tags LIKE ?
                )
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (
                    pattern,
                    pattern,
                    pattern,
                    limit,
                ),
            ).fetchall()

        return [
            self._row_to_record(row)
            for row in rows
        ]

    def update(
        self,
        record_id: int,
        *,
        topic: str | None = None,
        content: str | None = None,
        source: str | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
    ) -> KnowledgeRecord | None:

        existing = self.get(record_id)

        if existing is None:
            return None

        from datetime import datetime, timezone

        updated = datetime.now(
            timezone.utc
        ).isoformat()

        new_topic = (
            topic
            if topic is not None
            else existing.topic
        )

        new_content = (
            content
            if content is not None
            else existing.content
        )

        new_source = (
            source
            if source is not None
            else existing.source
        )

        new_confidence = (
            confidence
            if confidence is not None
            else existing.confidence
        )

        new_tags = (
            tags
            if tags is not None
            else existing.tags
        )

        self.connection.execute(
            """
            UPDATE knowledge
            SET topic = ?,
                content = ?,
                source = ?,
                tags = ?,
                confidence = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                new_topic,
                new_content,
                new_source,
                json.dumps(new_tags),
                max(0.0, min(1.0, new_confidence)),
                updated,
                record_id,
            ),
        )

        self.connection.commit()

        return self.get(record_id)

    def delete(self, record_id: int) -> bool:

        cursor = self.connection.execute(
            """
            DELETE FROM knowledge
            WHERE id = ?
            """,
            (record_id,),
        )

        self.connection.commit()

        return cursor.rowcount > 0

    def count(self) -> int:

        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM knowledge"
        ).fetchone()

        return int(row["count"])

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> KnowledgeRecord:

        return KnowledgeRecord(
            id=row["id"],
            topic=row["topic"],
            content=row["content"],
            source=row["source"],
            record_type=row["record_type"],
            tags=json.loads(row["tags"] or "[]"),
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def close(self) -> None:
        self.connection.close()
