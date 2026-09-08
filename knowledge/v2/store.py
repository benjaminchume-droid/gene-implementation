from __future__ import annotations

import json
import sqlite3

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class KnowledgeRecord:
    id: str
    subject: str
    predicate: str
    object: str

    source: str | None = None
    confidence: float = 0.5

    namespace: str = "default"

    metadata: dict[str, Any] | None = None

    created_at: str = ""


class KnowledgeStoreV2:

    def __init__(
        self,
        path: str = (
            "gene/data/knowledge/v2/knowledge.db"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.path
        )

        self._initialize()

    def _initialize(self) -> None:

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                source TEXT,
                confidence REAL NOT NULL,
                namespace TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_facts_subject
            ON facts(subject)
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_facts_predicate
            ON facts(predicate)
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_facts_namespace
            ON facts(namespace)
            """
        )

        self.connection.commit()

    def add(
        self,
        subject: str,
        predicate: str,
        object: str,
        source: str | None = None,
        confidence: float = 0.5,
        namespace: str = "default",
        metadata: dict | None = None,
    ) -> KnowledgeRecord:

        import uuid

        record = KnowledgeRecord(
            id=str(uuid.uuid4()),
            subject=subject,
            predicate=predicate,
            object=object,
            source=source,
            confidence=max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            namespace=namespace,
            metadata=metadata or {},
            created_at=
                datetime.now(
                    timezone.utc
                ).isoformat(),
        )

        self.connection.execute(
            """
            INSERT INTO facts (
                id,
                subject,
                predicate,
                object,
                source,
                confidence,
                namespace,
                metadata,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.subject,
                record.predicate,
                record.object,
                record.source,
                record.confidence,
                record.namespace,
                json.dumps(
                    record.metadata
                ),
                record.created_at,
            ),
        )

        self.connection.commit()

        return record

    def search(
        self,
        query: str,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[KnowledgeRecord]:

        terms = [
            term.strip()
            for term in query.split()
            if term.strip()
        ]

        if not terms:
            return []

        conditions = []
        values = []

        # A query is treated as a set of independent concepts.
        # Each term may match subject, predicate, or object.
        for term in terms:
            value = f"%{term}%"

            conditions.append(
                """
                (
                    subject LIKE ?
                    OR predicate LIKE ?
                    OR object LIKE ?
                )
                """
            )

            values.extend(
                [value, value, value]
            )

        where_clause = " OR ".join(
            conditions
        )

        if namespace is None:

            sql = f"""
                SELECT *
                FROM facts
                WHERE {where_clause}
                ORDER BY confidence DESC
                LIMIT ?
            """

            values.append(limit)

        else:

            sql = f"""
                SELECT *
                FROM facts
                WHERE namespace = ?
                  AND ({where_clause})
                ORDER BY confidence DESC
                LIMIT ?
            """

            values = [
                namespace,
                *values,
                limit,
            ]

        rows = self.connection.execute(
            sql,
            tuple(values),
        ).fetchall()

        return [
            KnowledgeRecord(
                id=row[0],
                subject=row[1],
                predicate=row[2],
                object=row[3],
                source=row[4],
                confidence=row[5],
                namespace=row[6],
                metadata=json.loads(
                    row[7] or "{}"
                ),
                created_at=row[8],
            )
            for row in rows
        ]

    def count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        )

    def status(self) -> dict:
        return {
            "count":
                self.count(),
            "database":
                str(self.path),
        }

    def close(self) -> None:
        self.connection.close()
