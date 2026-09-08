from __future__ import annotations

import json
import uuid

from dataclasses import asdict
from pathlib import Path

from .models import MemoryRecord, MemoryType


class MemoryStoreV2:

    def __init__(
        self,
        path: str = (
            "gene/data/memory/v2/records.jsonl"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _read_all(self) -> list[MemoryRecord]:

        if not self.path.exists():
            return []

        records = []

        with self.path.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:

            for line in handle:

                line = line.strip()

                if not line:
                    continue

                data = json.loads(line)

                records.append(
                    MemoryRecord(
                        id=data["id"],
                        memory_type=MemoryType(
                            data["memory_type"]
                        ),
                        content=data["content"],
                        importance=data.get(
                            "importance",
                            0.5,
                        ),
                        confidence=data.get(
                            "confidence",
                            0.5,
                        ),
                        source=data.get(
                            "source"
                        ),
                        namespace=data.get(
                            "namespace",
                            "default",
                        ),
                        tags=data.get(
                            "tags",
                            [],
                        ),
                        metadata=data.get(
                            "metadata",
                            {},
                        ),
                        created_at=data.get(
                            "created_at"
                        ),
                        updated_at=data.get(
                            "updated_at"
                        ),
                        access_count=data.get(
                            "access_count",
                            0,
                        ),
                        last_accessed_at=data.get(
                            "last_accessed_at"
                        ),
                    )
                )

        return records

    def _rewrite(
        self,
        records: list[MemoryRecord],
    ) -> None:

        tmp = self.path.with_suffix(
            ".tmp"
        )

        with tmp.open(
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

        tmp.replace(self.path)

    def add(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        confidence: float = 0.5,
        source: str | None = None,
        namespace: str = "default",
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:

        importance = max(
            0.0,
            min(1.0, importance),
        )

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        record = MemoryRecord(
            id=str(uuid.uuid4()),
            memory_type=memory_type,
            content=content,
            importance=importance,
            confidence=confidence,
            source=source,
            namespace=namespace,
            tags=tags or [],
            metadata=metadata or {},
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                )
                + "\n"
            )

        return record

    def get(
        self,
        record_id: str,
    ) -> MemoryRecord | None:

        for record in self._read_all():

            if record.id == record_id:
                return record

        return None

    def list(
        self,
        memory_type: MemoryType | None = None,
        namespace: str | None = None,
    ) -> list[MemoryRecord]:

        records = self._read_all()

        if memory_type is not None:
            records = [
                record
                for record in records
                if record.memory_type
                == memory_type
            ]

        if namespace is not None:
            records = [
                record
                for record in records
                if record.namespace
                == namespace
            ]

        return records

    def update(
        self,
        record_id: str,
        **changes,
    ) -> MemoryRecord:

        records = self._read_all()

        target = None

        for record in records:

            if record.id == record_id:
                target = record
                break

        if target is None:
            raise KeyError(
                f"Unknown memory record: "
                f"{record_id}"
            )

        for key, value in changes.items():

            if not hasattr(
                target,
                key,
            ):
                raise ValueError(
                    f"Unknown memory field: "
                    f"{key}"
                )

            setattr(
                target,
                key,
                value,
            )

        target.updated_at = (
            __import__(
                "datetime"
            )
            .datetime.now(
                timezone.utc
            )
            .isoformat()
        )

        self._rewrite(records)

        return target

    def touch(
        self,
        record_id: str,
    ) -> MemoryRecord:

        record = self.get(record_id)

        if record is None:
            raise KeyError(
                record_id
            )

        record.access_count += 1
        record.last_accessed_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        self.update(
            record.id,
            access_count=
                record.access_count,
            last_accessed_at=
                record.last_accessed_at,
        )

        return record

    def delete(
        self,
        record_id: str,
    ) -> bool:

        records = self._read_all()

        filtered = [
            record
            for record in records
            if record.id != record_id
        ]

        changed = (
            len(filtered)
            != len(records)
        )

        if changed:
            self._rewrite(
                filtered
            )

        return changed

    def count(self) -> int:
        return len(
            self._read_all()
        )

    def status(self) -> dict:

        records = self._read_all()

        by_type: dict[str, int] = {}

        for record in records:

            key = record.memory_type.value

            by_type[key] = (
                by_type.get(
                    key,
                    0,
                )
                + 1
            )

        return {
            "count":
                len(records),
            "by_type":
                by_type,
            "storage":
                str(self.path),
        }
