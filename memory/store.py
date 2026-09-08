from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryStore:
    """Persistent append-only memory storage for Gene."""

    def __init__(self, root: str | Path = "gene/memory/data") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.memory_file = self.root / "memories.jsonl"

    def add(
        self,
        memory_type: str,
        content: str,
        *,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        record = {
            "id": str(uuid.uuid4()),
            "type": memory_type,
            "content": content,
            "importance": max(0.0, min(1.0, importance)),
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with self.memory_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def all(self) -> list[dict[str, Any]]:
        if not self.memory_file.exists():
            return []

        records = []

        with self.memory_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return records

    def count(self) -> int:
        return len(self.all())
