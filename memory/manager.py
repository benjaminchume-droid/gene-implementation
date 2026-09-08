from __future__ import annotations

from .retriever import MemoryRetriever
from .store import MemoryStore


class MemoryManager:
    """High-level memory interface used by the Gene runtime."""

    def __init__(self, root: str = "gene/memory/data") -> None:
        self.store = MemoryStore(root)
        self.retriever = MemoryRetriever(self.store)

    def remember(
        self,
        content: str,
        *,
        memory_type: str = "interaction",
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> dict:

        return self.store.add(
            memory_type,
            content,
            importance=importance,
            metadata=metadata,
        )

    def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[dict]:

        return self.retriever.search(
            query,
            limit=limit,
            memory_type=memory_type,
        )

    def status(self) -> dict:
        return {
            "memory_count": self.store.count(),
            "storage": str(self.store.memory_file),
        }
