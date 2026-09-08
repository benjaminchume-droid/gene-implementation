from __future__ import annotations

import re
from typing import Any


class MemoryRetriever:
    """Lightweight deterministic memory retrieval.

    This is deliberately model-independent.
    A vector/embedding backend can be added later without
    changing the MemoryStore interface.
    """

    def __init__(self, store) -> None:
        self.store = store

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:

        query_tokens = self._tokens(query)

        scored = []

        for memory in self.store.all():

            if memory_type and memory.get("type") != memory_type:
                continue

            memory_tokens = self._tokens(memory.get("content", ""))

            overlap = len(query_tokens & memory_tokens)

            if overlap == 0:
                continue

            score = overlap + float(memory.get("importance", 0.5))

            scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)

        return [memory for _, memory in scored[:limit]]
