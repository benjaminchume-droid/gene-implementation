from __future__ import annotations

from dataclasses import dataclass

from gene.memory.v2 import (
    MemoryRetriever,
    MemoryStoreV2,
    MemoryType,
)
from gene.knowledge.v2.store import (
    KnowledgeStoreV2,
)


@dataclass
class LongTermContext:

    memories: list
    knowledge: list

    def as_dict(self) -> dict:
        return {
            "memories": [
                item.record.content
                if hasattr(
                    item,
                    "record",
                )
                else str(item)
                for item in self.memories
            ],
            "knowledge": [
                {
                    "subject":
                        item.subject,
                    "predicate":
                        item.predicate,
                    "object":
                        item.object,
                    "confidence":
                        item.confidence,
                    "source":
                        item.source,
                }
                for item in self.knowledge
            ],
        }


class LongTermIntelligence:

    def __init__(
        self,
        memory_store=None,
        knowledge_store=None,
    ) -> None:

        self.memory = (
            memory_store
            or MemoryStoreV2()
        )

        self.memory_retriever = (
            MemoryRetriever(
                self.memory
            )
        )

        self.knowledge = (
            knowledge_store
            or KnowledgeStoreV2()
        )

    def remember(
        self,
        content: str,
        memory_type: MemoryType =
            MemoryType.SEMANTIC,
        importance: float = 0.5,
        confidence: float = 0.5,
        source: str | None = None,
        namespace: str = "default",
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ):

        return self.memory.add(
            content=content,
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
            source=source,
            namespace=namespace,
            tags=tags,
            metadata=metadata,
        )

    def learn_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        source: str | None = None,
        confidence: float = 0.5,
        namespace: str = "default",
        metadata: dict | None = None,
    ):

        return self.knowledge.add(
            subject=subject,
            predicate=predicate,
            object=object,
            source=source,
            confidence=confidence,
            namespace=namespace,
            metadata=metadata,
        )

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        memory_type=None,
    ) -> LongTermContext:

        memories = (
            self.memory_retriever.search(
                query=query,
                namespace=namespace,
                memory_type=memory_type,
                limit=limit,
            )
        )

        knowledge = self.knowledge.search(
            query=query,
            namespace=namespace,
            limit=limit,
        )

        return LongTermContext(
            memories=memories,
            knowledge=knowledge,
        )

    def status(self) -> dict:

        return {
            "memory":
                self.memory.status(),
            "knowledge":
                self.knowledge.status(),
        }
