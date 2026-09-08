from gene.knowledge.v2 import (
    KnowledgeStoreV2,
    LongTermIntelligence,
)
from gene.memory.v2 import (
    MemoryStoreV2,
    MemoryType,
    MemoryRetriever,
)


def test_memory_store(tmp_path):

    store = MemoryStoreV2(
        str(
            tmp_path / "memory.jsonl"
        )
    )

    record = store.add(
        content="Important persistent information.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
        confidence=0.8,
    )

    assert record.content
    assert store.count() == 1


def test_memory_retrieval(tmp_path):

    store = MemoryStoreV2(
        str(
            tmp_path / "memory.jsonl"
        )
    )

    store.add(
        content="Python is a programming language.",
        importance=0.8,
        confidence=0.9,
    )

    store.add(
        content="The sky is blue.",
        importance=0.1,
        confidence=0.9,
    )

    retriever = MemoryRetriever(
        store
    )

    results = retriever.search(
        "Python programming",
        limit=2,
    )

    assert results
    assert "Python" in (
        results[0]
        .record
        .content
    )


def test_knowledge_store(tmp_path):

    store = KnowledgeStoreV2(
        str(
            tmp_path / "knowledge.db"
        )
    )

    store.add(
        "Python",
        "is_a",
        "programming language",
        confidence=0.95,
    )

    results = store.search(
        "Python"
    )

    assert len(results) == 1
    assert results[0].subject == "Python"


def test_long_term_intelligence(
    tmp_path,
):

    memory = MemoryStoreV2(
        str(
            tmp_path / "memory.jsonl"
        )
    )

    knowledge = KnowledgeStoreV2(
        str(
            tmp_path / "knowledge.db"
        )
    )

    system = LongTermIntelligence(
        memory_store=memory,
        knowledge_store=knowledge,
    )

    system.remember(
        "Persistent user information.",
        importance=0.9,
        confidence=0.9,
    )

    system.learn_fact(
        "Gene",
        "is",
        "modular",
        confidence=0.9,
    )

    context = system.retrieve(
        "Gene modular"
    )

    assert context.memories
    assert context.knowledge
