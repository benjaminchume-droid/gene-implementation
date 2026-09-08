from pathlib import Path

from gene.knowledge import KnowledgeDatabase, KnowledgeService


def test_knowledge_database(tmp_path: Path) -> None:

    db = KnowledgeDatabase(
        str(tmp_path / "knowledge.db")
    )

    record = db.add(
        topic="Gene",
        content="Gene is a modular AI architecture.",
        source="test",
        tags=["gene", "architecture"],
        confidence=0.95,
    )

    assert record.id is not None
    assert db.count() == 1

    found = db.search("modular")

    assert len(found) == 1
    assert found[0].topic == "Gene"


def test_knowledge_service(tmp_path: Path) -> None:

    service = KnowledgeService(
        KnowledgeDatabase(
            str(tmp_path / "knowledge.db")
        )
    )

    service.remember_fact(
        "Gene",
        "Gene uses modular capability systems.",
        tags=["architecture"],
    )

    results = service.search("capability")

    assert len(results) == 1
    assert results[0].record_type == "fact"
