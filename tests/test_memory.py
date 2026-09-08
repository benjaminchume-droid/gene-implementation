from pathlib import Path

from gene.memory import MemoryManager


def test_memory_roundtrip(tmp_path: Path) -> None:
    manager = MemoryManager(str(tmp_path / "memory"))

    manager.remember(
        "The user is building Gene AI.",
        memory_type="user",
        importance=0.9,
    )

    results = manager.recall("Gene AI")

    assert len(results) == 1
    assert results[0]["type"] == "user"
    assert "Gene AI" in results[0]["content"]


def test_memory_status(tmp_path: Path) -> None:
    manager = MemoryManager(str(tmp_path / "memory"))

    assert manager.status()["memory_count"] == 0
