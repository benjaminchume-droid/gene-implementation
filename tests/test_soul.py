from pathlib import Path

from gene.soul import SoulManager


def test_soul_boots(tmp_path: Path) -> None:
    manager = SoulManager(
        str(tmp_path / "soul.json")
    )

    assert manager.identity()["name"] == "Gene"
    assert "assistant" in manager.active_sparks()


def test_soul_spark_activation(tmp_path: Path) -> None:
    manager = SoulManager(
        str(tmp_path / "soul.json")
    )

    manager.activate_spark("coding_partner")

    assert "coding_partner" in manager.active_sparks()


def test_soul_persistence(tmp_path: Path) -> None:
    path = str(tmp_path / "soul.json")

    manager = SoulManager(path)
    manager.activate_spark("brutally_honest")

    restored = SoulManager(path)

    assert (
        "brutally_honest"
        in restored.active_sparks()
    )
