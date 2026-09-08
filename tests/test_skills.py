from pathlib import Path

from gene.skills import SkillManager
from gene.evolution import EvolutionEngine


def test_skill_creation(tmp_path: Path) -> None:

    manager = SkillManager(
        str(tmp_path / "skills.json")
    )

    skill = manager.create(
        name="coding",
        domain="software",
        description="General software engineering.",
        instructions="Analyze, implement, test and verify code.",
    )

    assert skill.name == "coding"
    assert skill.latest is not None
    assert skill.validated is False


def test_skill_branching(tmp_path: Path) -> None:

    manager = SkillManager(
        str(tmp_path / "skills.json")
    )

    manager.create(
        name="coding",
        domain="software",
        description="Software engineering.",
        instructions="General coding workflow.",
    )

    child = manager.branch(
        parent="coding",
        name="python",
        description="Python engineering.",
        instructions="Analyze and implement Python code.",
    )

    assert child.parent == "coding"
    assert child.level == 2


def test_skill_validation(tmp_path: Path) -> None:

    manager = SkillManager(
        str(tmp_path / "skills.json")
    )

    manager.create(
        name="coding",
        domain="software",
        description="Software engineering.",
        instructions="Code, test and verify.",
    )

    manager.validate(
        "coding",
        score=0.95,
        tests_passed=10,
        tests_failed=0,
    )

    skill = manager.registry.get(
        "coding"
    )

    assert skill is not None
    assert skill.validated is True


def test_evolution_engine(tmp_path: Path) -> None:

    manager = SkillManager(
        str(tmp_path / "skills.json")
    )

    engine = EvolutionEngine(manager)

    skill = engine.create_validated_skill(
        name="research",
        domain="research",
        description="Structured research.",
        instructions="Find, compare and verify sources.",
        score=0.90,
        tests_passed=5,
        tests_failed=0,
    )

    assert skill.validated is True
