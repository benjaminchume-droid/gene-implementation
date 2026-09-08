from gene.levels import (
    GeneLevel,
    LevelManager,
)
from gene.levels.power import (
    PowerController,
)


def test_level_order_and_budgets():

    insight = LevelManager(
        GeneLevel.INSIGHT
    ).budget()

    scholar = LevelManager(
        GeneLevel.SCHOLAR
    ).budget()

    transcendent = LevelManager(
        GeneLevel.TRANSCENDENT
    ).budget()

    omniscient = LevelManager(
        GeneLevel.OMNISCIENT
    ).budget()

    assert (
        insight.reasoning_units
        < scholar.reasoning_units
        < transcendent.reasoning_units
        < omniscient.reasoning_units
    )

    assert (
        insight.context_budget
        < scholar.context_budget
        < transcendent.context_budget
        <= omniscient.context_budget
    )


def test_level_resource_limits():

    manager = LevelManager(
        GeneLevel.SCHOLAR
    )

    assert manager.can_use_tools(10)
    assert not manager.can_use_tools(100)

    assert manager.can_retrieve(20)
    assert not manager.can_retrieve(100)

    assert manager.can_spawn_workers(2)
    assert not manager.can_spawn_workers(10)


def test_power_controller():

    controller = PowerController()

    assert (
        controller.level
        == GeneLevel.INSIGHT
    )

    result = controller.set_level(
        "transcendent"
    )

    assert result["success"] is True
    assert result["level"] == (
        "transcendent"
    )
    assert (
        result["neural_weights_modified"]
        is False
    )


def test_power_status():

    controller = PowerController(
        GeneLevel.OMNISCIENT
    )

    status = controller.status()

    assert status["level"] == "omniscient"
    assert status["budget"]["worker_count"] == 16
