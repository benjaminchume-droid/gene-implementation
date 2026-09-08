from pathlib import Path

from gene.evolution.external import (
    EvolutionController,
    ExternalParameterStore,
)


def test_external_parameter_persistence(
    tmp_path: Path,
):

    store = ExternalParameterStore(
        str(tmp_path / "parameters.json")
    )

    evolution = EvolutionController(store)

    proposal = evolution.propose(
        namespace="memory",
        changes={
            "importance_threshold": 0.75
        },
        reason="Improve memory selection.",
    )

    result = evolution.evaluate_and_activate(
        proposal,
        score=0.92,
    )

    assert result["success"] is True
    assert (
        result["neural_weights_modified"]
        is False
    )

    second = ExternalParameterStore(
        str(tmp_path / "parameters.json")
    )

    assert (
        second.get("memory")[
            "importance_threshold"
        ]
        == 0.75
    )


def test_external_parameter_rejection(
    tmp_path: Path,
):

    store = ExternalParameterStore(
        str(tmp_path / "parameters.json")
    )

    evolution = EvolutionController(store)

    proposal = evolution.propose(
        namespace="routing",
        changes={"screen": 0.99},
        reason="Test.",
    )

    result = evolution.evaluate_and_activate(
        proposal,
        score=0.20,
    )

    assert result["success"] is False
    assert store.get("routing") == {}