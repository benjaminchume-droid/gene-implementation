from gene.evolution.evaluator import (
    EvaluationCase,
    ParameterEvaluator,
)
from gene.evolution.external import (
    ExternalParameterStore,
    EvolutionController,
)
from gene.evolution.guard import (
    NeuralWeightGuard,
)
from gene.evolution.base import (
    EvolutionInvariantError,
)


def test_parameter_evaluator():

    def runner(parameters, value):
        return parameters["answer"] == value

    evaluator = ParameterEvaluator(
        runner
    )

    result = evaluator.evaluate(
        {
            "answer": 42
        },
        [
            EvaluationCase(
                name="correct",
                input=42,
                expected=True,
            )
        ],
    )

    assert result.passed is True
    assert result.score == 1.0


def test_rejected_evolution(
    tmp_path,
):

    store = ExternalParameterStore(
        str(
            tmp_path / "parameters.json"
        )
    )

    controller = EvolutionController(
        store
    )

    proposal = controller.propose(
        namespace="routing",
        changes={
            "browser": 0.9
        },
        reason="test",
    )

    result = controller.activate(
        proposal,
        score=0.2,
        minimum_score=0.8,
    )

    assert result["success"] is False
    assert store.get("routing") == {}


def test_activation_and_rollback(
    tmp_path,
):

    store = ExternalParameterStore(
        str(
            tmp_path / "parameters.json"
        )
    )

    controller = EvolutionController(
        store
    )

    first = controller.propose(
        namespace="memory",
        changes={
            "threshold": 0.5
        },
        reason="first",
    )

    result = controller.activate(
        first,
        score=0.90,
    )

    assert result["success"] is True
    assert result["version"] == 1

    second = controller.propose(
        namespace="memory",
        changes={
            "threshold": 0.8
        },
        reason="second",
    )

    result = controller.activate(
        second,
        score=0.95,
    )

    assert result["version"] == 2
    assert (
        store.get("memory")["threshold"]
        == 0.8
    )

    rollback = controller.rollback(
        "memory",
        1,
    )

    assert rollback["success"] is True

    assert (
        store.get("memory")["threshold"]
        == 0.5
    )


def test_neural_weight_guard():

    guard = NeuralWeightGuard()

    guard.check_model_mutation(
        "inference"
    )

    try:
        guard.check_model_mutation(
            "optimizer_step"
        )
    except EvolutionInvariantError:
        return

    raise AssertionError(
        "Neural weight mutation was not blocked."
    )
