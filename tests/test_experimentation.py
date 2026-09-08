from gene.discovery import (
    DiscoveryEngine,
    DiscoveryRegistry,
    Resource,
    SyntheticEnvironment,
)

from gene.experimentation import (
    Experiment,
    ExperimentAction,
    ExperimentPromoter,
    ExperimentRunner,
    VerificationEngine,
)


def build_engine():

    resource = Resource(
        id="unknown-resource",
        kind="unknown",
        name="Unknown",
        actions=[
            {
                "name":
                    "test",
            }
        ],
    )

    registry = DiscoveryRegistry()

    registry.register(
        SyntheticEnvironment(
            name="test-env",
            resources=[
                resource
            ],
        )
    )

    return (
        DiscoveryEngine(
            registry=registry
        ),
        resource,
    )


def test_experiment_execution():

    discovery, resource = (
        build_engine()
    )

    verification = VerificationEngine()

    verification.register(
        "success",
        lambda result, expected:
            result["success"] is True,
    )

    runner = ExperimentRunner(
        discovery,
        verification,
    )

    experiment = Experiment.create(
        objective=
            "Test an arbitrary capability.",
        resource_id=
            resource.id,
    )

    experiment.actions.append(
        ExperimentAction(
            action={
                "name":
                    "test",
            },
            expected={
                "success":
                    True,
            },
        )
    )

    result = runner.run(
        experiment
    )

    assert (
        result.status
        == "verified"
    )

    assert result.observations


def test_failed_experiment():

    discovery, resource = (
        build_engine()
    )

    verification = VerificationEngine()

    verification.register(
        "always-fail",
        lambda result, expected:
            False,
    )

    runner = ExperimentRunner(
        discovery,
        verification,
    )

    experiment = Experiment.create(
        objective="Test failure.",
        resource_id=resource.id,
    )

    experiment.actions.append(
        ExperimentAction(
            action={
                "name":
                    "test",
            }
        )
    )

    result = runner.run(
        experiment
    )

    assert (
        result.status
        == "failed"
    )


def test_verified_experiment_promotion(
    tmp_path,
):

    discovery, resource = (
        build_engine()
    )

    verification = VerificationEngine()

    verification.register(
        "success",
        lambda result, expected:
            result["success"] is True,
    )

    runner = ExperimentRunner(
        discovery,
        verification,
    )

    experiment = Experiment.create(
        objective=
            "Learn a reusable operation.",
        resource_id=
            resource.id,
    )

    experiment.actions.append(
        ExperimentAction(
            action={
                "name":
                    "test",
            }
        )
    )

    runner.run(
        experiment
    )

    from gene.discovery import (
        CapabilityStore,
    )

    store = CapabilityStore(
        str(
            tmp_path
            / "capabilities.jsonl"
        )
    )

    promoter = ExperimentPromoter(
        store
    )

    candidate = promoter.promote(
        experiment,
        "discovered-operation",
    )

    assert candidate.verified
    assert candidate.reusable
    assert len(
        store.reusable()
    ) == 1
