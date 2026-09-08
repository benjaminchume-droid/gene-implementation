from gene.capabilities.runtime import (
    CapabilityRuntime,
    RuntimeCapability,
)

from gene.planning import (
    CapabilityComposer,
    PlanExecutor,
)


def build_runtime(
    tmp_path,
):

    runtime = CapabilityRuntime(
        path=str(
            tmp_path
            / "runtime.json"
        )
    )

    runtime.activate(
        RuntimeCapability(
            id="first",
            name="First operation",
            description=(
                "perform first operation"
            ),
            kind="discovered",
            procedure=[],
        )
    )

    runtime.activate(
        RuntimeCapability(
            id="second",
            name="Second operation",
            description=(
                "perform second operation"
            ),
            kind="discovered",
            procedure=[],
        )
    )

    return runtime


def test_composition(
    tmp_path,
):

    runtime = build_runtime(
        tmp_path
    )

    composer = CapabilityComposer(
        runtime
    )

    result = composer.compose(
        objective=(
            "Complete a multi-step objective."
        ),
        requested_steps=[
            "first operation",
            "second operation",
        ],
    )

    assert result.success
    assert len(
        result.plan.steps
    ) == 2

    assert (
        result.plan.steps[1]
        .dependencies
    )


def test_unresolved_capability(
    tmp_path,
):

    runtime = build_runtime(
        tmp_path
    )

    composer = CapabilityComposer(
        runtime
    )

    result = composer.compose(
        objective="Unknown objective.",
        requested_steps=[
            "something nonexistent",
        ],
    )

    assert not result.success
    assert result.unresolved_steps


def test_plan_execution(
    tmp_path,
):

    runtime = build_runtime(
        tmp_path
    )

    composer = CapabilityComposer(
        runtime
    )

    plan = composer.compose(
        objective="Execute known capabilities.",
        requested_steps=[
            "first operation",
            "second operation",
        ],
    ).plan

    executor = PlanExecutor(
        runtime
    )

    result = executor.execute(
        plan
    )

    assert result.status == (
        "completed"
    )
