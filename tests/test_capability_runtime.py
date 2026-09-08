from gene.capabilities.runtime import (
    CapabilityRuntime,
    RuntimeCapability,
)


def test_runtime_capability_registration(
    tmp_path,
):

    runtime = CapabilityRuntime(
        path=str(
            tmp_path
            / "runtime.json"
        )
    )

    capability = RuntimeCapability(
        id="dynamic-capability",
        name="Discovered operation",
        description=(
            "A capability learned dynamically."
        ),
        kind="discovered",
        procedure=[],
        confidence=0.95,
    )

    runtime.activate(
        capability
    )

    found = runtime.registry.get(
        "dynamic-capability"
    )

    assert found.name == (
        "Discovered operation"
    )


def test_runtime_persistence(
    tmp_path,
):

    path = (
        tmp_path
        / "runtime.json"
    )

    first = CapabilityRuntime(
        path=str(path)
    )

    first.activate(
        RuntimeCapability(
            id="persistent",
            name="Persistent capability",
            description="test",
            kind="discovered",
        )
    )

    second = CapabilityRuntime(
        path=str(path)
    )

    assert (
        second.registry.get(
            "persistent"
        ).name
        == "Persistent capability"
    )


def test_runtime_deactivation(
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
            id="switchable",
            name="Switchable capability",
            description="test",
            kind="discovered",
        )
    )

    runtime.deactivate(
        "switchable"
    )

    assert (
        runtime.registry.get(
            "switchable"
        ).enabled
        is False
    )
