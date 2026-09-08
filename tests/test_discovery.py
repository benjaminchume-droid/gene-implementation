from gene.discovery import (
    CapabilityStore,
    DiscoveryEngine,
    DiscoveryMission,
    DiscoveryRegistry,
    Resource,
    SyntheticEnvironment,
)


def test_dynamic_discovery(
    tmp_path,
):

    resource = Resource(
        id="unknown-resource",
        kind="application",
        name="Discovered Application",
        description="An unknown reachable resource.",
        actions=[
            {
                "name":
                    "example_action",
                "parameters":
                    {
                        "value":
                            "any",
                    },
            }
        ],
    )

    registry = DiscoveryRegistry()

    registry.register(
        SyntheticEnvironment(
            name="runtime-environment",
            resources=[
                resource
            ],
        )
    )

    capabilities = CapabilityStore(
        str(
            tmp_path
            / "capabilities.jsonl"
        )
    )

    engine = DiscoveryEngine(
        registry=registry,
        capabilities=capabilities,
    )

    result = engine.discover(
        DiscoveryMission(
            objective=
                "Accomplish an arbitrary task.",
        )
    )

    assert result["resources"]
    assert result["observations"]


def test_capability_acquisition(
    tmp_path,
):

    resource = Resource(
        id="resource",
        kind="unknown",
        name="Unknown",
    )

    registry = DiscoveryRegistry()

    registry.register(
        SyntheticEnvironment(
            name="test",
            resources=[
                resource
            ],
        )
    )

    capabilities = CapabilityStore(
        str(
            tmp_path
            / "capabilities.jsonl"
        )
    )

    engine = DiscoveryEngine(
        registry=registry,
        capabilities=capabilities,
    )

    candidate = (
        engine.acquire_capability(
            mission=DiscoveryMission(
                objective=
                    "Learn something unknown."
            ),
            resource=resource,
            capability=
                "perform discovered operation",
            procedure=[
                {
                    "action":
                        "example"
                }
            ],
            confidence=0.9,
            verified=True,
            reusable=True,
        )
    )

    assert candidate.verified
    assert candidate.reusable

    assert len(
        capabilities.reusable()
    ) == 1
