from gene.agent.autonomy import (
    AgentAutonomyRuntime,
    Mission,
    MissionState,
)

from gene.capabilities.runtime import (
    CapabilityRuntime,
    RuntimeCapability,
)

from gene.discovery import (
    DiscoveryEngine,
    DiscoveryRegistry,
    Resource,
    SyntheticEnvironment,
)

from gene.learning.loop import (
    LearningLoop,
)

from gene.planning import (
    CapabilityComposer,
    PlanExecutor,
)


def test_complete_agent_mission(
    tmp_path,
):

    resource = Resource(
        id="resource-1",
        kind="discovered",
        name="Generic resource",
        actions=[
            {
                "name":
                    "perform",
            }
        ],
    )

    registry = DiscoveryRegistry()

    registry.register(
        SyntheticEnvironment(
            name="generic-environment",
            resources=[
                resource
            ],
        )
    )

    discovery = DiscoveryEngine(
        registry=registry
    )

    runtime = CapabilityRuntime(
        discovery_engine=discovery,
        path=str(
            tmp_path
            / "runtime.json"
        ),
    )

    capability = RuntimeCapability(
        id="capability-1",
        name="perform operation",
        description=(
            "perform a generic operation"
        ),
        kind="discovered",
        procedure=[
            {
                "resource_id":
                    "resource-1",
                "action": {
                    "name":
                        "perform",
                },
            }
        ],
        confidence=1.0,
    )

    runtime.activate(
        capability
    )

    composer = CapabilityComposer(
        runtime
    )

    executor = PlanExecutor(
        runtime
    )

    learning = LearningLoop(
        store=__import__(
            "gene.learning.loop",
            fromlist=[
                "ExperienceStore"
            ],
        ).ExperienceStore(
            str(
                tmp_path
                / "experiences.jsonl"
            )
        )
    )

    autonomy = AgentAutonomyRuntime(
        discovery=discovery,
        composer=composer,
        executor=executor,
        learning=learning,
        store=__import__(
            "gene.agent.autonomy",
            fromlist=[
                "MissionStore"
            ],
        ).MissionStore(
            str(
                tmp_path
                / "missions.jsonl"
            )
        ),
    )

    autonomy.configure_planner(
        lambda objective, context: [
            "perform operation"
        ]
    )

    mission = Mission.create(
        "Complete an arbitrary operation."
    )

    result = autonomy.run(
        mission
    )

    assert (
        result.state
        == MissionState.COMPLETED
    )

    assert result.plan is not None

    assert result.result is not None

    assert result.learning

    events = (
        autonomy.events.for_mission(
            result.id
        )
    )

    names = [
        event.event
        for event in events
    ]

    assert (
        "discovery_started"
        in names
    )

    assert (
        "planning_completed"
        in names
    )

    assert (
        "execution_completed"
        in names
    )

    assert (
        "mission_completed"
        in names
    )
