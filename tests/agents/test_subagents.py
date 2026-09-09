from gene.core.agents import (
    SubAgentRegistry,
    SubAgentRuntime,
    SubAgentSpec,
    SubAgentStatus,
)


def test_subagent_registry():
    registry = SubAgentRegistry()

    registry.register(
        SubAgentSpec(
            role="test_worker",
            description="Test worker",
            system_instructions="Complete the task.",
            allowed_tools=[],
        )
    )

    assert registry.exists("test_worker")
    assert registry.get("test_worker") is not None


def test_subagent_execution():
    registry = SubAgentRegistry()

    registry.register(
        SubAgentSpec(
            role="test_worker",
            description="Test worker",
            system_instructions="Complete the task.",
            allowed_tools=[],
        )
    )

    def runner(prompt, metadata):
        assert metadata["role"] == "test_worker"
        return "sub-agent completed the delegated task"

    runtime = SubAgentRuntime(
        registry=registry,
        model_runner=runner,
    )

    task = runtime.create_task(
        "Perform the delegated test.",
        role="test_worker",
    )

    result = runtime.run(task)

    assert result.status == SubAgentStatus.COMPLETED
    assert result.success
    assert "completed" in result.output


def test_subagent_tool_boundary():
    registry = SubAgentRegistry()

    registry.register(
        SubAgentSpec(
            role="restricted",
            description="Restricted worker",
            system_instructions="Complete the task.",
            allowed_tools=[],
        )
    )

    runtime = SubAgentRuntime(
        registry=registry,
        model_runner=lambda prompt, metadata: "done",
    )

    try:
        runtime.create_task(
            "Do something.",
            role="restricted",
            tools=["filesystem"],
        )
    except PermissionError:
        return

    raise AssertionError(
        "Sub-agent received a tool it was not authorized to use."
    )
