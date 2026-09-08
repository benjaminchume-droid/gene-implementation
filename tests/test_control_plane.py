from gene.agent.control_plane import GeneControlPlane
from gene.desktop.policy import DesktopPolicy
from gene.tools.runtime import ToolRuntime


def test_control_plane_boots():

    tools = ToolRuntime()

    control = GeneControlPlane(
        policy=DesktopPolicy(),
        tools=tools,
    )

    status = control.status()

    assert "external_parameters" in status
    assert "scheduler" in status
    assert "mcp" in status
    assert "vision" in status


def test_external_parameter_evolution():

    tools = ToolRuntime()

    control = GeneControlPlane(
        policy=DesktopPolicy(),
        tools=tools,
    )

    proposal = control.evolution.propose(
        namespace="routing",
        changes={
            "screen": 0.9,
        },
        reason="Improved screen-task routing.",
    )

    result = (
        control.evolution.evaluate_and_activate(
            proposal,
            score=0.95,
        )
    )

    assert result["success"] is True
    assert (
        result["neural_weights_modified"]
        is False
    )

    assert (
        control.parameters.get(
            "routing"
        )["screen"]
        == 0.9
    )