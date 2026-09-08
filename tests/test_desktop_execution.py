from gene.desktop.agent import DesktopAgent
from gene.desktop.policy import DesktopPolicy


def test_desktop_agent_has_execution_interfaces():
    agent = DesktopAgent(
        DesktopPolicy()
    )

    assert agent.computer is not None
    assert agent.app_controller is not None
    assert agent.screen is not None


def test_mouse_position():
    agent = DesktopAgent(
        DesktopPolicy()
    )

    result = agent.mouse_position()

    assert result["success"] is True
    assert "x" in result
    assert "y" in result


def test_running_apps():
    agent = DesktopAgent(
        DesktopPolicy()
    )

    result = agent.running_apps()

    assert result["success"] is True
    assert isinstance(
        result["results"],
        list,
    )
