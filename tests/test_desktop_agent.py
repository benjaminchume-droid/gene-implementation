from gene.agents.worker import (
    TaskWorker,
    WorkerBudget,
    WorkerManager,
)

from gene.desktop.agent import DesktopAgent
from gene.desktop.policy import (
    DesktopPolicy,
    InternetLevel,
    PrivilegeLevel,
)


def test_privilege_policy():
    policy = DesktopPolicy(
        privilege_level=PrivilegeLevel.USER
    )

    assert policy.can_use_privilege(
        PrivilegeLevel.USER
    )

    assert not policy.can_use_privilege(
        PrivilegeLevel.ADMINISTRATOR
    )


def test_internet_policy():
    policy = DesktopPolicy(
        internet_level=InternetLevel.STANDARD
    )

    assert policy.can_use_internet(
        InternetLevel.STANDARD
    )

    assert not policy.can_use_internet(
        InternetLevel.UNRESTRICTED
    )


def test_worker_manager():
    manager = WorkerManager()

    manager.register(
        TaskWorker(
            name="test",
            purpose="Testing",
            budget=WorkerBudget(),
        )
    )

    assert manager.get("test") is not None
    assert len(manager.list()) == 1


def test_worker_budget_validation():
    manager = WorkerManager()

    try:
        manager.register(
            TaskWorker(
                name="invalid",
                purpose="Invalid",
                budget=WorkerBudget(
                    reasoning_fraction=2.0
                ),
            )
        )
    except ValueError:
        return

    raise AssertionError(
        "Invalid worker budget was accepted."
    )


def test_desktop_agent_creation():
    policy = DesktopPolicy(
        allow_screen_capture=False
    )

    agent = DesktopAgent(policy)

    assert agent.screen is None
