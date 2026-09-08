from gene.agent.confirmation import ConfirmationBroker
from gene.agent.action import ActionRequest
from gene.desktop.policy import (
    ConfirmationMode,
    DesktopPolicy,
)


def test_action_request():
    request = ActionRequest(
        action="computer.click"
    )

    assert request.status == "requested"

    request.approve()

    assert request.status == "approved"

    request.complete()

    assert request.status == "completed"


def test_risk_confirmation():
    policy = DesktopPolicy(
        confirmation_mode=ConfirmationMode.RISK_BASED
    )

    broker = ConfirmationBroker(policy)

    safe = broker.evaluate(
        "screen.capture"
    )

    risky = broker.evaluate(
        "browser.click",
        external=True,
    )

    assert safe.approved is True
    assert risky.approved is False


def test_always_confirmation():
    policy = DesktopPolicy(
        confirmation_mode=ConfirmationMode.ALWAYS
    )

    broker = ConfirmationBroker(policy)

    decision = broker.evaluate(
        "computer.click"
    )

    assert decision.approved is False