from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfirmationDecision:
    approved: bool
    reason: str


class ConfirmationBroker:

    def __init__(self, policy):
        self.policy = policy

    def evaluate(
        self,
        action: str,
        *,
        destructive: bool = False,
        elevated: bool = False,
        external: bool = False,
    ) -> ConfirmationDecision:

        mode = self.policy.confirmation_mode.value

        if mode == "never":
            return ConfirmationDecision(
                approved=True,
                reason="Policy allows automatic execution.",
            )

        if mode == "always":
            return ConfirmationDecision(
                approved=False,
                reason=(
                    f"Confirmation required for action: {action}"
                ),
            )

        if destructive or elevated or external:
            return ConfirmationDecision(
                approved=False,
                reason=(
                    f"Risk-based confirmation required: {action}"
                ),
            )

        return ConfirmationDecision(
            approved=True,
            reason="Action is within automatic policy.",
        )