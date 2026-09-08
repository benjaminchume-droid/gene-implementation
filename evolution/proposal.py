from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class EvolutionProposal:
    target: str
    operation: str
    reason: str
    proposed_change: str
    evidence: list[str] = field(
        default_factory=list
    )
    status: str = "proposed"
    created_at: str = field(
        default_factory=lambda:
        datetime.now(timezone.utc).isoformat()
    )

    def approve(self) -> None:
        self.status = "approved"

    def reject(self) -> None:
        self.status = "rejected"
