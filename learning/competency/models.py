from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CompetencyStage(str, Enum):

    INITIATE = "initiate"
    FOUNDATION = "foundation"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    MASTER = "master"
    PROFESSIONAL = "professional"


@dataclass
class CompetencyProfile:

    stage: CompetencyStage

    knowledge_depth: float = 0.0
    procedural_ability: float = 0.0
    independence: float = 0.0
    transfer_ability: float = 0.0
    verification_strength: float = 0.0
    professional_applicability: float = 0.0

    confidence: float = 0.0

    domain: str | None = None
    subdomain: str | None = None

    prerequisites: list[str] = field(
        default_factory=list
    )

    evidence_ids: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "stage":
                self.stage.value,
            "knowledge_depth":
                self.knowledge_depth,
            "procedural_ability":
                self.procedural_ability,
            "independence":
                self.independence,
            "transfer_ability":
                self.transfer_ability,
            "verification_strength":
                self.verification_strength,
            "professional_applicability":
                self.professional_applicability,
            "confidence":
                self.confidence,
            "domain":
                self.domain,
            "subdomain":
                self.subdomain,
            "prerequisites":
                self.prerequisites,
            "evidence_ids":
                self.evidence_ids,
            "metadata":
                self.metadata,
        }
