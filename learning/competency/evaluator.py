from __future__ import annotations

from .models import (
    CompetencyProfile,
    CompetencyStage,
)


class CompetencyEvaluator:

    def evaluate(
        self,
        *,
        knowledge_depth: float,
        procedural_ability: float,
        independence: float,
        transfer_ability: float,
        verification_strength: float,
        professional_applicability: float,
        confidence: float,
        domain: str | None = None,
        subdomain: str | None = None,
        prerequisites: list[str] | None = None,
        evidence_ids: list[str] | None = None,
    ) -> CompetencyProfile:

        values = [
            knowledge_depth,
            procedural_ability,
            independence,
            transfer_ability,
            verification_strength,
            professional_applicability,
            confidence,
        ]

        if any(
            value < 0.0 or value > 1.0
            for value in values
        ):
            raise ValueError(
                "Competency scores must be between 0 and 1."
            )

        # General competency score.
        score = (
            knowledge_depth * 0.20
            + procedural_ability * 0.20
            + independence * 0.15
            + transfer_ability * 0.15
            + verification_strength * 0.15
            + professional_applicability * 0.15
        )

        # The stage is derived from evidence dimensions,
        # not from a hard-coded domain or profession.
        if (
            professional_applicability >= 0.85
            and verification_strength >= 0.85
            and independence >= 0.85
        ):
            stage = CompetencyStage.PROFESSIONAL

        elif (
            knowledge_depth >= 0.85
            and transfer_ability >= 0.80
            and verification_strength >= 0.80
        ):
            stage = CompetencyStage.MASTER

        elif score >= 0.72:
            stage = CompetencyStage.ADVANCED

        elif score >= 0.52:
            stage = CompetencyStage.INTERMEDIATE

        elif score >= 0.30:
            stage = CompetencyStage.FOUNDATION

        else:
            stage = CompetencyStage.INITIATE

        return CompetencyProfile(
            stage=stage,
            knowledge_depth=knowledge_depth,
            procedural_ability=procedural_ability,
            independence=independence,
            transfer_ability=transfer_ability,
            verification_strength=verification_strength,
            professional_applicability=
                professional_applicability,
            confidence=confidence,
            domain=domain,
            subdomain=subdomain,
            prerequisites=
                prerequisites or [],
            evidence_ids=
                evidence_ids or [],
        )
