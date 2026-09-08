from __future__ import annotations

from gene.learning.candidates.store import (
    CandidateStore,
)

from .models import Experience


class ExperienceConsolidator:

    def __init__(
        self,
        candidates: CandidateStore | None = None,
    ) -> None:

        self.candidates = (
            candidates
            or CandidateStore()
        )

    def consolidate(
        self,
        experience: Experience,
    ):

        if not experience.success:
            return None

        content = (
            f"Objective: "
            f"{experience.objective}\n\n"
            f"Outcome: "
            f"{experience.outcome}"
        )

        return self.candidates.add(
            objective=
                experience.objective,
            candidate_type=
                "experience",
            content=
                content,
            evidence_ids=
                experience.evidence_ids,
            score=1.0,
            confidence=1.0,
            accepted=True,
            metadata={
                "experience_id":
                    experience.id,
                "plan_id":
                    experience.plan_id,
                "capability_ids":
                    experience.capability_ids,
                "source":
                    "experience",
            },
        )
