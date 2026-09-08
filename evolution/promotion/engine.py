from __future__ import annotations

from .models import (
    GeneUnit,
    Genome,
    SkillCandidate,
)
from .store import PromotionStore


class PromotionEngine:

    def __init__(
        self,
        store: PromotionStore | None = None,
        skill_manager=None,
    ) -> None:

        self.store = (
            store
            or PromotionStore()
        )

        self.skill_manager = (
            skill_manager
        )

    def candidate_from_capability(
        self,
        capability,
        *,
        name: str,
        domain: str,
        description: str,
        instructions: str,
        evidence_ids: list[str] | None = None,
    ) -> SkillCandidate:

        return SkillCandidate(
            id=capability.id,
            name=name,
            domain=domain,
            description=description,
            instructions=instructions,
            evidence_ids=(
                evidence_ids
                or capability.evidence_ids
            ),
            capability_id=capability.id,
            confidence=capability.confidence,
            score=capability.confidence,
            validated=capability.verified,
            metadata={
                "resource_id":
                    capability.resource_id,
                "resource_kind":
                    capability.resource_kind,
                "source_metadata":
                    capability.metadata,
            },
        )

    def validate_candidate(
        self,
        candidate: SkillCandidate,
        *,
        score: float,
        tests_passed: int,
        tests_failed: int,
    ) -> SkillCandidate:

        total = (
            tests_passed
            + tests_failed
        )

        test_ratio = (
            tests_passed / total
            if total
            else 0.0
        )

        candidate.score = (
            max(
                0.0,
                min(1.0, score)
            )
            * 0.7
            + test_ratio
            * 0.3
        )

        candidate.validated = (
            tests_failed == 0
            and candidate.score >= 0.70
        )

        return candidate

    def promote_skill(
        self,
        candidate: SkillCandidate,
    ):

        if not candidate.validated:
            raise ValueError(
                "Only validated candidates can "
                "be promoted to skills."
            )

        if self.skill_manager is not None:

            existing = {
                item.name
                for item
                in self.skill_manager.registry.list()
            }

            if candidate.name in existing:
                skill = next(
                    item
                    for item
                    in self.skill_manager.registry.list()
                    if item.name
                    == candidate.name
                )
            else:

                skill = (
                    self.skill_manager.create(
                        name=candidate.name,
                        domain=candidate.domain,
                        description=
                            candidate.description,
                        instructions=
                            candidate.instructions,
                    )
                )

            self.store.save_skill(
                candidate
            )

            return skill

        self.store.save_skill(
            candidate
        )

        return candidate

    def create_gene(
        self,
        name: str,
        skill_ids: list[str],
        *,
        prerequisites=None,
        metadata=None,
    ) -> GeneUnit:

        gene = GeneUnit(
            id=self._id(
                "gene"
            ),
            name=name,
            skill_ids=list(skill_ids),
            prerequisites=(
                prerequisites
                or []
            ),
            metadata=(
                metadata
                or {}
            ),
        )

        self.store.save_gene(
            gene
        )

        return gene

    def compose_genome(
        self,
        name: str,
        gene_ids: list[str],
        *,
        metadata=None,
    ) -> Genome:

        if not gene_ids:
            raise ValueError(
                "A genome requires at least "
                "one gene."
            )

        genome = Genome(
            id=self._id(
                "genome"
            ),
            name=name,
            gene_ids=list(gene_ids),
            metadata=(
                metadata
                or {}
            ),
        )

        self.store.save_genome(
            genome
        )

        return genome

    @staticmethod
    def _id(
        prefix: str,
    ) -> str:

        import uuid

        return (
            f"{prefix}-"
            f"{uuid.uuid4()}"
        )
