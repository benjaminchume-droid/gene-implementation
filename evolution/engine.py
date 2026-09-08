from __future__ import annotations

from .proposal import EvolutionProposal
from .validator import EvolutionValidator


class EvolutionEngine:

    def __init__(
        self,
        skill_manager,
    ) -> None:

        self.skill_manager = skill_manager
        self.validator = (
            EvolutionValidator()
        )

    def propose_skill(
        self,
        name: str,
        domain: str,
        reason: str,
        proposed_change: str,
        instructions: str,
        *,
        parent: str | None = None,
        evidence: list[str] | None = None,
    ) -> EvolutionProposal:

        return EvolutionProposal(
            target=name,
            operation="create_skill",
            reason=reason,
            proposed_change=proposed_change,
            evidence=evidence or [],
        )

    def create_validated_skill(
        self,
        name: str,
        domain: str,
        description: str,
        instructions: str,
        *,
        score: float,
        tests_passed: int,
        tests_failed: int,
        parent: str | None = None,
        tags: list[str] | None = None,
    ):

        skill = self.skill_manager.create(
            name=name,
            domain=domain,
            description=description,
            instructions=instructions,
            parent=parent,
            tags=tags,
        )

        return self.skill_manager.validate(
            skill.name,
            score,
            tests_passed,
            tests_failed,
        )

    def branch_validated_skill(
        self,
        parent: str,
        name: str,
        description: str,
        instructions: str,
        *,
        score: float,
        tests_passed: int,
        tests_failed: int,
    ):

        skill = self.skill_manager.branch(
            parent=parent,
            name=name,
            description=description,
            instructions=instructions,
        )

        return self.skill_manager.validate(
            skill.name,
            score,
            tests_passed,
            tests_failed,
        )
