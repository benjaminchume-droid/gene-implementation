from __future__ import annotations

from gene.skills.schema import Skill


class EvolutionValidator:

    MIN_VALIDATION_SCORE = 0.80

    def validate_skill(
        self,
        skill: Skill,
    ) -> dict:

        if skill.latest is None:
            return {
                "valid": False,
                "reason": "Skill has no version.",
            }

        version = skill.latest

        valid = (
            version.score
            >= self.MIN_VALIDATION_SCORE
            and version.tests_failed == 0
            and version.tests_passed > 0
        )

        return {
            "valid": valid,
            "score": version.score,
            "tests_passed":
                version.tests_passed,
            "tests_failed":
                version.tests_failed,
        }
