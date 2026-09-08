from __future__ import annotations

import re

from gene.learning.competency import (
    CompetencyEvaluator,
)


class PreliminaryCompetencyEstimator:

    COMPLEXITY_PATTERNS = (
        r"\bwhy\b",
        r"\bexplain\b",
        r"\bcompare\b",
        r"\banalyze\b",
        r"\bevaluate\b",
        r"\bdesign\b",
        r"\bdevelop\b",
        r"\bimplement\b",
        r"\boptimize\b",
        r"\bcritique\b",
        r"\bderive\b",
        r"\bsolve\b",
        r"\bcreate\b",
    )

    PROCEDURE_PATTERNS = (
        r"\bstep\b",
        r"\bprocedure\b",
        r"\bworkflow\b",
        r"\binstructions?\b",
        r"\bhow to\b",
        r"\bprocess\b",
    )

    PROFESSIONAL_PATTERNS = (
        r"\bproduction\b",
        r"\bclient\b",
        r"\bindustry\b",
        r"\bprofessional\b",
        r"\bdeployment\b",
        r"\brequirements\b",
        r"\bdeliverable\b",
        r"\bquality assurance\b",
    )

    def __init__(
        self,
        evaluator: CompetencyEvaluator | None = None,
    ) -> None:

        self.evaluator = (
            evaluator
            or CompetencyEvaluator()
        )

    @staticmethod
    def _ratio(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    def estimate(
        self,
        text: str,
        *,
        metadata: dict | None = None,
    ) -> dict:

        metadata = metadata or {}

        normalized = text.lower()

        word_count = len(
            normalized.split()
        )

        complexity_hits = sum(
            bool(
                re.search(
                    pattern,
                    normalized,
                )
            )
            for pattern
            in self.COMPLEXITY_PATTERNS
        )

        procedure_hits = sum(
            bool(
                re.search(
                    pattern,
                    normalized,
                )
            )
            for pattern
            in self.PROCEDURE_PATTERNS
        )

        professional_hits = sum(
            bool(
                re.search(
                    pattern,
                    normalized,
                )
            )
            for pattern
            in self.PROFESSIONAL_PATTERNS
        )

        # These are generic evidence signals.
        # They do not identify or whitelist any subject.
        knowledge_depth = self._ratio(
            0.25
            + min(
                0.40,
                word_count / 4000,
            )
            + min(
                0.35,
                complexity_hits * 0.08,
            )
        )

        procedural_ability = self._ratio(
            0.20
            + min(
                0.50,
                procedure_hits * 0.10,
            )
            + (
                0.10
                if word_count > 250
                else 0.0
            )
        )

        independence = self._ratio(
            0.25
            + min(
                0.45,
                complexity_hits * 0.09,
            )
            + (
                0.10
                if word_count > 500
                else 0.0
            )
        )

        transfer_ability = self._ratio(
            0.20
            + min(
                0.50,
                complexity_hits * 0.10,
            )
        )

        verification_strength = self._ratio(
            0.25
            + min(
                0.30,
                procedure_hits * 0.07,
            )
            + min(
                0.30,
                complexity_hits * 0.06,
            )
        )

        professional_applicability = self._ratio(
            0.10
            + min(
                0.50,
                professional_hits * 0.12,
            )
            + (
                0.15
                if procedure_hits >= 2
                else 0.0
            )
        )

        confidence = self._ratio(
            0.45
            + (
                0.10
                if word_count >= 100
                else 0.0
            )
            + (
                0.10
                if complexity_hits
                else 0.0
            )
            + (
                0.10
                if procedure_hits
                else 0.0
            )
        )

        profile = self.evaluator.evaluate(
            knowledge_depth=
                knowledge_depth,
            procedural_ability=
                procedural_ability,
            independence=
                independence,
            transfer_ability=
                transfer_ability,
            verification_strength=
                verification_strength,
            professional_applicability=
                professional_applicability,
            confidence=
                confidence,
            domain=
                metadata.get("domain"),
            subdomain=
                metadata.get("subdomain"),
        )

        result = profile.to_dict()

        result["source"] = (
            "preliminary_evidence_estimate"
        )

        result["signals"] = {
            "word_count":
                word_count,
            "complexity_hits":
                complexity_hits,
            "procedure_hits":
                procedure_hits,
            "professional_hits":
                professional_hits,
        }

        return result
