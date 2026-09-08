from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass


@dataclass
class EvaluationResult:

    accepted: bool

    score: float

    confidence: float

    reasons: list[str]


class LearningEvaluator:

    def evaluate(
        self,
        response: str,
        *,
        verification: dict | None = None,
        corroboration: list[str] | None = None,
    ) -> EvaluationResult:

        reasons = []

        text = response.strip()

        if len(text) < 20:
            return EvaluationResult(
                accepted=False,
                score=0.0,
                confidence=0.0,
                reasons=[
                    "insufficient_content"
                ],
            )

        verification = (
            verification or {}
        )

        corroboration = (
            corroboration or []
        )

        score = 0.50

        # Evidence from an actual executable/tested result.
        if verification.get(
            "passed"
        ) is True:
            score += 0.25
            reasons.append(
                "externally_verified"
            )

        # Independent agreement increases confidence.
        if corroboration:

            score += min(
                0.20,
                0.05 * len(
                    corroboration
                ),
            )

            reasons.append(
                "corroborated"
            )

        # Detect obvious uncertainty language.
        uncertainty_markers = (
            "i don't know",
            "i am not sure",
            "probably",
            "might be",
            "possibly",
        )

        lowered = text.lower()

        if any(
            marker in lowered
            for marker
            in uncertainty_markers
        ):
            score -= 0.10
            reasons.append(
                "uncertainty_detected"
            )

        # Repetition penalty.
        words = lowered.split()

        if words:

            unique_ratio = (
                len(set(words))
                / len(words)
            )

            if unique_ratio < 0.35:

                score -= 0.15

                reasons.append(
                    "high_repetition"
                )

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        confidence = score

        return EvaluationResult(
            accepted=(
                score >= 0.70
            ),
            score=score,
            confidence=confidence,
            reasons=reasons,
        )

    @staticmethod
    def fingerprint(
        text: str,
    ) -> str:

        normalized = re.sub(
            r"\s+",
            " ",
            text.lower().strip(),
        )

        return hashlib.sha256(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()
