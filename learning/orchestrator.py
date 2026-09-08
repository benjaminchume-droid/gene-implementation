from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .candidates.store import (
    CandidateStore,
)
from .evaluation.evaluator import (
    LearningEvaluator,
)
from .evaluation.evidence import (
    EvidenceStore,
)
from .sources.registry import (
    LearningSourceRegistry,
)


@dataclass
class LearningMission:

    objective: str

    context: dict[str, Any] = field(
        default_factory=dict
    )

    required_capabilities: list[str] = field(
        default_factory=list
    )

    max_sources: int | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class LearningOrchestrator:

    def __init__(
        self,
        sources:
            LearningSourceRegistry | None = None,
        evidence:
            EvidenceStore | None = None,
        evaluator:
            LearningEvaluator | None = None,
        candidates:
            CandidateStore | None = None,
    ) -> None:

        self.sources = (
            sources
            or LearningSourceRegistry()
        )

        self.evidence = (
            evidence
            or EvidenceStore()
        )

        self.evaluator = (
            evaluator
            or LearningEvaluator()
        )

        self.candidates = (
            candidates
            or CandidateStore()
        )

    def discover_sources(
        self,
        mission: LearningMission,
    ):

        discovered = []

        for capability in (
            mission.required_capabilities
            or [None]
        ):

            found = self.sources.discover(
                capability=capability
            )

            for source in found:

                if source not in discovered:
                    discovered.append(
                        source
                    )

        if mission.max_sources is not None:
            discovered = discovered[
                :mission.max_sources
            ]

        return discovered

    def learn(
        self,
        mission: LearningMission,
    ) -> dict:

        sources = self.discover_sources(
            mission
        )

        exchanges = []
        evidence_records = []
        candidates = []

        for source in sources:

            exchange = source.learn(
                mission.objective,
                mission.context,
            )

            exchanges.append(
                exchange
            )

            evidence = self.evidence.add(
                source_id=
                    exchange.source_id,
                request=
                    exchange.request,
                response=
                    exchange.response,
                observations=
                    exchange.observations,
                actions=
                    exchange.actions,
                provenance={
                    "mission":
                        mission.objective,
                    **mission.metadata,
                },
            )

            evidence_records.append(
                evidence
            )

        # Responses can corroborate one another.
        responses = [
            exchange.response
            for exchange in exchanges
            if exchange.response
        ]

        for exchange, evidence in zip(
            exchanges,
            evidence_records,
        ):

            result = (
                self.evaluator.evaluate(
                    exchange.response,
                    corroboration=[
                        response
                        for response
                        in responses
                        if response
                        != exchange.response
                    ],
                )
            )

            candidate = self.candidates.add(
                objective=
                    mission.objective,
                candidate_type=
                    "learning",
                content=
                    exchange.response,
                evidence_ids=[
                    evidence.id
                ],
                score=
                    result.score,
                confidence=
                    result.confidence,
                accepted=
                    result.accepted,
                metadata={
                    "reasons":
                        result.reasons,
                    "source":
                        exchange.source_id,
                },
            )

            candidates.append(
                candidate
            )

        accepted = [
            candidate
            for candidate
            in candidates
            if candidate.accepted
        ]

        return {
            "success": True,
            "objective":
                mission.objective,
            "sources_used":
                len(sources),
            "evidence":
                len(evidence_records),
            "candidates":
                len(candidates),
            "accepted":
                len(accepted),
        }

    def status(self) -> dict:

        return {
            "sources":
                self.sources.status(),
            "candidate_storage":
                str(
                    self.candidates.path
                ),
            "evidence_storage":
                str(
                    self.evidence.path
                ),
        }
