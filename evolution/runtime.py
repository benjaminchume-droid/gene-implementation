from __future__ import annotations

from gene.evolution.external import (
    EvolutionController,
    ExternalParameterStore,
)
from gene.evolution.evaluator import (
    EvaluationCase,
    ParameterEvaluator,
)


class EvolutionRuntime:

    def __init__(self):

        self.store = (
            ExternalParameterStore()
        )

        self.controller = (
            EvolutionController(
                self.store
            )
        )

    def evaluate_and_activate(
        self,
        namespace: str,
        changes: dict,
        reason: str,
        cases: list[EvaluationCase],
        runner,
        evidence: list[str] | None = None,
        minimum_score: float = 0.80,
    ) -> dict:

        proposal = self.controller.propose(
            namespace=namespace,
            changes=changes,
            reason=reason,
            evidence=evidence,
        )

        candidate = (
            self.controller
            .candidate_parameters(
                proposal
            )
        )

        evaluation = ParameterEvaluator(
            runner
        ).evaluate(
            candidate,
            cases,
            minimum_score=minimum_score,
        )

        if not evaluation.passed:

            proposal.status = "rejected"

            return {
                "success": False,
                "status": "rejected",
                "proposal_id":
                    proposal.id,
                "score":
                    evaluation.score,
                "case_scores":
                    evaluation.case_scores,
                "failures":
                    evaluation.failures,
                "neural_weights_modified":
                    False,
            }

        return self.controller.activate(
            proposal,
            evaluation.score,
            minimum_score,
        )

    def rollback(
        self,
        namespace: str,
        version: int,
    ) -> dict:

        return self.controller.rollback(
            namespace,
            version,
        )

    def status(self) -> dict:

        return self.store.status()
