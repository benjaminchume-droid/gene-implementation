from __future__ import annotations

from gene.discovery.capabilities import (
    CapabilityStore,
)
from gene.evolution.promotion import (
    PromotionEngine,
)


class CapabilityPromotionBridge:

    def __init__(
        self,
        capabilities:
            CapabilityStore | None = None,
        promotion:
            PromotionEngine | None = None,
    ) -> None:

        self.capabilities = (
            capabilities
            or CapabilityStore()
        )

        self.promotion = (
            promotion
            or PromotionEngine()
        )

    def promote(
        self,
        capability_id: str,
        *,
        name: str,
        domain: str,
        description: str,
        instructions: str,
        score: float,
        tests_passed: int,
        tests_failed: int,
    ):

        candidates = [
            item
            for item
            in self.capabilities.list()
            if item.get("id")
            == capability_id
        ]

        if not candidates:
            raise KeyError(
                f"Unknown capability: "
                f"{capability_id}"
            )

        record = candidates[0]

        if not record.get(
            "verified"
        ):
            raise ValueError(
                "Capability must be verified first."
            )

        # Reconstruct a minimal capability object
        # without introducing a fixed taxonomy.
        class Capability:
            pass

        capability = Capability()
        capability.id = record["id"]
        capability.resource_id = (
            record["resource_id"]
        )
        capability.resource_kind = (
            record["resource_kind"]
        )
        capability.evidence_ids = (
            record.get(
                "evidence_ids",
                [],
            )
        )
        capability.confidence = (
            record.get(
                "confidence",
                0.0,
            )
        )
        capability.verified = (
            record.get(
                "verified",
                False,
            )
        )
        capability.metadata = (
            record.get(
                "metadata",
                {},
            )
        )

        candidate = (
            self.promotion
            .candidate_from_capability(
                capability,
                name=name,
                domain=domain,
                description=description,
                instructions=instructions,
            )
        )

        candidate = (
            self.promotion.validate_candidate(
                candidate,
                score=score,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
            )
        )

        return self.promotion.promote_skill(
            candidate
        )
