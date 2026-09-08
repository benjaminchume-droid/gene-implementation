from __future__ import annotations

import json
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LearningCandidate:

    id: str

    objective: str

    candidate_type: str

    content: str

    evidence_ids: list[str] = field(
        default_factory=list
    )

    score: float = 0.0
    confidence: float = 0.0

    accepted: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )


class CandidateStore:

    def __init__(
        self,
        path: str = (
            "gene/data/learning/"
            "candidates/candidates.jsonl"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def add(
        self,
        objective: str,
        candidate_type: str,
        content: str,
        *,
        evidence_ids=None,
        score: float = 0.0,
        confidence: float = 0.0,
        accepted: bool = False,
        metadata=None,
    ) -> LearningCandidate:

        candidate = LearningCandidate(
            id=str(
                uuid.uuid4()
            ),
            objective=objective,
            candidate_type=
                candidate_type,
            content=content,
            evidence_ids=
                evidence_ids or [],
            score=score,
            confidence=confidence,
            accepted=accepted,
            metadata=
                metadata or {},
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    asdict(candidate),
                    ensure_ascii=False,
                )
                + "\n"
            )

        return candidate
