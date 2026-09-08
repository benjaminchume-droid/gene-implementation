from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass
class SkillCandidate:

    id: str

    name: str
    domain: str
    description: str
    instructions: str

    evidence_ids: list[str] = field(
        default_factory=list
    )

    capability_id: str | None = None

    confidence: float = 0.0
    score: float = 0.0

    validated: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneUnit:

    id: str

    name: str
    skill_ids: list[str]

    prerequisites: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Genome:

    id: str

    name: str
    gene_ids: list[str]

    version: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)
