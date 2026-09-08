from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SkillVersion:
    version: str
    instructions: str
    created_at: str = field(
        default_factory=lambda:
        datetime.now(timezone.utc).isoformat()
    )
    validated: bool = False
    score: float = 0.0
    tests_passed: int = 0
    tests_failed: int = 0

    def validate(
        self,
        score: float,
        tests_passed: int,
        tests_failed: int,
    ) -> None:
        self.score = max(0.0, min(1.0, score))
        self.tests_passed = tests_passed
        self.tests_failed = tests_failed
        self.validated = (
            self.score >= 0.80
            and self.tests_failed == 0
        )


@dataclass
class Skill:
    name: str
    domain: str
    description: str

    parent: str | None = None

    enabled: bool = True

    level: int = 1

    tags: list[str] = field(
        default_factory=list
    )

    dependencies: list[str] = field(
        default_factory=list
    )

    versions: list[SkillVersion] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def add_version(
        self,
        version: SkillVersion,
    ) -> None:

        self.versions.append(version)

    @property
    def latest(self) -> SkillVersion | None:
        if not self.versions:
            return None

        return self.versions[-1]

    @property
    def validated(self) -> bool:
        version = self.latest

        return bool(
            version and version.validated
        )
