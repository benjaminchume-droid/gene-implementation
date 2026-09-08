from __future__ import annotations

from .analyzer import ExperienceAnalyzer
from .consolidator import ExperienceConsolidator
from .models import Experience
from .store import ExperienceStore


class LearningLoop:

    def __init__(
        self,
        store: ExperienceStore | None = None,
        analyzer: ExperienceAnalyzer | None = None,
        consolidator: ExperienceConsolidator | None = None,
    ) -> None:

        self.store = (
            store
            or ExperienceStore()
        )

        self.analyzer = (
            analyzer
            or ExperienceAnalyzer()
        )

        self.consolidator = (
            consolidator
            or ExperienceConsolidator()
        )

    def record(
        self,
        experience: Experience,
    ) -> dict:

        self.store.save(
            experience
        )

        candidate = (
            self.consolidator
            .consolidate(
                experience
            )
        )

        return {
            "experience":
                experience,
            "candidate":
                candidate,
        }

    def analyze(self) -> dict:

        return self.analyzer.analyze(
            self.store.list()
        )

    def status(self) -> dict:

        return {
            "storage":
                self.store.status(),
            "analysis":
                self.analyze(),
        }
