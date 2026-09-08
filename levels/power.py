from __future__ import annotations

from dataclasses import asdict

from gene.levels import (
    GeneLevel,
    LevelManager,
)


class PowerController:

    def __init__(
        self,
        default_level: GeneLevel = GeneLevel.INSIGHT,
    ) -> None:

        self.levels = LevelManager(
            default_level
        )

    @property
    def level(self) -> GeneLevel:
        return self.levels.level

    def set_level(
        self,
        level: GeneLevel | str,
    ) -> dict:

        definition = self.levels.set_level(
            level
        )

        return {
            "success": True,
            "level":
                definition.level.value,
            "budget":
                asdict(definition.budget),
            "neural_weights_modified":
                False,
        }

    def allow(
        self,
        resource: str,
        amount: int,
    ) -> bool:

        checks = {
            "tools":
                self.levels.can_use_tools,
            "retrieval":
                self.levels.can_retrieve,
            "workers":
                self.levels.can_spawn_workers,
            "background":
                self.levels.can_schedule,
        }

        checker = checks.get(resource)

        if checker is None:
            raise ValueError(
                f"Unknown resource: {resource}"
            )

        return checker(amount)

    def status(self) -> dict:
        return self.levels.status()
