from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class GeneLevel(str, Enum):
    INSIGHT = "insight"
    SCHOLAR = "scholar"
    TRANSCENDENT = "transcendent"
    OMNISCIENT = "omniscient"


@dataclass(frozen=True)
class ResourceBudget:
    reasoning_units: int
    worker_count: int
    max_parallel_workers: int

    max_tool_calls: int
    max_retrieval_items: int

    context_budget: int
    verification_passes: int

    max_browser_actions: int
    max_background_tasks: int

    compute_multiplier: float


@dataclass(frozen=True)
class LevelDefinition:
    level: GeneLevel
    description: str
    budget: ResourceBudget


LEVELS: dict[GeneLevel, LevelDefinition] = {
    GeneLevel.INSIGHT: LevelDefinition(
        level=GeneLevel.INSIGHT,
        description=(
            "Efficient execution with conservative "
            "resource usage."
        ),
        budget=ResourceBudget(
            reasoning_units=100,
            worker_count=1,
            max_parallel_workers=1,
            max_tool_calls=8,
            max_retrieval_items=12,
            context_budget=32000,
            verification_passes=1,
            max_browser_actions=10,
            max_background_tasks=2,
            compute_multiplier=1.0,
        ),
    ),

    GeneLevel.SCHOLAR: LevelDefinition(
        level=GeneLevel.SCHOLAR,
        description=(
            "Deeper reasoning, retrieval and "
            "multi-step execution."
        ),
        budget=ResourceBudget(
            reasoning_units=250,
            worker_count=3,
            max_parallel_workers=2,
            max_tool_calls=24,
            max_retrieval_items=32,
            context_budget=100000,
            verification_passes=2,
            max_browser_actions=30,
            max_background_tasks=5,
            compute_multiplier=2.0,
        ),
    ),

    GeneLevel.TRANSCENDENT: LevelDefinition(
        level=GeneLevel.TRANSCENDENT,
        description=(
            "High-depth orchestration with parallel "
            "workers and aggressive verification."
        ),
        budget=ResourceBudget(
            reasoning_units=600,
            worker_count=8,
            max_parallel_workers=6,
            max_tool_calls=64,
            max_retrieval_items=96,
            context_budget=250000,
            verification_passes=3,
            max_browser_actions=100,
            max_background_tasks=12,
            compute_multiplier=4.0,
        ),
    ),

    GeneLevel.OMNISCIENT: LevelDefinition(
        level=GeneLevel.OMNISCIENT,
        description=(
            "Maximum configured Gene resource allocation."
        ),
        budget=ResourceBudget(
            reasoning_units=1200,
            worker_count=16,
            max_parallel_workers=12,
            max_tool_calls=128,
            max_retrieval_items=256,
            context_budget=500000,
            verification_passes=5,
            max_browser_actions=250,
            max_background_tasks=32,
            compute_multiplier=8.0,
        ),
    ),
}


class LevelManager:

    def __init__(
        self,
        level: GeneLevel = GeneLevel.INSIGHT,
    ) -> None:
        self._level = level

    @property
    def level(self) -> GeneLevel:
        return self._level

    def set_level(
        self,
        level: GeneLevel | str,
    ) -> LevelDefinition:

        if isinstance(level, str):
            level = GeneLevel(
                level.lower()
            )

        self._level = level
        return self.definition()

    def definition(self) -> LevelDefinition:
        return LEVELS[self._level]

    def budget(self) -> ResourceBudget:
        return self.definition().budget

    def can_use_tools(
        self,
        calls: int,
    ) -> bool:
        return calls <= self.budget().max_tool_calls

    def can_retrieve(
        self,
        items: int,
    ) -> bool:
        return items <= (
            self.budget().max_retrieval_items
        )

    def can_spawn_workers(
        self,
        workers: int,
    ) -> bool:
        return workers <= (
            self.budget().max_parallel_workers
        )

    def can_schedule(
        self,
        tasks: int,
    ) -> bool:
        return tasks <= (
            self.budget().max_background_tasks
        )

    def context_budget(self) -> int:
        return self.budget().context_budget

    def verification_passes(self) -> int:
        return self.budget().verification_passes

    def status(self) -> dict:

        definition = self.definition()

        return {
            "level": definition.level.value,
            "description":
                definition.description,
            "budget":
                asdict(definition.budget),
            "neural_weights_modified": False,
        }
