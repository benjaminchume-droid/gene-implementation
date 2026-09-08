from __future__ import annotations

from gene.soul import SoulManager
from gene.memory import MemoryManager
from gene.knowledge import KnowledgeService
from gene.skills import SkillManager
from gene.genome import GenomeLoader, GenomeValidator
from gene.context import ContextManager
from gene.tools.runtime import ToolRuntime

from .state import GeneState


class GeneRuntime:

    def __init__(
        self,
        genome_manifest: str = "gene/manifests/genome.json",
    ) -> None:

        self.soul = SoulManager()
        self.memory = MemoryManager()
        self.knowledge = KnowledgeService()
        self.skills = SkillManager()
        self.genome = GenomeLoader.load(
            genome_manifest
        )
        self.context = ContextManager()
        self.tools = ToolRuntime()

        errors = GenomeValidator.validate(
            self.genome
        )

        if errors:
            raise ValueError(
                "Invalid Gene genome: "
                + "; ".join(errors)
            )

    def state(self) -> GeneState:

        return GeneState(
            identity=self.soul.identity(),
            active_sparks=self.soul.active_sparks(),
            genome={
                "name": self.genome.name,
                "version": self.genome.version,
                "modules": [
                    {
                        "name": module.name,
                        "category": module.category,
                        "enabled": module.enabled,
                        "version": module.version,
                    }
                    for module in self.genome.modules
                ],
            },
            memory_status=self.memory.status(),
            knowledge_status=self.knowledge.status(),
            skill_status=self.skills.status(),
            context_config=self.context.configuration(),
        )

    def health(self) -> dict:

        state = self.state()

        return {
            "status": "healthy",
            "identity": state.identity,
            "genome": {
                "name": self.genome.name,
                "version": self.genome.version,
                "modules": len(
                    self.genome.enabled_modules()
                ),
            },
            "memory": state.memory_status,
            "knowledge": state.knowledge_status,
            "skills": state.skill_status,
            "context": state.context_config,
            "tools": len(
                self.tools.capability_status()
            ),
        }
