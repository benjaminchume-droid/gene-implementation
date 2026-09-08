from __future__ import annotations

from gene.evolution.runtime import (
    EvolutionRuntime,
)

from gene.evolution.external import (
    EvolutionController,
    ExternalParameterStore,
)

from gene.scheduler import TaskScheduler

from gene.desktop.vision import (
    VisionService,
)

from gene.tools.mcp_lifecycle import (
    MCPLifecycle,
)

from .execution import ActionExecutor


class GeneControlPlane:

    """
    Final agent-layer coordination surface.

    This is deliberately independent of the neural model.
    """

    def __init__(
        self,
        policy,
        tools,
        desktop=None,
        browser=None,
    ) -> None:

        self.policy = policy
        self.tools = tools
        self.desktop = desktop
        self.browser = browser

        from gene.levels.power import (
            PowerController,
        )

        from gene.workers import (
            WorkerRuntime,
        )

        self.power = PowerController()

        from gene.knowledge.v2 import (
            LongTermIntelligence,
        )

        self.workers = (
            WorkerRuntime()
        )

        from gene.model.agent_runtime import (
            GeneModelRuntime,
        )

        self.long_term = (
            LongTermIntelligence()
        )

        self.model = (
            GeneModelRuntime()
        )

        self.parameters = (
            ExternalParameterStore()
        )

        self.evolution = (
            EvolutionController(
                self.parameters
            )
        )

        self.evolution_runtime = (
            EvolutionRuntime()
        )

        self.scheduler = (
            TaskScheduler()
        )

        self.vision = (
            VisionService()
        )

        from gene.mcp.runtime import MCPRuntime

        self.mcp = MCPRuntime()

        self.actions = ActionExecutor(
            policy=policy,
            tools=tools,
            desktop=desktop,
            browser=browser,
        )

    def status(self) -> dict:

        return {
            "power":
                self.power.status(),

            "workers":
                self.workers.status(),

            "long_term":
                self.long_term.status(),

            "model":
                self.model.status(),

            "external_parameters":
                self.evolution_runtime.status(),

            "scheduler": {
                "tasks":
                    len(
                        self.scheduler.list()
                    ),
            },

            "mcp":
                self.mcp.status(),

            "vision": {
                "provider":
                    self.vision.provider.name,
            },
        }