from __future__ import annotations

from gene.context import ContextManager
from gene.genome import GenomeLoader
from gene.knowledge import KnowledgeService
from gene.memory import MemoryManager
from gene.skills import SkillManager
from gene.soul import SoulManager
from gene.tools.runtime import ToolRuntime

from .executor import AgentExecutor
from .router import AgentRouter
from .task import AgentTask


class GeneAgent:

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

        self.desktop = None
        self.scheduler = None
        self.browser = None

        # Desktop services are optional at import/runtime level.
        # They become active when their dependencies are available.
        self.desktop_runtime = None
        self.desktop = None
        self.desktop_error = None

        try:
            from gene.desktop.runtime import DesktopRuntime
            self.desktop_runtime = DesktopRuntime()
            self.desktop = self.desktop_runtime.agent
        except Exception as exc:
            self.desktop_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

        try:
            from gene.scheduler import TaskScheduler

            self.scheduler = TaskScheduler()

        except Exception as exc:
            self.scheduler = None
            self.scheduler_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

        try:
            from gene.browser import BrowserService

            self.browser = BrowserService()

        except Exception:
            self.browser = None


        self.voice_runtime = None

        try:
            from gene.voice.runtime import VoiceRuntime

            voice_config = (
                self.desktop_runtime.voice_config
                if self.desktop_runtime is not None
                else self.soul_voice_config()
            )

            self.voice_runtime = VoiceRuntime(
                config=voice_config,
                on_command=self.run_voice_command,
            )

        except Exception as exc:
            self.voice_runtime = None
            self.voice_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

        from gene.levels import (
            GeneLevel,
            LevelManager,
        )
        from gene.levels.power import (
            PowerController,
        )

        self.power_controller = (
            PowerController(
                GeneLevel.INSIGHT
            )
        )

        from gene.workers import (
            WorkerRuntime,
        )

        self.workers = (
            WorkerRuntime()
        )

        from gene.model.agent_runtime import (
            GeneModelRuntime,
        )

        self.model_runtime = (
            GeneModelRuntime()
        )

        self.router = AgentRouter()

        self.executor = AgentExecutor(
            tools=self.tools,
            desktop=self.desktop,
            scheduler=self.scheduler,
        )

    def build_context(
        self,
        instruction: str,
    ) -> str:

        memories = [
            item["content"]
            for item in self.memory.recall(
                instruction,
                limit=8,
            )
        ]

        knowledge = [
            item.content
            for item in self.knowledge.search(
                instruction,
                limit=8,
            )
        ]

        skills = [
            (
                f"{skill.name}: "
                f"{skill.description}"
            )
            for skill in self.skills.registry.list()
            if skill.enabled
        ]

        soul = self.soul.identity()

        system = [
            f"Name: {soul['name']}",
            f"Role: {soul['role']}",
            (
                "Active Soul Sparks: "
                + ", ".join(
                    self.soul.active_sparks()
                )
            ),
            (
                "Gene is modular and must verify "
                "actions before claiming success."
            ),
        ]

        return self.context.build(
            instruction,
            system=system,
            memories=memories,
            knowledge=knowledge,
            skills=skills,
            conversation=[],
        )

    def run(
        self,
        instruction: str,
    ) -> dict:

        task = AgentTask(
            instruction=instruction
        )

        task.update("routing")

        task.route = self.router.route(
            instruction
        )

        task.worker = self.router.worker_for(
            task.route
        )

        task.update("context")

        task.context = self.build_context(
            instruction
        )

        task.update("executing")

        result = self.executor.execute(
            task.route,
            instruction,
        )

        task.observations.append(result)

        if result.get("success"):
            task.update("verified")

            task.result = result

            self.memory.remember(
                instruction,
                memory_type="interaction",
                importance=0.4,
                metadata={
                    "task_id": task.id,
                    "route": task.route,
                    "worker": task.worker,
                    "result": "success",
                },
            )

            task.update("completed")

        elif result.get(
            "requires_confirmation"
        ):
            task.update(
                "awaiting_confirmation"
            )
            task.result = result

        elif result.get(
            "requires_model_planning"
        ):
            task.update(
                "awaiting_model"
            )
            task.result = result

        else:
            task.update("failed")
            task.error = result.get(
                "error",
                result.get("message"),
            )

        return {
            "task_id": task.id,
            "status": task.status,
            "route": task.route,
            "worker": task.worker,
            "observations": task.observations,
            "result": task.result,
            "error": task.error,
        }


    def soul_voice_config(self):
        from gene.voice.config import VoiceConfig
        return VoiceConfig()

    def run_voice_command(
        self,
        command: str,
    ) -> str:
        result = self.run(command)

        if result["status"] == "completed":
            return str(
                result.get("result")
                or "Task completed."
            )

        if result["status"] == "awaiting_model":
            return (
                "I understand the task, "
                "but the neural model is not "
                "connected yet."
            )

        if result["status"] == "awaiting_confirmation":
            return (
                "That action requires confirmation."
            )

        return (
            result.get("error")
            or "I could not complete that task."
        )

    def start_voice(self) -> dict:
        if self.voice_runtime is None:
            return {
                "success": False,
                "error": "Voice runtime unavailable.",
            }

        return {
            "success": True,
            "voice": self.voice_runtime.start(),
        }

    def stop_voice(self) -> dict:
        if self.voice_runtime is None:
            return {
                "success": False,
                "error": "Voice runtime unavailable.",
            }

        return {
            "success": True,
            "voice": self.voice_runtime.stop(),
        }

    def model_status(self) -> dict:
        if self.model_runtime is None:
            return {
                "available": False,
                "error": "Model runtime unavailable.",
            }

        return self.model_runtime.status()

    def status(self) -> dict:

        return {
            "level":
                self.power_controller.status(),

            "workers":
                self.workers.status(),
            "name": self.soul.identity()["name"],
            "genome": self.genome.name,
            "genome_version":
                self.genome.version,
            "active_sparks":
                self.soul.active_sparks(),
            "memory":
                self.memory.status(),
            "knowledge":
                self.knowledge.status(),
            "skills":
                self.skills.status(),
            "context":
                self.context.configuration(),
            "tools":
                len(
                    self.tools.capability_status()
                ),
            "desktop":
                self.desktop is not None,
            "scheduler": {
                "available":
                    self.scheduler is not None,
                "running":
                    self.scheduler.running
                    if self.scheduler
                    else False,
                "status":
                    self.scheduler.status()
                    if self.scheduler
                    else None,
                "error":
                    getattr(
                        self,
                        "scheduler_error",
                        None,
                    ),
            },
            "model_runtime":
                self.model_status(),
        }