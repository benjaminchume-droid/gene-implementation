from __future__ import annotations

from gene.agents.worker import (
    TaskWorker,
    WorkerBudget,
    WorkerManager,
)

from gene.voice.wake import WakeController

from .agent import DesktopAgent
from .config import DesktopConfiguration


class DesktopRuntime:

    def __init__(self):
        configuration = DesktopConfiguration()

        self.voice_config, self.policy = (
            configuration.objects()
        )

        self.workers = WorkerManager()

        self.workers.register(
            TaskWorker(
                name="screen_analyst",
                purpose="Analyze screen content.",
                budget=WorkerBudget(
                    reasoning_fraction=0.20,
                    tool_fraction=0.20,
                    context_fraction=0.40,
                    memory_fraction=0.20,
                ),
            )
        )

        self.workers.register(
            TaskWorker(
                name="web_researcher",
                purpose="Research external information.",
                budget=WorkerBudget(
                    reasoning_fraction=0.30,
                    tool_fraction=0.35,
                    context_fraction=0.20,
                    memory_fraction=0.15,
                ),
            )
        )

        self.workers.register(
            TaskWorker(
                name="local_operator",
                purpose="Perform bounded local computer tasks.",
                budget=WorkerBudget(
                    reasoning_fraction=0.25,
                    tool_fraction=0.45,
                    context_fraction=0.15,
                    memory_fraction=0.15,
                ),
            )
        )

        self.agent = DesktopAgent(
            policy=self.policy
        )

        self.wake = WakeController(
            config=self.voice_config
        )

    def status(self) -> dict:
        return {
            "voice": {
                "wake_phrase":
                    self.voice_config.wake_phrase,
                "hotkey":
                    self.voice_config.hotkey,
                "tts":
                    self.voice_config.tts_enabled,
                "stt":
                    self.voice_config.stt_enabled,
            },
            "desktop": {
                "privilege_level":
                    self.policy.privilege_level.name,
                "internet_level":
                    self.policy.internet_level.name,
                "confirmation_mode":
                    self.policy.confirmation_mode.value,
            },
            "workers": [
                worker.describe()
                for worker in self.workers.list()
            ],
        }
