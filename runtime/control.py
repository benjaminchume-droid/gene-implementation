from __future__ import annotations

from gene.desktop.config import (
    DesktopConfiguration,
)

from gene.tools.runtime import ToolRuntime
from gene.agent.control_plane import (
    GeneControlPlane,
)


class ControlRuntime:

    def __init__(self):

        self.desktop_config = (
            DesktopConfiguration()
        )

        voice, policy = (
            self.desktop_config.objects()
        )

        self.voice = voice
        self.policy = policy

        self.tools = ToolRuntime()

        self.control = GeneControlPlane(
            policy=policy,
            tools=self.tools,
        )

    def status(self) -> dict:
        return self.control.status()