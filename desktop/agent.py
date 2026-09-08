from __future__ import annotations

from .apps import ApplicationController
from .computer import ComputerControl
from .policy import ConfirmationMode, DesktopPolicy
from .screen import ScreenService
from .shortcuts import ShortcutService


class DesktopAgent:

    def __init__(
        self,
        policy: DesktopPolicy,
    ) -> None:

        self.policy = policy

        self.screen = (
            ScreenService()
            if policy.allow_screen_capture
            else None
        )

        self.computer = ComputerControl()
        self.app_controller = ApplicationController()
        self.shortcuts = ShortcutService()

    def inspect_screen(
        self,
        output: str = "gene/data/screens/latest.png",
    ) -> dict:

        if not self.policy.allow_screen_capture:
            return {
                "success": False,
                "error":
                    "Screen capture disabled by policy.",
            }

        if self.screen is None:
            self.screen = ScreenService()

        return self.screen.capture(output)

    def analyze_screen(
        self,
        instruction: str = "",
        output: str = (
            "gene/data/vision/latest.png"
        ),
    ) -> dict:

        from gene.vision.service import (
            VisionService,
        )

        service = VisionService()

        return service.analyze_screen(
            self,
            instruction=instruction,
            output=output,
        )

    def mouse_position(self) -> dict:
        return self.computer.position()

    def move_mouse(
        self,
        x: int,
        y: int,
        duration: float = 0.15,
    ) -> dict:
        return self.computer.move(
            x,
            y,
            duration,
        )

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
    ) -> dict:
        return self.computer.click(
            x=x,
            y=y,
            button=button,
            clicks=clicks,
        )

    def type_text(
        self,
        text: str,
        interval: float = 0.01,
    ) -> dict:
        return self.computer.type_text(
            text,
            interval,
        )

    def press(
        self,
        key: str,
    ) -> dict:
        return self.computer.press(key)

    def hotkey(
        self,
        *keys: str,
    ) -> dict:
        return self.computer.hotkey(*keys)

    def scroll(
        self,
        amount: int,
    ) -> dict:
        return self.computer.scroll(amount)

    def find_app(
        self,
        query: str,
    ) -> dict:
        return {
            "success": True,
            "results":
                self.app_controller.find(query),
        }

    def running_apps(self) -> dict:
        return {
            "success": True,
            "results":
                self.app_controller.list_running(),
        }

    def launch_app(
        self,
        executable: str,
        arguments: list[str] | None = None,
    ) -> dict:

        if not self.policy.allow_app_launch:
            return {
                "success": False,
                "error":
                    "Application launch disabled by policy.",
            }

        if (
            self.policy.confirmation_mode
            != ConfirmationMode.NEVER
        ):
            return {
                "success": False,
                "requires_confirmation": True,
                "error":
                    "Application launch requires confirmation.",
            }

        return self.app_controller.launch(
            executable,
            arguments,
        )

    def open_target(
        self,
        target: str,
    ) -> dict:

        if not self.policy.allow_app_launch:
            return {
                "success": False,
                "error":
                    "Target opening disabled by policy.",
            }

        return self.app_controller.open_target(
            target
        )

    def create_shortcut(
        self,
        name: str,
        target: str,
        arguments: str = "",
    ) -> dict:

        if not self.policy.allow_shortcut_creation:
            return {
                "success": False,
                "error":
                    "Shortcut creation disabled by policy.",
            }

        if (
            self.policy.confirmation_mode
            != ConfirmationMode.NEVER
        ):
            return {
                "success": False,
                "requires_confirmation": True,
                "error":
                    "Shortcut creation requires confirmation.",
            }

        return self.shortcuts.create_windows_shortcut(
            name=name,
            target=target,
            arguments=arguments,
        )
