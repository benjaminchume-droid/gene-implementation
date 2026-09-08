from __future__ import annotations

from .action import ActionRequest
from .confirmation import ConfirmationBroker


class ActionExecutor:

    def __init__(
        self,
        policy,
        tools,
        desktop=None,
        browser=None,
    ):
        self.policy = policy
        self.tools = tools
        self.desktop = desktop
        self.browser = browser

        self.confirmation = (
            ConfirmationBroker(policy)
        )

    def execute(
        self,
        action: str,
        payload: dict | None = None,
    ) -> dict:

        payload = payload or {}

        request = ActionRequest(
            action=action,
            payload=payload,
        )

        destructive = action in {
            "filesystem.delete",
            "process.kill",
        }

        elevated = action in {
            "process.execute_elevated",
            "desktop.admin_action",
        }

        external = action in {
            "browser.navigate",
            "browser.click",
            "browser.type",
            "network.request",
        }

        decision = self.confirmation.evaluate(
            action,
            destructive=destructive,
            elevated=elevated,
            external=external,
        )

        if not decision.approved:
            request.requires_confirmation = True
            request.confirmation_reason = (
                decision.reason
            )

            return {
                "success": False,
                "requires_confirmation": True,
                "action_id": request.id,
                "action": action,
                "reason": decision.reason,
            }

        request.approve()

        try:

            if action == "computer.position":
                result = self.desktop.mouse_position()

            elif action == "computer.move":
                result = self.desktop.move_mouse(
                    payload["x"],
                    payload["y"],
                    payload.get("duration", 0.15),
                )

            elif action == "computer.click":
                result = self.desktop.click(
                    x=payload.get("x"),
                    y=payload.get("y"),
                    button=payload.get(
                        "button",
                        "left",
                    ),
                    clicks=payload.get(
                        "clicks",
                        1,
                    ),
                )

            elif action == "computer.type":
                result = self.desktop.type_text(
                    payload["text"]
                )

            elif action == "computer.press":
                result = self.desktop.press(
                    payload["key"]
                )

            elif action == "computer.hotkey":
                result = self.desktop.hotkey(
                    *payload["keys"]
                )

            elif action == "computer.scroll":
                result = self.desktop.scroll(
                    payload["amount"]
                )

            elif action == "app.find":
                result = self.desktop.find_app(
                    payload["query"]
                )

            elif action == "app.list":
                result = self.desktop.running_apps()

            elif action == "app.launch":
                result = self.desktop.launch_app(
                    payload["executable"],
                    payload.get("arguments"),
                )

            elif action == "app.open":
                result = self.desktop.open_target(
                    payload["target"]
                )

            elif action == "screen.capture":
                result = self.desktop.inspect_screen(
                    payload.get(
                        "output",
                        "gene/data/screens/latest.png",
                    )
                )

            elif action == "screen.analyze":
                if self.desktop is None:
                    result = {
                        "success": False,
                        "error":
                            "Desktop runtime unavailable.",
                    }
                else:
                    from gene.vision.service import (
                        VisionService,
                    )

                    if not hasattr(
                        self,
                        "_vision",
                    ):
                        self._vision = (
                            VisionService()
                        )

                    result = (
                        self._vision.analyze_screen(
                            self.desktop,
                            instruction=payload.get(
                                "instruction",
                                "",
                            ),
                            output=payload.get(
                                "output",
                                "gene/data/vision/latest.png",
                            ),
                        )
                    )

            elif action == "browser.start":
                result = self.browser.start(
                    headless=payload.get(
                        "headless",
                        False,
                    )
                )

            elif action == "browser.navigate":
                result = self.browser.navigate(
                    payload["url"]
                )

            elif action == "browser.current":
                result = self.browser.current_page()

            elif action == "browser.find":
                result = self.browser.find(
                    payload["selector"]
                )

            elif action == "browser.text":
                result = self.browser.visible_text()

            elif action == "browser.content":
                result = self.browser.page_content()

            elif action == "browser.click":
                result = self.browser.click(
                    payload["selector"]
                )

            elif action == "browser.type":
                result = self.browser.type(
                    payload["selector"],
                    payload["text"],
                    payload.get(
                        "clear",
                        True,
                    ),
                )

            elif action == "browser.press":
                result = self.browser.press(
                    payload["key"],
                    payload.get(
                        "selector"
                    ),
                )

            elif action == "browser.scroll":
                result = self.browser.scroll(
                    payload["amount"]
                )

            elif action == "browser.screenshot":
                result = self.browser.screenshot(
                    payload.get(
                        "output",
                        "gene/data/browser/latest.png",
                    ),
                    payload.get(
                        "full_page",
                        False,
                    ),
                )

            elif action == "browser.download":
                result = self.browser.download(
                    payload["selector"],
                    payload.get(
                        "output_dir",
                        "gene/data/browser/downloads",
                    ),
                )

            elif action == "browser.close":
                result = self.browser.close()

            elif action == "filesystem.list":
                result = self.tools.execute(
                    "filesystem.list",
                    path=payload.get(
                        "path",
                        ".",
                    ),
                )

            else:
                return {
                    "success": False,
                    "error":
                        f"Unknown action: {action}",
                }

            if result.get("success"):
                request.complete()
            else:
                request.fail(
                    result.get(
                        "error",
                        "Action failed.",
                    )
                )

            return {
                **result,
                "action_id": request.id,
                "action_status": request.status,
            }

        except Exception as exc:

            request.fail(str(exc))

            return {
                "success": False,
                "action_id": request.id,
                "action_status": request.status,
                "error": str(exc),
            }