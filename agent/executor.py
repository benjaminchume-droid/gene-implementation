from __future__ import annotations


class AgentExecutor:

    def __init__(
        self,
        tools,
        desktop=None,
        scheduler=None,
    ):
        self.tools = tools
        self.desktop = desktop
        self.scheduler = scheduler

    def execute(
        self,
        route: str,
        instruction: str,
    ) -> dict:

        if route == "filesystem":
            return self._filesystem(instruction)

        if route == "screen":
            return self._screen()

        if route == "desktop":
            return {
                "success": False,
                "requires_model_planning": True,
                "route": route,
                "message": (
                    "Desktop task identified. "
                    "Detailed action planning will be "
                    "handled by the model layer."
                ),
            }

        if route == "web":
            return {
                "success": False,
                "requires_model_planning": True,
                "route": route,
                "message": (
                    "Web task identified. "
                    "Detailed navigation/search planning "
                    "will be handled by the model layer."
                ),
            }

        if route == "coding":
            return {
                "success": False,
                "requires_model_planning": True,
                "route": route,
                "message": (
                    "Coding task identified. "
                    "Detailed implementation planning "
                    "will be handled by the model layer."
                ),
            }

        if route == "scheduler":
            return {
                "success": False,
                "requires_model_planning": True,
                "route": route,
            }

        return {
            "success": True,
            "route": "general",
            "message": (
                "General reasoning task received."
            ),
        }

    def _filesystem(
        self,
        instruction: str,
    ) -> dict:

        text = instruction.lower()

        if any(
            x in text
            for x in [
                "list files",
                "list the files",
                "show files",
                "show the files",
                "list folder",
            ]
        ):
            return self.tools.execute(
                "filesystem.list",
                path=".",
            )

        return {
            "success": False,
            "route": "filesystem",
            "message": (
                "Filesystem target/action requires "
                "model planning."
            ),
        }

    def _screen(self) -> dict:

        if self.desktop is None:
            return {
                "success": False,
                "error": "Desktop runtime unavailable.",
            }

        try:
            return self.desktop.inspect_screen()
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
            }
