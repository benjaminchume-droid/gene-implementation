from __future__ import annotations


class AgentRouter:

    def route(self, instruction: str) -> str:
        text = instruction.lower()

        if any(
            phrase in text
            for phrase in [
                "screen",
                "on my screen",
                "look at my screen",
                "screenshot",
            ]
        ):
            return "screen"

        if any(
            phrase in text
            for phrase in [
                "search the web",
                "scour the web",
                "search online",
                "website",
                "browse",
            ]
        ):
            return "web"

        if any(
            phrase in text
            for phrase in [
                "file",
                "folder",
                "directory",
                "document",
                "read this",
                "write this",
                "edit this",
            ]
        ):
            return "filesystem"

        if any(
            phrase in text
            for phrase in [
                "app",
                "application",
                "launch",
                "open",
                "shortcut",
            ]
        ):
            return "desktop"

        if any(
            phrase in text
            for phrase in [
                "code",
                "python",
                "program",
                "debug",
                "build",
            ]
        ):
            return "coding"

        if any(
            phrase in text
            for phrase in [
                "schedule",
                "remind",
                "later",
                "background",
            ]
        ):
            return "scheduler"

        return "general"

    def worker_for(self, route: str) -> str | None:
        return {
            "screen": "screen_analyst",
            "web": "web_researcher",
            "desktop": "local_operator",
            "filesystem": "local_operator",
            "coding": "local_operator",
        }.get(route)
