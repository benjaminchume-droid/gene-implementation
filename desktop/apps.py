from __future__ import annotations

import subprocess
from pathlib import Path


class ApplicationController:

    def __init__(self) -> None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "Application execution requires psutil."
            ) from exc

        self.psutil = psutil

    def list_running(self) -> list[dict]:
        results = []

        for process in self.psutil.process_iter(
            ["pid", "name", "exe"]
        ):
            try:
                info = process.info

                results.append({
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "exe": info.get("exe"),
                })

            except (
                self.psutil.NoSuchProcess,
                self.psutil.AccessDenied,
            ):
                continue

        return results

    def find(self, query: str) -> list[dict]:
        query = query.lower()

        return [
            item
            for item in self.list_running()
            if query in (item["name"] or "").lower()
            or query in (item["exe"] or "").lower()
        ]

    def launch(
        self,
        executable: str,
        arguments: list[str] | None = None,
    ) -> dict:

        arguments = arguments or []

        process = subprocess.Popen(
            [executable, *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return {
            "success": True,
            "pid": process.pid,
            "executable": executable,
            "arguments": arguments,
        }

    def open_target(
        self,
        target: str,
    ) -> dict:

        try:
            process = subprocess.Popen(
                [
                    "cmd",
                    "/c",
                    "start",
                    "",
                    target,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return {
                "success": True,
                "pid": process.pid,
                "target": target,
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "target": target,
            }
