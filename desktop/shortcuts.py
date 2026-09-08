from __future__ import annotations

import subprocess
from pathlib import Path


class ShortcutService:

    def create_windows_shortcut(
        self,
        name: str,
        target: str,
        arguments: str = "",
        working_directory: str | None = None,
        location: str | None = None,
        description: str = "Gene AI shortcut",
    ) -> dict:

        directory = Path(
            location or (Path.home() / "Desktop")
        )

        directory.mkdir(parents=True, exist_ok=True)

        shortcut = directory / f"{name}.lnk"

        escaped_target = target.replace("'", "''")
        escaped_args = arguments.replace("'", "''")
        escaped_description = description.replace("'", "''")

        script = (
            "$ws = New-Object -ComObject WScript.Shell; "
            f"$s = $ws.CreateShortcut('{shortcut}'); "
            f"$s.TargetPath = '{escaped_target}'; "
            f"$s.Arguments = '{escaped_args}'; "
            f"$s.Description = '{escaped_description}'; "
            + (
                f"$s.WorkingDirectory = '{working_directory}'; "
                if working_directory
                else ""
            )
            + "$s.Save()"
        )

        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr.strip(),
            }

        return {
            "success": True,
            "path": str(shortcut),
            "target": target,
        }
