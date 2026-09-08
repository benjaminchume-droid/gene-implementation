import subprocess
from typing import Optional


class ProcessTools:

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 120,
    ):
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
            }

        except subprocess.TimeoutExpired as exc:
            return {
                "success": False,
                "error": "Process timed out",
                "command": command,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "command": command,
            }
