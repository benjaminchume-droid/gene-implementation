from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class CodeService:
    """
    Code execution boundary.

    This is deliberately not a sandbox yet. Production execution must be
    isolated before arbitrary model-generated code is allowed.
    """

    def run_python_file(self, path: str) -> dict:
        result = subprocess.run(
            [sys.executable, str(Path(path))],
            capture_output=True,
            text=True,
        )

        return {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
