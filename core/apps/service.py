from __future__ import annotations

import shutil
import subprocess


class ApplicationService:
    def find(self, executable: str) -> str | None:
        return shutil.which(executable)

    def launch(
        self,
        executable: str,
        *arguments: str,
    ) -> subprocess.Popen:
        return subprocess.Popen(
            [executable, *arguments],
            shell=False,
        )
