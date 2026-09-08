import os
import shutil
from pathlib import Path


class AppTools:

    def find(self, name: str):
        matches = []

        command = shutil.which(name)

        if command:
            matches.append({
                "name": name,
                "path": command,
                "type": "command",
            })

        roots = [
            Path(os.environ.get("ProgramFiles", "")),
            Path(os.environ.get("ProgramFiles(x86)", "")),
            Path(os.environ.get("LOCALAPPDATA", "")),
        ]

        query = name.lower()

        for root in roots:
            if not root.exists():
                continue

            try:
                for item in root.rglob("*.exe"):
                    if query in item.stem.lower():
                        matches.append({
                            "name": item.stem,
                            "path": str(item),
                            "type": "application",
                        })

                        if len(matches) >= 25:
                            break
            except (PermissionError, OSError):
                continue

            if len(matches) >= 25:
                break

        return {
            "success": True,
            "query": name,
            "results": matches,
            "count": len(matches),
        }
