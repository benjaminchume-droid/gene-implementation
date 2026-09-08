from pathlib import Path


class FilesystemTools:

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    def _resolve(self, path: str) -> Path:
        target = Path(path)

        if not target.is_absolute():
            target = self.workspace.parent / target

        target = target.resolve()

        return target

    def read(self, path: str):
        target = self._resolve(path)

        if not target.exists():
            return {
                "success": False,
                "error": "File does not exist",
                "path": str(target),
            }

        if not target.is_file():
            return {
                "success": False,
                "error": "Path is not a file",
                "path": str(target),
            }

        return {
            "success": True,
            "path": str(target),
            "content": target.read_text(encoding="utf-8"),
        }

    def write(self, path: str, content: str):
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "path": str(target),
            "bytes": len(content.encode("utf-8")),
        }

    def list(self, path: str = "gene"):
        target = self._resolve(path)

        if not target.exists():
            return {
                "success": False,
                "error": "Directory does not exist",
                "path": str(target),
            }

        entries = []

        for item in sorted(target.iterdir(), key=lambda x: x.name.lower()):
            entries.append({
                "name": item.name,
                "path": str(item),
                "type": "directory" if item.is_dir() else "file",
            })

        return {
            "success": True,
            "path": str(target),
            "entries": entries,
        }

    def search(self, root: str, query: str):
        target = self._resolve(root)
        results = []

        if not target.exists():
            return {
                "success": False,
                "error": "Search root does not exist",
                "path": str(target),
            }

        query_lower = query.lower()

        for item in target.rglob("*"):
            if query_lower in item.name.lower():
                results.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                })

        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
        }
