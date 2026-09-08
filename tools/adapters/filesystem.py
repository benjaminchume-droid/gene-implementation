from pathlib import Path
from typing import Any


class FilesystemAdapter:
    """Safe filesystem operations exposed to Gene."""

    def read(self, path: str) -> dict[str, Any]:
        target = Path(path).resolve()

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

        try:
            return {
                "success": True,
                "path": str(target),
                "content": target.read_text(encoding="utf-8"),
            }
        except UnicodeDecodeError:
            return {
                "success": False,
                "error": "File is not valid UTF-8 text",
                "path": str(target),
            }

    def write(self, path: str, content: str) -> dict[str, Any]:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            target.write_text(content, encoding="utf-8")

            return {
                "success": True,
                "path": str(target),
                "bytes": target.stat().st_size,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "path": str(target),
            }

    def list(self, path: str) -> dict[str, Any]:
        target = Path(path).resolve()

        if not target.exists():
            return {
                "success": False,
                "error": "Directory does not exist",
                "path": str(target),
            }

        if not target.is_dir():
            return {
                "success": False,
                "error": "Path is not a directory",
                "path": str(target),
            }

        entries = []

        for item in target.iterdir():
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

    def search(self, root: str, query: str) -> dict[str, Any]:
        target = Path(root).resolve()
        results = []

        if not target.exists():
            return {
                "success": False,
                "error": "Search root does not exist",
            }

        for item in target.rglob("*"):
            if query.lower() in item.name.lower():
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
