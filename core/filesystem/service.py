from __future__ import annotations

from pathlib import Path


class FileSystemService:
    def read_file(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target)

    def append_file(self, path: str, content: str) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as file:
            file.write(content)
        return str(target)

    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def list_directory(self, path: str = ".") -> list[str]:
        return [str(item) for item in Path(path).iterdir()]

    def search(self, root: str, pattern: str) -> list[str]:
        return [str(item) for item in Path(root).rglob(pattern)]
