from __future__ import annotations

import json
from pathlib import Path

from .schema import Genome, GenomeModule


class GenomeLoader:

    @staticmethod
    def load(path: str | Path) -> Genome:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Genome manifest not found: {path}"
            )

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        modules = [
            GenomeModule(
                name=module["name"],
                category=module["category"],
                enabled=module.get("enabled", True),
                priority=module.get("priority", 0),
                version=module.get("version", "1.0"),
                configuration=module.get("configuration", {}),
            )
            for module in data.get("modules", [])
        ]

        return Genome(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            modules=modules,
            metadata=data.get("metadata", {}),
        )
