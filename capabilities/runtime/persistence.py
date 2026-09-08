from __future__ import annotations

import json

from dataclasses import asdict
from pathlib import Path

from .models import RuntimeCapability
from .registry import CapabilityRuntimeRegistry


class CapabilityPersistence:

    def __init__(
        self,
        path: str = (
            "gene/data/capabilities/"
            "runtime.json"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        registry: CapabilityRuntimeRegistry,
    ) -> None:

        payload = [
            asdict(capability)
            for capability
            in registry.list()
        ]

        temporary = self.path.with_suffix(
            ".tmp"
        )

        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temporary.replace(
            self.path
        )

    def load(
        self,
        registry: CapabilityRuntimeRegistry,
    ) -> int:

        if not self.path.exists():
            return 0

        raw = self.path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not raw:
            return 0

        records = json.loads(raw)

        count = 0

        for record in records:

            registry.register(
                RuntimeCapability(
                    **record
                ),
                replace=True,
            )

            count += 1

        return count
