from __future__ import annotations

import json

from datetime import datetime, timezone
from pathlib import Path


class TrainingManifest:

    def __init__(
        self,
        path: str = (
            "gene/data/training/factory/"
            "manifest.json"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        factory,
        *,
        tokenizer: dict | None = None,
        mixture: dict | None = None,
        metadata: dict | None = None,
    ) -> dict:

        payload = {
            "version": 1,
            "created_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),
            "statistics":
                factory.statistics(),
            "tokenizer":
                tokenizer or {},
            "mixture":
                mixture or {},
            "metadata":
                metadata or {},
        }

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

        temporary.replace(self.path)

        return payload

    def load(self) -> dict:

        return json.loads(
            self.path.read_text(
                encoding="utf-8-sig"
            )
        )
