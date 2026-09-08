from __future__ import annotations

import hashlib
import json

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunManifest:

    run_id: str

    model_name: str

    model_config: dict[str, Any]

    training_config: dict[str, Any]

    tokenizer_path: str

    train_data_path: str

    validation_data_path: str | None

    hardware: dict[str, Any]

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return asdict(self)


class RunManifestStore:

    def __init__(
        self,
        root: str = (
            "gene/data/training/runs"
        ),
    ) -> None:

        self.root = Path(root)

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        manifest: RunManifest,
    ) -> str:

        path = (
            self.root
            / f"{manifest.run_id}.json"
        )

        temporary = path.with_suffix(
            ".tmp"
        )

        temporary.write_text(
            json.dumps(
                manifest.to_dict(),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        temporary.replace(path)

        return str(path)

    def load(
        self,
        run_id: str,
    ) -> dict:

        path = (
            self.root
            / f"{run_id}.json"
        )

        return json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )


def sha256_file(
    path: str,
) -> str:

    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda:
            handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()
