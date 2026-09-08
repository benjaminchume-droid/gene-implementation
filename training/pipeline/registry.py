from __future__ import annotations

import json
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DatasetSpec:
    id: str
    name: str
    source: str

    path: str

    license: str = "unknown"

    purpose: str = ""

    format: str = "jsonl"

    enabled: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    registered_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )


class DatasetRegistry:

    def __init__(
        self,
        path: str = (
            "gene/data/training/"
            "datasets.json"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.datasets: dict[
            str,
            DatasetSpec,
        ] = {}

        self._load()

    def _load(self) -> None:

        if not self.path.exists():
            return

        raw = self.path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not raw:
            return

        for item in json.loads(raw):

            spec = DatasetSpec(**item)

            self.datasets[
                spec.id
            ] = spec

    def _save(self) -> None:

        tmp = self.path.with_suffix(
            ".tmp"
        )

        tmp.write_text(
            json.dumps(
                [
                    asdict(spec)
                    for spec
                    in self.datasets.values()
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        tmp.replace(self.path)

    def register(
        self,
        name: str,
        source: str,
        path: str,
        license: str = "unknown",
        purpose: str = "",
        format: str = "jsonl",
        metadata: dict | None = None,
    ) -> DatasetSpec:

        dataset_id = str(
            uuid.uuid4()
        )

        spec = DatasetSpec(
            id=dataset_id,
            name=name,
            source=source,
            path=path,
            license=license,
            purpose=purpose,
            format=format,
            metadata=metadata or {},
        )

        self.datasets[
            dataset_id
        ] = spec

        self._save()

        return spec

    def add(
        self,
        spec: DatasetSpec,
    ) -> DatasetSpec:

        self.datasets[
            spec.id
        ] = spec

        self._save()

        return spec

    def get(
        self,
        dataset_id: str,
    ) -> DatasetSpec:

        try:
            return self.datasets[
                dataset_id
            ]
        except KeyError:
            raise KeyError(
                f"Unknown dataset: {dataset_id}"
            ) from None

    def enabled(self) -> list[DatasetSpec]:

        return [
            item
            for item
            in self.datasets.values()
            if item.enabled
        ]

    def list(self) -> list[DatasetSpec]:
        return list(
            self.datasets.values()
        )

    def status(self) -> dict:

        return {
            "total":
                len(self.datasets),
            "enabled":
                len(self.enabled()),
            "datasets": [
                asdict(item)
                for item
                in self.datasets.values()
            ],
            "storage":
                str(self.path),
        }
