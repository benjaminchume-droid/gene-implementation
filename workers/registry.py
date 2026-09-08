from __future__ import annotations

import json
from dataclasses import asdict

from pathlib import Path

from .models import WorkerDefinition


class WorkerRegistry:

    def __init__(
        self,
        path: str = (
            "gene/data/workers/registry.json"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.workers: dict[
            str,
            WorkerDefinition,
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

            worker = WorkerDefinition(
                **item
            )

            self.workers[
                worker.name
            ] = worker

    def _save(self) -> None:

        tmp = self.path.with_suffix(
            ".tmp"
        )

        tmp.write_text(
            json.dumps(
                [
                    asdict(worker)
                    for worker
                    in self.workers.values()
                ],
                indent=2,
                ensure_ascii=False,
                default=lambda value:
                    value.value
                    if hasattr(
                        value,
                        "value",
                    )
                    else str(value),
            ),
            encoding="utf-8",
        )

        tmp.replace(
            self.path
        )

    def register(
        self,
        worker: WorkerDefinition,
    ) -> WorkerDefinition:

        if worker.name in self.workers:
            raise ValueError(
                f"Worker already exists: "
                f"{worker.name}"
            )

        self.workers[
            worker.name
        ] = worker

        self._save()

        return worker

    def update(
        self,
        name: str,
        **changes,
    ) -> WorkerDefinition:

        worker = self.get(name)

        for key, value in changes.items():

            if not hasattr(
                worker,
                key,
            ):
                raise ValueError(
                    f"Unknown worker field: "
                    f"{key}"
                )

            setattr(
                worker,
                key,
                value,
            )

        self._save()

        return worker

    def enable(
        self,
        name: str,
    ) -> WorkerDefinition:

        return self.update(
            name,
            enabled=True,
        )

    def disable(
        self,
        name: str,
    ) -> WorkerDefinition:

        return self.update(
            name,
            enabled=False,
        )

    def remove(
        self,
        name: str,
    ) -> bool:

        if name not in self.workers:
            return False

        del self.workers[
            name
        ]

        self._save()

        return True

    def get(
        self,
        name: str,
    ) -> WorkerDefinition:

        try:
            return self.workers[
                name
            ]
        except KeyError:
            raise KeyError(
                f"Unknown worker: {name}"
            ) from None

    def list(
        self,
        capability: str | None = None,
    ) -> list[WorkerDefinition]:

        values = list(
            self.workers.values()
        )

        if capability is None:
            return values

        return [
            worker
            for worker in values
            if capability
            in worker.capabilities
        ]

    def enabled(
        self,
    ) -> list[WorkerDefinition]:

        return [
            worker
            for worker
            in self.workers.values()
            if worker.enabled
        ]

    def status(self) -> dict:

        return {
            "total":
                len(self.workers),

            "enabled":
                len(self.enabled()),

            "workers": [
                {
                    "name":
                        worker.name,

                    "purpose":
                        worker.purpose,

                    "capabilities":
                        worker.capabilities,

                    "backend":
                        worker.backend,

                    "model":
                        worker.model,

                    "state":
                        worker.state.value,

                    "enabled":
                        worker.enabled,
                }
                for worker
                in self.workers.values()
            ],

            "storage":
                str(self.path),
        }
