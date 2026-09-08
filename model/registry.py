from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .interface import (
    ModelBackend,
    ModelCapabilities,
)


class ModelRegistry:

    def __init__(
        self,
        path: str = (
            "gene/data/models/registry.json"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.backends: dict[
            str,
            ModelBackend,
        ] = {}

        self.metadata: dict[
            str,
            ModelCapabilities,
        ] = {}

        self._load_metadata()

    def _load_metadata(self) -> None:

        if not self.path.exists():
            return

        raw = self.path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not raw:
            return

        data = json.loads(raw)

        for item in data:

            capabilities = ModelCapabilities(
                **item
            )

            self.metadata[
                capabilities.name
            ] = capabilities

    def _save_metadata(self) -> None:

        self.path.write_text(
            json.dumps(
                [
                    asdict(capabilities)
                    for capabilities
                    in self.metadata.values()
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def register(
        self,
        backend: ModelBackend,
        *,
        replace: bool = False,
    ) -> ModelCapabilities:

        capabilities = (
            backend.capabilities
        )

        if (
            capabilities.name
            in self.backends
            and not replace
        ):
            raise ValueError(
                "Model backend already registered: "
                f"{capabilities.name}"
            )

        self.backends[
            capabilities.name
        ] = backend

        self.metadata[
            capabilities.name
        ] = capabilities

        self._save_metadata()

        return capabilities

    def unregister(
        self,
        name: str,
    ) -> bool:

        removed = False

        if name in self.backends:
            del self.backends[name]
            removed = True

        if name in self.metadata:
            del self.metadata[name]
            removed = True

        if removed:
            self._save_metadata()

        return removed

    def get(
        self,
        name: str,
    ) -> ModelBackend:

        try:
            return self.backends[name]
        except KeyError:
            raise KeyError(
                f"Model backend not loaded: {name}"
            ) from None

    def capabilities(
        self,
        name: str,
    ) -> ModelCapabilities:

        try:
            return self.metadata[name]
        except KeyError:
            raise KeyError(
                f"Unknown model: {name}"
            ) from None

    def list(
        self,
    ) -> list[ModelCapabilities]:

        return list(
            self.metadata.values()
        )

    def status(self) -> dict:

        return {
            "registered":
                len(self.metadata),
            "loaded":
                len(self.backends),
            "models": [
                asdict(item)
                for item
                in self.metadata.values()
            ],
            "storage":
                str(self.path),
        }
