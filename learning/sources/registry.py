from __future__ import annotations

from .base import LearningSource


class LearningSourceRegistry:

    def __init__(self) -> None:
        self._sources: dict[
            str,
            LearningSource,
        ] = {}

    def register(
        self,
        source: LearningSource,
        *,
        replace: bool = False,
    ) -> None:

        source_id = source.info.id

        if (
            source_id in self._sources
            and not replace
        ):
            raise ValueError(
                f"Learning source already exists: "
                f"{source_id}"
            )

        self._sources[
            source_id
        ] = source

    def unregister(
        self,
        source_id: str,
    ) -> bool:

        return (
            self._sources.pop(
                source_id,
                None,
            )
            is not None
        )

    def get(
        self,
        source_id: str,
    ) -> LearningSource:

        try:
            return self._sources[
                source_id
            ]
        except KeyError:
            raise KeyError(
                f"Unknown learning source: "
                f"{source_id}"
            ) from None

    def discover(
        self,
        *,
        kind: str | None = None,
        capability: str | None = None,
    ) -> list[LearningSource]:

        results = []

        for source in self._sources.values():

            info = source.info

            if kind is not None and info.kind != kind:
                continue

            if (
                capability is not None
                and capability
                not in info.capabilities
            ):
                continue

            if source.available():
                results.append(source)

        return results

    def list(self) -> list[LearningSource]:
        return list(
            self._sources.values()
        )

    def status(self) -> dict:

        return {
            "count":
                len(self._sources),

            "sources": [
                {
                    "id":
                        source.info.id,
                    "name":
                        source.info.name,
                    "kind":
                        source.info.kind,
                    "capabilities":
                        list(
                            source.info.capabilities
                        ),
                    "available":
                        source.available(),
                }
                for source
                in self._sources.values()
            ],
        }
