from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MixtureSource:
    name: str
    path: str
    weight: float


class DatasetMixture:

    def __init__(
        self,
        sources: list[MixtureSource],
    ) -> None:

        if not sources:
            raise ValueError(
                "At least one dataset source is required."
            )

        if any(
            source.weight <= 0
            for source in sources
        ):
            raise ValueError(
                "Dataset weights must be positive."
            )

        self.sources = sources

    def normalized(self) -> list[MixtureSource]:
        total = sum(
            source.weight
            for source in self.sources
        )

        return [
            MixtureSource(
                name=source.name,
                path=source.path,
                weight=source.weight / total,
            )
            for source in self.sources
        ]
