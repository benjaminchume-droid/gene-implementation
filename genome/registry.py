from __future__ import annotations

from .schema import Genome


class GenomeRegistry:

    def __init__(self) -> None:
        self._genomes: dict[str, Genome] = {}

    def register(self, genome: Genome) -> None:
        if genome.name in self._genomes:
            raise ValueError(
                f"Genome already registered: {genome.name}"
            )

        self._genomes[genome.name] = genome

    def get(self, name: str) -> Genome:
        try:
            return self._genomes[name]
        except KeyError:
            raise KeyError(f"Unknown genome: {name}") from None

    def has(self, name: str) -> bool:
        return name in self._genomes

    def list(self) -> list[Genome]:
        return list(self._genomes.values())

    def names(self) -> list[str]:
        return list(self._genomes.keys())
