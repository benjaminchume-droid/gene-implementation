from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random


@dataclass(frozen=True)
class MixtureSource:
    name: str
    path: str
    weight: float
    enabled: bool = True
    metadata: dict | None = None


class MixtureScheduler:

    def __init__(
        self,
        sources: list[MixtureSource],
        seed: int = 42,
    ) -> None:

        if not sources:
            raise ValueError(
                "At least one mixture source is required."
            )

        for source in sources:
            if source.weight <= 0:
                raise ValueError(
                    f"Invalid weight for {source.name}: "
                    f"{source.weight}"
                )

        self.sources = [
            source
            for source in sources
            if source.enabled
        ]

        if not self.sources:
            raise ValueError(
                "At least one enabled source is required."
            )

        self.seed = seed
        self.random = random.Random(seed)

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
                enabled=True,
                metadata=source.metadata or {},
            )
            for source in self.sources
        ]

    def probabilities(self) -> dict[str, float]:

        return {
            source.name: source.weight
            for source in self.normalized()
        }

    def sample(
        self,
        count: int,
    ) -> list[MixtureSource]:

        if count < 0:
            raise ValueError(
                "count cannot be negative."
            )

        normalized = self.normalized()

        names = [
            source.name
            for source in normalized
        ]

        weights = [
            source.weight
            for source in normalized
        ]

        return [
            self.random.choices(
                normalized,
                weights=weights,
                k=1,
            )[0]
            for _ in range(count)
        ]

    def deterministic_schedule(
        self,
        count: int,
    ) -> list[MixtureSource]:

        if count < 0:
            raise ValueError(
                "count cannot be negative."
            )

        normalized = self.normalized()

        planned = []

        for source in normalized:

            planned.append(
                (
                    source,
                    source.weight * count,
                )
            )

        schedule = []

        for source, expected in planned:

            whole = int(expected)

            schedule.extend(
                [source] * whole
            )

        remaining = (
            count - len(schedule)
        )

        remainders = sorted(
            (
                (
                    source,
                    (source.weight * count)
                    - int(source.weight * count),
                )
                for source in normalized
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        for index in range(
            remaining
        ):
            schedule.append(
                remainders[index][0]
            )

        randomizer = random.Random(
            self.seed
        )

        randomizer.shuffle(
            schedule
        )

        return schedule

    def update_weight(
        self,
        name: str,
        weight: float,
    ) -> None:

        if weight <= 0:
            raise ValueError(
                "weight must be positive."
            )

        updated = []

        found = False

        for source in self.sources:

            if source.name == name:

                updated.append(
                    MixtureSource(
                        name=source.name,
                        path=source.path,
                        weight=weight,
                        enabled=source.enabled,
                        metadata=source.metadata,
                    )
                )

                found = True

            else:
                updated.append(
                    source
                )

        if not found:
            raise KeyError(
                f"Unknown mixture source: {name}"
            )

        self.sources = updated

    def status(self) -> dict:

        return {
            "sources": [
                asdict(source)
                for source in self.normalized()
            ],
            "probabilities":
                self.probabilities(),
            "seed":
                self.seed,
        }


class MixtureManifest:

    def __init__(
        self,
        path: str = (
            "gene/data/training/mixture/"
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
        scheduler: MixtureScheduler,
        metadata: dict | None = None,
    ) -> dict:

        payload = {
            "version": 1,
            "scheduler":
                scheduler.status(),
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

        temporary.replace(
            self.path
        )

        return payload

    def load(self) -> dict:

        if not self.path.exists():
            raise FileNotFoundError(
                str(self.path)
            )

        return json.loads(
            self.path.read_text(
                encoding="utf-8-sig"
            )
        )
