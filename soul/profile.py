from __future__ import annotations

from dataclasses import dataclass, field

from .identity import Identity
from .spark import SoulSpark


@dataclass
class SoulProfile:
    identity: Identity
    sparks: dict[str, SoulSpark] = field(default_factory=dict)
    active_sparks: list[str] = field(default_factory=lambda: ["assistant"])
    guidelines: list[str] = field(default_factory=list)
    characteristics: list[str] = field(default_factory=list)

    def add_spark(self, spark: SoulSpark) -> None:
        self.sparks[spark.name] = spark

    def activate(self, name: str) -> None:
        if name not in self.sparks:
            raise KeyError(f"Unknown Soul Spark: {name}")

        if name not in self.active_sparks:
            self.active_sparks.append(name)

    def deactivate(self, name: str) -> None:
        if name in self.active_sparks:
            self.active_sparks.remove(name)

    def add_guideline(self, guideline: str) -> None:
        if guideline not in self.guidelines:
            self.guidelines.append(guideline)

    def add_characteristic(self, characteristic: str) -> None:
        if characteristic not in self.characteristics:
            self.characteristics.append(characteristic)

    def active_spark_profiles(self) -> list[SoulSpark]:
        return [
            self.sparks[name]
            for name in self.active_sparks
            if name in self.sparks
        ]

    def to_dict(self) -> dict:
        return {
            "identity": self.identity.to_dict(),
            "sparks": {
                name: spark.to_dict()
                for name, spark in self.sparks.items()
            },
            "active_sparks": self.active_sparks,
            "guidelines": self.guidelines,
            "characteristics": self.characteristics,
        }
