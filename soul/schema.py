from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Soul:
    name: str = "Gene"
    age: str = "1"
    soul_spark: str = "assistant"
    guidelines: list[str] = field(default_factory=list)

    def add_guideline(self, guideline: str) -> None:
        self.guidelines.append(guideline)
