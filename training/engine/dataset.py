from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class JSONLTextDataset:

    def __init__(
        self,
        path: str,
        text_key: str = "text",
    ) -> None:
        self.path = Path(path)
        self.text_key = text_key

        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

    def __iter__(self) -> Iterator[str]:
        with self.path.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:

            for line in handle:
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(item, dict):
                    continue

                text = item.get(self.text_key)

                if isinstance(text, str) and text.strip():
                    yield text

    def count(self) -> int:
        return sum(1 for _ in self)
