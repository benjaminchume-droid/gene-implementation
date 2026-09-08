from __future__ import annotations

import hashlib
import json
import re

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NormalizedRecord:

    text: str

    dataset_id: str
    source: str

    record_id: str

    split: str = "train"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class DatasetNormalizer:

    TEXT_KEYS = (
        "text",
        "content",
        "body",
        "document",
        "prompt",
        "completion",
        "response",
        "question",
        "answer",
        "instruction",
    )

    def normalize(
        self,
        item: dict[str, Any],
        dataset_id: str,
        source: str,
    ) -> NormalizedRecord | None:

        text = self._extract_text(
            item
        )

        if not text:
            return None

        normalized = self._clean(
            text
        )

        if not normalized:
            return None

        record_id = hashlib.sha256(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()

        metadata = {
            key: value
            for key, value
            in item.items()
            if key not in self.TEXT_KEYS
        }

        return NormalizedRecord(
            text=normalized,
            dataset_id=dataset_id,
            source=source,
            record_id=record_id,
            metadata=metadata,
        )

    def _extract_text(
        self,
        item: dict[str, Any],
    ) -> str:

        fragments = []

        for key in self.TEXT_KEYS:

            value = item.get(key)

            if isinstance(
                value,
                str,
            ):
                fragments.append(value)

            elif isinstance(
                value,
                list,
            ):

                for part in value:

                    if isinstance(
                        part,
                        str,
                    ):
                        fragments.append(
                            part
                        )

                    elif isinstance(
                        part,
                        dict,
                    ):
                        for value2 in part.values():
                            if isinstance(
                                value2,
                                str,
                            ):
                                fragments.append(
                                    value2
                                )

        if fragments:
            return "\n\n".join(
                fragments
            )

        # Generic fallback: recursively collect
        # human-readable strings from arbitrary JSON.
        collected = []

        def walk(value):

            if isinstance(value, str):
                collected.append(value)

            elif isinstance(value, dict):
                for child in value.values():
                    walk(child)

            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(item)

        return "\n\n".join(
            collected
        )

    @staticmethod
    def _clean(
        text: str,
    ) -> str:

        text = text.replace(
            "\x00",
            " ",
        )

        text = re.sub(
            r"\r\n?",
            "\n",
            text,
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()
