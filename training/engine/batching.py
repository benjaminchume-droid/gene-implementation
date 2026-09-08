from __future__ import annotations

from typing import Iterable

import torch


class TokenBatcher:

    def __init__(
        self,
        tokenizer,
        max_sequence_length: int,
        pad_token_id: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text)

        ids = (
            encoded.ids
            if hasattr(encoded, "ids")
            else list(encoded)
        )

        return ids[: self.max_sequence_length]

    def batch(
        self,
        texts: Iterable[str],
        batch_size: int,
    ):
        current = []

        for text in texts:
            ids = self.encode(text)

            if len(ids) < 2:
                continue

            current.append(ids)

            if len(current) >= batch_size:
                yield self._collate(current)
                current = []

        if current:
            yield self._collate(current)

    def _collate(self, examples: list[list[int]]):
        maximum = max(len(item) for item in examples)

        input_ids = []
        attention_mask = []

        for ids in examples:
            padding = maximum - len(ids)

            input_ids.append(
                ids + [self.pad_token_id] * padding
            )

            attention_mask.append(
                [1] * len(ids) + [0] * padding
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
        }
