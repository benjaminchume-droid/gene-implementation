from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextChunk:
    input_ids: list[int]
    start: int
    end: int


def chunk_token_sequence(
    input_ids: list[int],
    chunk_size: int = 8192,
    overlap: int = 1024,
) -> list[ContextChunk]:

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    if not input_ids:
        return []

    stride = chunk_size - overlap

    chunks = []

    start = 0

    while start < len(input_ids):

        end = min(
            start + chunk_size,
            len(input_ids),
        )

        chunks.append(
            ContextChunk(
                input_ids=input_ids[
                    start:end
                ],
                start=start,
                end=end,
            )
        )

        if end >= len(input_ids):
            break

        start += stride

    return chunks
