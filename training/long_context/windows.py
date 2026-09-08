from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingWindow:
    input_ids: list[int]
    start: int
    end: int
    loss_start: int
    loss_end: int


def build_training_windows(
    input_ids: list[int],
    context_length: int = 112000,
    chunk_size: int = 8192,
    overlap: int = 1024,
) -> list[TrainingWindow]:

    if context_length <= 0:
        raise ValueError(
            "context_length must be positive."
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive."
        )

    if chunk_size > context_length:
        raise ValueError(
            "chunk_size cannot exceed context_length."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    if len(input_ids) > context_length:
        raise ValueError(
            "Input exceeds logical context length."
        )

    if not input_ids:
        return []

    stride = chunk_size - overlap

    windows = []

    start = 0

    while start < len(input_ids):

        end = min(
            start + chunk_size,
            len(input_ids),
        )

        # Tokens in the overlap are context only.
        # Loss begins at the first non-overlapping token.
        loss_start = (
            start
            if start == 0
            else min(
                start + overlap,
                end,
            )
        )

        windows.append(
            TrainingWindow(
                input_ids=input_ids[
                    start:end
                ],
                start=start,
                end=end,
                loss_start=loss_start,
                loss_end=end,
            )
        )

        if end >= len(input_ids):
            break

        start += stride

    return windows
