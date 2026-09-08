from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .deduplicate import Deduplicator
from .normalize import DatasetNormalizer
from .registry import (
    DatasetRegistry,
)
from .split import DatasetSplitter


class TrainingDataPipeline:

    def __init__(
        self,
        registry: DatasetRegistry | None = None,
        output_root: str = (
            "gene/data/training/normalized"
        ),
    ) -> None:

        self.registry = (
            registry
            or DatasetRegistry()
        )

        self.output_root = Path(
            output_root
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.normalizer = (
            DatasetNormalizer()
        )

        self.deduplicator = (
            Deduplicator()
        )

        self.splitter = (
            DatasetSplitter()
        )

    def _iter_jsonl(
        self,
        path: Path,
    ) -> Iterator[dict]:

        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:

            for line in handle:

                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(
                        line
                    )
                except json.JSONDecodeError:
                    continue

                if isinstance(
                    item,
                    dict,
                ):
                    yield item

    def _iter_files(
        self,
        root: Path,
    ):

        for path in sorted(
            root.rglob("*")
        ):

            if not path.is_file():
                continue

            if path.suffix.lower() == ".jsonl":
                yield path

    def normalize_dataset(
        self,
        dataset,
    ) -> dict:

        root = Path(
            dataset.path
        )

        if not root.exists():
            return {
                "success": False,
                "dataset":
                    dataset.name,
                "error":
                    f"Path does not exist: {root}",
            }

        train_path = (
            self.output_root
            / f"{dataset.id}_train.jsonl"
        )

        validation_path = (
            self.output_root
            / f"{dataset.id}_validation.jsonl"
        )

        train_count = 0
        validation_count = 0
        skipped = 0
        duplicates = 0

        with (
            train_path.open(
                "w",
                encoding="utf-8",
            ) as train_handle,
            validation_path.open(
                "w",
                encoding="utf-8",
            ) as validation_handle
        ):

            for file in self._iter_files(
                root
            ):

                for item in self._iter_jsonl(
                    file
                ):

                    normalized = (
                        self.normalizer.normalize(
                            item,
                            dataset.id,
                            dataset.source,
                        )
                    )

                    if normalized is None:
                        skipped += 1
                        continue

                    if not self.deduplicator.accept(
                        normalized.text
                    ):
                        duplicates += 1
                        continue

                    split = (
                        self.splitter.assign(
                            normalized.record_id
                        )
                    )

                    normalized.split = split

                    line = (
                        json.dumps(
                            asdict(
                                normalized
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    if split == "validation":

                        validation_handle.write(
                            line
                        )

                        validation_count += 1

                    else:

                        train_handle.write(
                            line
                        )

                        train_count += 1

        return {
            "success": True,
            "dataset":
                dataset.name,
            "train":
                train_count,
            "validation":
                validation_count,
            "skipped":
                skipped,
            "duplicates":
                duplicates,
            "train_path":
                str(train_path),
            "validation_path":
                str(validation_path),
        }

    def run(
        self,
    ) -> dict:

        results = []

        for dataset in (
            self.registry.enabled()
        ):

            results.append(
                self.normalize_dataset(
                    dataset
                )
            )

        return {
            "success":
                all(
                    item["success"]
                    for item
                    in results
                )
                if results
                else True,

            "datasets":
                results,

            "unique_records":
                self.deduplicator.count,
        }
