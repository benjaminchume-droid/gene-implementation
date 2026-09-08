from __future__ import annotations

from .base import Benchmark


class BenchmarkRegistry:

    def __init__(self) -> None:
        self._items = {}

    def register(
        self,
        benchmark: Benchmark,
        *,
        replace: bool = False,
    ) -> None:

        if (
            benchmark.name
            in self._items
            and not replace
        ):
            raise ValueError(
                f"Benchmark already exists: "
                f"{benchmark.name}"
            )

        self._items[
            benchmark.name
        ] = benchmark

    def get(
        self,
        name: str,
    ) -> Benchmark:

        return self._items[name]

    def list(self):
        return list(
            self._items.values()
        )

    def status(self):
        return {
            "count":
                len(self._items),
            "benchmarks":
                list(
                    self._items.keys()
                ),
        }
