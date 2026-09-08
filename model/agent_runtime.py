from __future__ import annotations

from gene.model import (
    ModelRequest,
    ModelRuntime,
)
from gene.model.backends import (
    NullModelBackend,
)
from gene.model.registry import (
    ModelRegistry,
)


class GeneModelRuntime:

    def __init__(
        self,
    ) -> None:

        self.registry = ModelRegistry()

        self.runtime = ModelRuntime(
            self.registry
        )

        # The null backend only verifies that Gene's
        # agent stack does not depend on a specific model.
        if "null" not in self.registry.backends:
            self.runtime.register(
                NullModelBackend()
            )

    def status(self) -> dict:
        return self.runtime.status()

    def generate(
        self,
        request: ModelRequest,
    ):
        return self.runtime.generate(
            request
        )
