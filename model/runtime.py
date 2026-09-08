from __future__ import annotations

from .interface import (
    ModelBackend,
    ModelRequest,
    ModelResponse,
)
from .registry import ModelRegistry


class ModelRuntime:

    def __init__(
        self,
        registry: ModelRegistry | None = None,
    ) -> None:

        self.registry = (
            registry
            or ModelRegistry()
        )

        self.active_model: str | None = None

    def register(
        self,
        backend: ModelBackend,
        *,
        replace: bool = False,
    ):
        return self.registry.register(
            backend,
            replace=replace,
        )

    def select(
        self,
        model_name: str,
    ) -> dict:

        backend = self.registry.get(
            model_name
        )

        backend.load()

        self.active_model = model_name

        return {
            "success": True,
            "model": model_name,
            "capabilities":
                self.registry
                .capabilities(
                    model_name
                ).__dict__,
        }

    def unload(self) -> dict:

        if self.active_model is None:
            return {
                "success": True,
                "status": "no_model_selected",
            }

        backend = self.registry.get(
            self.active_model
        )

        backend.unload()

        previous = self.active_model
        self.active_model = None

        return {
            "success": True,
            "status": "unloaded",
            "model": previous,
        }

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:

        if self.active_model is None:
            return ModelResponse(
                success=False,
                error=(
                    "No model backend is selected."
                ),
            )

        backend = self.registry.get(
            self.active_model
        )

        return backend.generate(
            request
        )

    def embed(
        self,
        text: str,
    ) -> list[float]:

        if self.active_model is None:
            raise RuntimeError(
                "No model backend is selected."
            )

        backend = self.registry.get(
            self.active_model
        )

        return backend.embed(text)

    def health(self) -> dict:

        if self.active_model is None:
            return {
                "healthy": True,
                "active_model": None,
                "status": "no_model_selected",
            }

        backend = self.registry.get(
            self.active_model
        )

        return {
            **backend.health(),
            "active_model":
                self.active_model,
        }

    def status(self) -> dict:

        return {
            **self.registry.status(),
            "active_model":
                self.active_model,
        }
