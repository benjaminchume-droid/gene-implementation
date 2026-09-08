from __future__ import annotations

from ..interface import (
    ModelBackend,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
)


class NullModelBackend(ModelBackend):

    """
    Model-independent placeholder backend.

    This is NOT the neural model.
    It exists solely to prove the interface and
    runtime contract before the actual neural backend
    is implemented.
    """

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            name="null",
            backend="runtime",
            parameter_count=0,
            native_context=0,
            modalities=("text",),
            supports_generation=True,
            supports_embeddings=False,
            supports_tool_calling=False,
            supports_streaming=False,
        )

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:

        return ModelResponse(
            success=True,
            text="",
            model=self.capabilities.name,
            backend=self.capabilities.backend,
            metadata={
                "status":
                    "backend_ready",
            },
        )
