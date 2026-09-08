from __future__ import annotations

from gene.model.interface import (
    ModelBackend,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
)

from gene.model.neural.transformer import (
    GeneTransformer,
)
from gene.model.neural.config import (
    TransformerConfig,
)


class GeneNeuralBackend(ModelBackend):

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:

        self.model = GeneTransformer(
            config
        )

        self._loaded = False

    @property
    def capabilities(
        self,
    ) -> ModelCapabilities:

        return ModelCapabilities(
            name=self.model.config.model_name,
            backend="gene_neural",
            parameter_count=
                self.model.parameter_count(),
            native_context=
                self.model.config.max_position_embeddings,
            modalities=("text",),
            supports_generation=True,
            supports_embeddings=False,
            supports_tool_calling=False,
            supports_streaming=False,
            metadata={
                "config":
                    self.model.config.to_dict(),
            },
        )

    def load(self) -> None:
        self.model.eval()
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:

        if not self._loaded:
            self.load()

        if not request.metadata.get(
            "input_ids"
        ):

            return ModelResponse(
                success=False,
                model=
                    self.capabilities.name,
                backend=
                    self.capabilities.backend,
                error=(
                    "Neural backend requires "
                    "tokenized input_ids in "
                    "request.metadata."
                ),
            )

        input_ids = request.metadata[
            "input_ids"
        ]

        output = self.model.generate(
            input_ids,
            max_new_tokens=
                request.max_tokens,
            temperature=
                request.temperature,
        )

        return ModelResponse(
            success=True,
            model=
                self.capabilities.name,
            backend=
                self.capabilities.backend,
            metadata={
                "output_ids":
                    output,
            },
        )
