from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelCapabilities:
    name: str
    backend: str

    parameter_count: int | None = None
    native_context: int | None = None

    modalities: tuple[str, ...] = ("text",)
    supports_generation: bool = True
    supports_embeddings: bool = False
    supports_tool_calling: bool = False
    supports_streaming: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ModelRequest:
    prompt: str | None = None

    messages: list[dict[str, Any]] = field(
        default_factory=list
    )

    context: dict[str, Any] = field(
        default_factory=dict
    )

    max_tokens: int = 512
    temperature: float = 0.7

    stream: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ModelResponse:
    success: bool
    text: str = ""

    model: str = ""
    backend: str = ""

    usage: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    error: str | None = None


class ModelBackend(ABC):

    @property
    @abstractmethod
    def capabilities(
        self,
    ) -> ModelCapabilities:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise NotImplementedError

    def embed(
        self,
        text: str,
    ) -> list[float]:
        raise NotImplementedError(
            "Embeddings are not supported by "
            f"{self.capabilities.name}."
        )

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def health(self) -> dict:
        return {
            "healthy": True,
            "model":
                self.capabilities.name,
            "backend":
                self.capabilities.backend,
        }
