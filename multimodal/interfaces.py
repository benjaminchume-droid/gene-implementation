from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VisionResult:

    success: bool
    embedding: Any = None
    shape: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )
    error: str | None = None


@dataclass
class TranscriptionResult:

    success: bool
    text: str = ""
    language: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )
    error: str | None = None


@dataclass
class SpeechResult:

    success: bool
    audio_path: str | None = None
    sample_rate: int | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )
    error: str | None = None


class VisionEncoder(ABC):

    @abstractmethod
    def encode(
        self,
        image,
    ) -> VisionResult:
        raise NotImplementedError


class SpeechRecognizer(ABC):

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
    ) -> TranscriptionResult:
        raise NotImplementedError


class SpeechSynthesizer(ABC):

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str,
    ) -> SpeechResult:
        raise NotImplementedError
