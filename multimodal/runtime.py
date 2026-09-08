from __future__ import annotations

from .interfaces import (
    SpeechRecognizer,
    SpeechSynthesizer,
    VisionEncoder,
    SpeechResult,
    TranscriptionResult,
    VisionResult,
)

from .projectors import (
    VisionProjector,
)


class MultimodalRuntime:

    def __init__(
        self,
        *,
        vision: VisionEncoder | None = None,
        vision_projector:
            VisionProjector | None = None,
        speech_recognizer:
            SpeechRecognizer | None = None,
        speech_synthesizer:
            SpeechSynthesizer | None = None,
    ) -> None:

        self.vision = vision

        self.vision_projector = (
            vision_projector
        )

        self.speech_recognizer = (
            speech_recognizer
        )

        self.speech_synthesizer = (
            speech_synthesizer
        )

    def inspect_image(
        self,
        image,
    ) -> VisionResult:

        if self.vision is None:
            return VisionResult(
                success=False,
                error=(
                    "No vision provider "
                    "is configured."
                ),
            )

        return self.vision.encode(
            image
        )

    def project_visual_embedding(
        self,
        embedding,
    ):

        if self.vision_projector is None:
            raise RuntimeError(
                "No vision projector "
                "is configured."
            )

        return self.vision_projector(
            embedding
        )

    def inspect_image_for_gene(
        self,
        image,
    ) -> dict:

        result = self.inspect_image(
            image
        )

        if not result.success:
            return {
                "success": False,
                "error":
                    result.error,
            }

        if self.vision_projector is None:
            return {
                "success": True,
                "embedding":
                    result.embedding,
                "embedding_shape":
                    result.shape,
                "projected":
                    False,
            }

        projected = (
            self.project_visual_embedding(
                result.embedding
            )
        )

        return {
            "success": True,
            "embedding":
                result.embedding,
            "embedding_shape":
                result.shape,
            "gene_representation":
                projected,
            "gene_representation_shape":
                tuple(
                    projected.shape
                ),
            "projected":
                True,
        }

    def transcribe(
        self,
        audio_path: str,
    ) -> TranscriptionResult:

        if self.speech_recognizer is None:
            return TranscriptionResult(
                success=False,
                error=(
                    "No speech-recognition "
                    "provider is configured."
                ),
            )

        return (
            self.speech_recognizer
            .transcribe(
                audio_path
            )
        )

    def speak(
        self,
        text: str,
        output_path: str,
    ) -> SpeechResult:

        if self.speech_synthesizer is None:
            return SpeechResult(
                success=False,
                error=(
                    "No speech-synthesis "
                    "provider is configured."
                ),
            )

        return (
            self.speech_synthesizer
            .synthesize(
                text,
                output_path,
            )
        )

    def status(self) -> dict:

        return {
            "vision":
                self.vision is not None,

            "vision_projector":
                self.vision_projector is not None,

            "speech_recognition":
                self.speech_recognizer is not None,

            "speech_synthesis":
                self.speech_synthesizer is not None,
        }
