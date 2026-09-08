from __future__ import annotations

from pathlib import Path

from gene.multimodal import (
    MultimodalRuntime,
)


class AgentMultimodalBridge:

    def __init__(
        self,
        runtime: MultimodalRuntime,
    ) -> None:

        self.runtime = runtime

    def observe_image(
        self,
        image,
    ) -> dict:

        result = (
            self.runtime.inspect_image(
                image
            )
        )

        if not result.success:

            return {
                "success": False,
                "error": result.error,
            }

        return {
            "success": True,
            "modality": "vision",
            "embedding":
                result.embedding,
            "shape":
                result.shape,
            "metadata":
                result.metadata,
        }

    def transcribe_audio(
        self,
        audio_path: str,
    ) -> dict:

        result = (
            self.runtime.transcribe(
                audio_path
            )
        )

        if not result.success:

            return {
                "success": False,
                "error": result.error,
            }

        return {
            "success": True,
            "modality": "speech",
            "text":
                result.text,
            "metadata":
                result.metadata,
        }

    def speak(
        self,
        text: str,
        output_path: str,
    ) -> dict:

        result = (
            self.runtime.speak(
                text,
                output_path,
            )
        )

        if not result.success:

            return {
                "success": False,
                "error": result.error,
            }

        return {
            "success": True,
            "modality": "speech_output",
            "audio_path":
                result.audio_path,
            "sample_rate":
                result.sample_rate,
            "duration_seconds":
                result.duration_seconds,
            "metadata":
                result.metadata,
        }

    def observe_and_transcribe(
        self,
        image=None,
        audio_path: str | None = None,
    ) -> dict:

        result = {
            "visual": None,
            "speech": None,
        }

        if image is not None:
            result["visual"] = (
                self.observe_image(
                    image
                )
            )

        if audio_path is not None:
            result["speech"] = (
                self.transcribe_audio(
                    audio_path
                )
            )

        return result
