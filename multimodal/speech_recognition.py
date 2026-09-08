from __future__ import annotations

from pathlib import Path

import torch
import soundfile as sf

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

from .interfaces import (
    SpeechRecognizer,
    TranscriptionResult,
)


class WhisperSpeechRecognizer(
    SpeechRecognizer
):

    def __init__(
        self,
        model_path: str = (
            "gene/models/audio/"
            "whisper-base"
        ),
        device: str | None = None,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.processor = None
        self.model = None

    def load(self) -> None:

        self.processor = (
            AutoProcessor.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        )

        self.model = (
            AutoModelForSpeechSeq2Seq
            .from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def transcribe(
        self,
        audio_path: str,
    ) -> TranscriptionResult:

        try:

            if self.model is None:
                self.load()

            path = Path(
                audio_path
            )

            if not path.exists():
                raise FileNotFoundError(
                    str(path)
                )

            audio, sample_rate = (
                sf.read(
                    str(path)
                )
            )

            if len(audio.shape) > 1:
                audio = audio.mean(
                    axis=1
                )

            target_rate = 16000

            if sample_rate != target_rate:

                import numpy as np

                duration = (
                    len(audio)
                    / sample_rate
                )

                new_length = max(
                    1,
                    int(
                        duration
                        * target_rate
                    ),
                )

                old_positions = np.linspace(
                    0.0,
                    duration,
                    num=len(audio),
                    endpoint=False,
                )

                new_positions = np.linspace(
                    0.0,
                    duration,
                    num=new_length,
                    endpoint=False,
                )

                audio = np.interp(
                    new_positions,
                    old_positions,
                    audio,
                )

                sample_rate = target_rate

            inputs = self.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )

            inputs = {
                key:
                    value.to(
                        self.device
                    )
                for key, value
                in inputs.items()
                if hasattr(value, "to")
            }

            with torch.no_grad():

                generated_ids = (
                    self.model.generate(
                        **inputs
                    )
                )

            text = (
                self.processor
                .batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
                .strip()
            )

            return TranscriptionResult(
                success=True,
                text=text,
                metadata={
                    "model":
                        "whisper",
                    "sample_rate":
                        sample_rate,
                    "device":
                        self.device,
                },
            )

        except Exception as exc:

            return TranscriptionResult(
                success=False,
                error=str(exc),
            )

    def unload(self) -> None:

        self.model = None
        self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
