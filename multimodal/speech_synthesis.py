from __future__ import annotations

import json
from pathlib import Path

import torch
import soundfile as sf

from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

from .interfaces import (
    SpeechResult,
    SpeechSynthesizer,
)


class SpeechT5Synthesizer(
    SpeechSynthesizer
):

    def __init__(
        self,
        model_path: str = (
            "gene/models/audio/"
            "speecht5-tts"
        ),
        vocoder_path: str = (
            "gene/models/audio/"
            "speecht5-hifigan"
        ),
        xvector_path: str = (
            "gene/models/audio/"
            "speecht5-xvectors"
        ),
        speaker_index: int = 7306,
        device: str | None = None,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        self.vocoder_path = Path(
            vocoder_path
        )

        self.xvector_path = Path(
            xvector_path
        )

        self.speaker_index = (
            speaker_index
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
        self.vocoder = None
        self.speaker_embedding = None

    def _load_xvector(
        self,
    ) -> torch.Tensor:

        # Prefer a locally cached Hugging Face
        # dataset artifact. The exact file layout
        # can differ between dataset revisions.
        metadata_files = list(
            self.xvector_path.rglob(
                "*.json"
            )
        )

        candidates = list(
            self.xvector_path.rglob(
                "*.bin"
            )
        )

        if candidates:

            try:

                embedding_data = torch.load(
                    candidates[0],
                    map_location="cpu",
                )

                if isinstance(
                    embedding_data,
                    dict,
                ):
                    vectors = (
                        embedding_data.get(
                            "xvector"
                        )
                        or embedding_data.get(
                            "embeddings"
                        )
                    )

                    if vectors is not None:
                        tensor = torch.tensor(
                            vectors
                        )

                        return tensor[
                            self.speaker_index
                        ].unsqueeze(0)

                tensor = torch.as_tensor(
                    embedding_data
                )

                if tensor.ndim >= 2:
                    return tensor[
                        self.speaker_index
                    ].unsqueeze(0)

            except Exception:
                pass

        # Fallback to an explicitly persisted local
        # x-vector file.
        explicit = (
            self.xvector_path
            / "speaker_embedding.pt"
        )

        if explicit.exists():

            tensor = torch.load(
                explicit,
                map_location="cpu",
            )

            return torch.as_tensor(
                tensor
            ).unsqueeze(0) if tensor.ndim == 1 else tensor

        raise FileNotFoundError(
            "No usable SpeechT5 speaker embedding "
            f"was found under {self.xvector_path}. "
            "Create speaker_embedding.pt or materialize "
            "the CMU-Arctic x-vector dataset."
        )

    def load(self) -> None:

        self.processor = (
            SpeechT5Processor.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        )

        self.model = (
            SpeechT5ForTextToSpeech
            .from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        )

        self.vocoder = (
            SpeechT5HifiGan
            .from_pretrained(
                str(self.vocoder_path),
                local_files_only=True,
            )
        )

        self.model.to(
            self.device
        )

        self.vocoder.to(
            self.device
        )

        self.model.eval()
        self.vocoder.eval()

        self.speaker_embedding = (
            self._load_xvector()
            .to(self.device)
        )

    def synthesize(
        self,
        text: str,
        output_path: str,
    ) -> SpeechResult:

        try:

            if not text.strip():
                raise ValueError(
                    "text must not be empty."
                )

            if self.model is None:
                self.load()

            inputs = self.processor(
                text=text,
                return_tensors="pt",
            )

            input_ids = (
                inputs["input_ids"]
                .to(self.device)
            )

            with torch.no_grad():

                speech = (
                    self.model
                    .generate_speech(
                        input_ids,
                        self.speaker_embedding,
                        vocoder=self.vocoder,
                    )
                )

            audio = (
                speech.detach()
                .cpu()
                .numpy()
            )

            output = Path(
                output_path
            )

            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            sf.write(
                str(output),
                audio,
                samplerate=16000,
            )

            duration = (
                len(audio)
                / 16000
            )

            return SpeechResult(
                success=True,
                audio_path=str(
                    output
                ),
                sample_rate=16000,
                duration_seconds=duration,
                metadata={
                    "device":
                        self.device,
                    "model":
                        "speecht5",
                },
            )

        except Exception as exc:

            return SpeechResult(
                success=False,
                error=str(exc),
            )

    def unload(self) -> None:

        self.model = None
        self.vocoder = None
        self.processor = None
        self.speaker_embedding = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
