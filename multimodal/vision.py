from __future__ import annotations

from pathlib import Path

import torch

from PIL import Image
from transformers import (
    SiglipImageProcessorPil,
    SiglipVisionModel,
)

from .interfaces import (
    VisionEncoder,
    VisionResult,
)


class SigLIPVisionEncoder(VisionEncoder):

    def __init__(
        self,
        model_path: str = (
            "gene/models/vision/"
            "siglip-base"
        ),
        device: str | None = None,
    ) -> None:

        self.model_path = Path(model_path)

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
            SiglipImageProcessorPil.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        )

        self.model = (
            SiglipVisionModel.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        )

        self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        image,
    ) -> VisionResult:

        try:

            if self.model is None:
                self.load()

            if isinstance(
                image,
                (str, Path),
            ):

                image = Image.open(
                    image
                ).convert("RGB")

            elif not isinstance(
                image,
                Image.Image,
            ):

                raise TypeError(
                    "image must be a PIL image "
                    "or a file path."
                )

            inputs = self.processor(
                images=image,
                return_tensors="pt",
            )

            pixel_values = (
                inputs["pixel_values"]
                .to(self.device)
            )

            with torch.no_grad():

                outputs = self.model(
                    pixel_values=pixel_values
                )

            embedding = (
                outputs.pooler_output
            )

            if embedding is None:

                embedding = (
                    outputs.last_hidden_state
                    .mean(dim=1)
                )

            embedding = (
                embedding.detach()
                .cpu()
            )

            return VisionResult(
                success=True,
                embedding=embedding,
                shape=tuple(
                    embedding.shape
                ),
                metadata={
                    "model":
                        "siglip-vision",
                    "device":
                        self.device,
                    "embedding_dimension":
                        embedding.shape[-1],
                },
            )

        except Exception as exc:

            return VisionResult(
                success=False,
                error=str(exc),
            )

    def unload(self) -> None:

        self.model = None
        self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
