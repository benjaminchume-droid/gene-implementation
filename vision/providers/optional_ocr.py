from __future__ import annotations

from pathlib import Path

from .base import VisionProvider
from ..models import (
    ScreenAnalysisRequest,
    VisionObservation,
)


class OptionalOCRProvider(
    VisionProvider
):

    name = "optional_ocr"

    def __init__(self) -> None:

        try:
            import pytesseract
            from PIL import Image

        except ImportError:
            self._pytesseract = None
            self._Image = None
            return

        self._pytesseract = pytesseract
        self._Image = Image

    @property
    def available(self) -> bool:
        return (
            self._pytesseract is not None
            and self._Image is not None
        )

    def analyze(
        self,
        request: ScreenAnalysisRequest,
    ) -> VisionObservation:

        path = Path(
            request.image_path
        )

        if not path.exists():
            return VisionObservation(
                success=False,
                provider=self.name,
                image_path=str(path),
                error="Image does not exist.",
            )

        if not self.available:
            return VisionObservation(
                success=False,
                provider=self.name,
                image_path=str(
                    path.resolve()
                ),
                error=(
                    "OCR provider is unavailable. "
                    "Install pytesseract and a compatible "
                    "Tesseract OCR engine."
                ),
            )

        image = self._Image.open(path)

        text = self._pytesseract.image_to_string(
            image
        )

        return VisionObservation(
            success=True,
            provider=self.name,
            image_path=str(
                path.resolve()
            ),
            text=text,
            description=(
                "OCR text extracted from the image."
            ),
            structured={
                "instruction":
                    request.instruction,
                "text_length":
                    len(text),
            },
        )
