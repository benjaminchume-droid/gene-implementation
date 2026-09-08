from __future__ import annotations

from pathlib import Path

from .base import VisionProvider
from ..models import (
    ScreenAnalysisRequest,
    VisionObservation,
)


class UnavailableVisionProvider(
    VisionProvider
):

    name = "unavailable"

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

        return VisionObservation(
            success=True,
            provider=self.name,
            image_path=str(
                path.resolve()
            ),
            description=None,
            text=None,
            regions=[],
            structured={
                "instruction":
                    request.instruction,
                "status":
                    "image_available_but_no_vision_backend",
            },
        )
