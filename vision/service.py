from __future__ import annotations

from pathlib import Path

from .models import (
    ScreenAnalysisRequest,
)
from .providers import (
    OptionalOCRProvider,
    UnavailableVisionProvider,
)
from .providers.base import (
    VisionProvider,
)


class VisionService:

    def __init__(
        self,
        provider: VisionProvider | None = None,
    ) -> None:

        self.provider = (
            provider
            or self._select_default_provider()
        )

    @staticmethod
    def _select_default_provider():

        provider = OptionalOCRProvider()

        if provider.available:
            return provider

        return UnavailableVisionProvider()

    def set_provider(
        self,
        provider: VisionProvider,
    ) -> None:

        self.provider = provider

    def analyze(
        self,
        image_path: str,
        instruction: str = "",
        metadata: dict | None = None,
    ) -> dict:

        request = ScreenAnalysisRequest(
            image_path=image_path,
            instruction=instruction,
            metadata=metadata or {},
        )

        observation = self.provider.analyze(
            request
        )

        return {
            "success":
                observation.success,
            "provider":
                observation.provider,
            "image_path":
                observation.image_path,
            "description":
                observation.description,
            "text":
                observation.text,
            "regions":
                observation.regions,
            "structured":
                observation.structured,
            "error":
                observation.error,
            "instruction":
                instruction,
        }

    def analyze_screen(
        self,
        desktop_agent,
        instruction: str = "",
        output: str = (
            "gene/data/vision/latest.png"
        ),
    ) -> dict:

        capture = desktop_agent.inspect_screen(
            output
        )

        if not capture.get("success"):
            return capture

        return self.analyze(
            image_path=capture["path"],
            instruction=instruction,
            metadata={
                "capture": capture,
            },
        )

    def status(self) -> dict:

        return {
            "provider":
                self.provider.name,
            "available":
                True,
        }
