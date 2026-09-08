from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VisionObservation:
    image_path: str
    provider: str
    description: str | None = None
    structured: dict | None = None


class VisionProvider:
    """
    Provider-neutral screen understanding interface.

    A real vision-language model can be attached later without
    changing the desktop agent API.
    """

    name = "base"

    def analyze(
        self,
        image_path: str,
        instruction: str = "",
    ) -> VisionObservation:

        raise NotImplementedError


class UnavailableVisionProvider(VisionProvider):

    name = "unavailable"

    def analyze(
        self,
        image_path: str,
        instruction: str = "",
    ) -> VisionObservation:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                image_path
            )

        return VisionObservation(
            image_path=str(path.resolve()),
            provider=self.name,
            description=None,
            structured=None,
        )


class VisionService:

    def __init__(
        self,
        provider: VisionProvider | None = None,
    ) -> None:

        self.provider = (
            provider
            or UnavailableVisionProvider()
        )

    def analyze(
        self,
        image_path: str,
        instruction: str = "",
    ) -> dict:

        observation = self.provider.analyze(
            image_path,
            instruction,
        )

        return {
            "success": True,
            "provider":
                observation.provider,
            "image_path":
                observation.image_path,
            "description":
                observation.description,
            "structured":
                observation.structured,
        }