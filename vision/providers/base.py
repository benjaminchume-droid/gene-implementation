from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import (
    ScreenAnalysisRequest,
    VisionObservation,
)


class VisionProvider(ABC):

    name = "base"

    @abstractmethod
    def analyze(
        self,
        request: ScreenAnalysisRequest,
    ) -> VisionObservation:
        raise NotImplementedError
