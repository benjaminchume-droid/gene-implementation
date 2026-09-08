from pathlib import Path

from PIL import Image

from gene.vision.models import ScreenAnalysisRequest
from gene.vision.providers.unavailable import (
    UnavailableVisionProvider,
)
from gene.vision.service import VisionService


def test_screen_analysis_request():

    request = ScreenAnalysisRequest(
        image_path="screen.png",
        instruction=(
            "Find whatever the user asks "
            "about in this image."
        ),
    )

    assert request.image_path == "screen.png"
    assert request.instruction


def test_unavailable_provider(tmp_path):

    image = tmp_path / "screen.png"

    Image.new(
        "RGB",
        (32, 32),
    ).save(image)

    provider = (
        UnavailableVisionProvider()
    )

    result = provider.analyze(
        ScreenAnalysisRequest(
            image_path=str(image),
            instruction=(
                "Describe the requested "
                "content."
            ),
        )
    )

    assert result.success is True
    assert result.image_path


def test_vision_service_with_image(tmp_path):

    image = tmp_path / "screen.png"

    Image.new(
        "RGB",
        (64, 64),
    ).save(image)

    service = VisionService(
        UnavailableVisionProvider()
    )

    result = service.analyze(
        image_path=str(image),
        instruction=(
            "Answer the user's screen "
            "question."
        ),
    )

    assert result["success"] is True
    assert result["instruction"] == (
        "Answer the user's screen question."
    )


def test_missing_image():

    service = VisionService(
        UnavailableVisionProvider()
    )

    result = service.analyze(
        image_path="missing-image.png",
        instruction="Anything the user requests.",
    )

    assert result["success"] is False
    assert result["error"]
