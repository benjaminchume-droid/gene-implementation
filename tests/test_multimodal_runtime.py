from pathlib import Path

from PIL import Image

from gene.multimodal import (
    MultimodalRuntime,
    SigLIPVisionEncoder,
    WhisperSpeechRecognizer,
    SpeechT5Synthesizer,
)


def test_provider_construction():

    vision = SigLIPVisionEncoder(
        model_path=(
            "gene/models/vision/"
            "siglip-base"
        )
    )

    stt = WhisperSpeechRecognizer(
        model_path=(
            "gene/models/audio/"
            "whisper-base"
        )
    )

    tts = SpeechT5Synthesizer(
        model_path=(
            "gene/models/audio/"
            "speecht5-tts"
        ),
        vocoder_path=(
            "gene/models/audio/"
            "speecht5-hifigan"
        ),
    )

    runtime = MultimodalRuntime(
        vision=vision,
        speech_recognizer=stt,
        speech_synthesizer=tts,
    )

    status = runtime.status()

    assert status["vision"] is True
    assert status[
        "speech_recognition"
    ] is True
    assert status[
        "speech_synthesis"
    ] is True


def test_image_input_exists():

    image = Image.new(
        "RGB",
        (224, 224),
        "white",
    )

    assert image.size == (
        224,
        224,
    )
