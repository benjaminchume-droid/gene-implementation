from PIL import Image

from gene.multimodal import (
    MultimodalRuntime,
    SigLIPVisionEncoder,
    WhisperSpeechRecognizer,
    SpeechT5Synthesizer,
)

from gene.agent.multimodal import (
    AgentMultimodalBridge,
)


def test_agent_multimodal_bridge():

    runtime = MultimodalRuntime(
        vision=SigLIPVisionEncoder(),
        speech_recognizer=(
            WhisperSpeechRecognizer()
        ),
        speech_synthesizer=(
            SpeechT5Synthesizer()
        ),
    )

    bridge = AgentMultimodalBridge(
        runtime
    )

    image = Image.new(
        "RGB",
        (224, 224),
        "white",
    )

    visual = bridge.observe_image(
        image
    )

    assert visual["success"] is True
    assert visual["shape"] == (
        1,
        768,
    )

    speech = bridge.transcribe_audio(
        "gene/data/training/pilot/"
        "whisper_smoke.wav"
    )

    assert speech["success"] is True

    output = bridge.speak(
        "Gene multimodal test.",
        "gene/data/training/pilot/"
        "bridge_tts.wav",
    )

    assert output["success"] is True
    assert output["audio_path"]
