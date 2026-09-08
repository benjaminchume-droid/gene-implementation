from gene.voice.config import VoiceConfig
from gene.voice.runtime import VoiceRuntime


def test_voice_process_wake_phrase():

    commands = []

    config = VoiceConfig(
        wake_phrase="hello gene",
        tts_enabled=False,
        stt_enabled=False,
    )

    runtime = VoiceRuntime(
        config=config,
        on_command=commands.append,
    )

    result = runtime.daemon.process_text(
        "hello gene"
    )

    assert result["activated"] is True
    assert result["command"] == ""


def test_voice_process_command():

    commands = []

    config = VoiceConfig(
        wake_phrase="hello gene",
        tts_enabled=False,
        stt_enabled=False,
    )

    def handler(command):
        commands.append(command)
        return "done"

    runtime = VoiceRuntime(
        config=config,
        on_command=handler,
    )

    result = runtime.daemon.process_text(
        "hello gene list the files"
    )

    assert result["activated"] is True
    assert result["command"] == "list the files"
    assert commands == [
        "list the files"
    ]


def test_custom_wake_phrase():

    config = VoiceConfig(
        wake_phrase="hello gene",
        custom_phrases=[
            "wake up gene"
        ],
        tts_enabled=False,
        stt_enabled=False,
    )

    runtime = VoiceRuntime(
        config=config
    )

    result = runtime.daemon.process_text(
        "wake up gene"
    )

    assert result["activated"] is True
