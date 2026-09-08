from __future__ import annotations

from typing import Callable

from .config import VoiceConfig
from .stt import WindowsSTT
from .tts import WindowsTTS


class WakeController:

    def __init__(
        self,
        config: VoiceConfig | None = None,
        on_command: Callable[[str], str | None] | None = None,
    ):
        self.config = config or VoiceConfig()
        self.on_command = on_command

        self.stt = WindowsSTT(
            self.config.listening_timeout_seconds
        )
        self.tts = WindowsTTS()

    def speak(self, text: str) -> None:
        if self.config.tts_enabled:
            self.tts.speak(text)

    def process_text(self, text: str) -> dict:
        normalized = text.strip().lower()

        for phrase in self.config.accepted_wake_phrases():

            if normalized == phrase:
                self.speak(
                    self.config.response_phrase
                )

                return {
                    "activated": True,
                    "command": "",
                    "result": None,
                }

            prefix = phrase + " "

            if normalized.startswith(prefix):
                command = text[len(prefix):].strip()

                result = (
                    self.on_command(command)
                    if self.on_command
                    else None
                )

                if result:
                    self.speak(result)

                return {
                    "activated": True,
                    "command": command,
                    "result": result,
                }

        return {
            "activated": False,
            "command": "",
            "result": None,
        }

    def listen_once(self) -> dict:
        result = self.stt.listen_once()

        if not result["success"]:
            return result

        return {
            **result,
            **self.process_text(result["text"]),
        }
