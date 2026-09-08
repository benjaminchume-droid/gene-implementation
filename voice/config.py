from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VoiceConfig:
    wake_phrase: str = "hello gene"
    hotkey: str = "x"
    response_phrase: str = "Hi, can I help you?"
    language: str = "en-US"

    tts_enabled: bool = True
    stt_enabled: bool = True
    hotkey_enabled: bool = True
    wake_phrase_enabled: bool = True

    activation_timeout_seconds: int = 8
    listening_timeout_seconds: int = 10

    custom_phrases: list[str] = field(default_factory=list)

    def accepted_wake_phrases(self) -> list[str]:
        return [
            self.wake_phrase.lower(),
            *[
                phrase.lower()
                for phrase in self.custom_phrases
            ],
        ]
