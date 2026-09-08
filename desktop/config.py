from __future__ import annotations

import json
from pathlib import Path

from gene.voice.config import VoiceConfig
from .policy import (
    ConfirmationMode,
    DesktopPolicy,
    InternetLevel,
    PrivilegeLevel,
)


class DesktopConfiguration:

    def __init__(
        self,
        path: str = "gene/config/desktop.json",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def create_default(self) -> dict:
        data = {
            "voice": {
                "wake_phrase": "hello gene",
                "hotkey": "x",
                "response_phrase": "Hi, can I help you?",
                "language": "en-US",
                "tts_enabled": True,
                "stt_enabled": True,
                "hotkey_enabled": True,
                "wake_phrase_enabled": True,
                "activation_timeout_seconds": 8,
                "listening_timeout_seconds": 10,
                "custom_phrases": [],
            },
            "desktop_policy": {
                "privilege_level": "USER",
                "internet_level": "STANDARD",
                "confirmation_mode": "RISK_BASED",
            },
        }

        self.path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return data

    def load(self) -> dict:
        if not self.path.exists():
            return self.create_default()

        # utf-8-sig accepts normal UTF-8 and UTF-8 files
        # containing a BOM.
        raw = self.path.read_text(
            encoding="utf-8-sig"
        )

        if not raw.strip():
            return self.create_default()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid Gene desktop configuration: "
                f"{self.path}: {exc}"
            ) from exc

    def objects(self):
        data = self.load()

        voice_data = data.get("voice", {})

        voice = VoiceConfig(
            wake_phrase=voice_data.get(
                "wake_phrase",
                "hello gene",
            ),
            hotkey=voice_data.get(
                "hotkey",
                "x",
            ),
            response_phrase=voice_data.get(
                "response_phrase",
                "Hi, can I help you?",
            ),
            language=voice_data.get(
                "language",
                "en-US",
            ),
            tts_enabled=voice_data.get(
                "tts_enabled",
                True,
            ),
            stt_enabled=voice_data.get(
                "stt_enabled",
                True,
            ),
            hotkey_enabled=voice_data.get(
                "hotkey_enabled",
                True,
            ),
            wake_phrase_enabled=voice_data.get(
                "wake_phrase_enabled",
                True,
            ),
            activation_timeout_seconds=voice_data.get(
                "activation_timeout_seconds",
                8,
            ),
            listening_timeout_seconds=voice_data.get(
                "listening_timeout_seconds",
                10,
            ),
            custom_phrases=voice_data.get(
                "custom_phrases",
                [],
            ),
        )

        policy_data = data.get(
            "desktop_policy",
            {},
        )

        try:
            privilege_level = PrivilegeLevel[
                policy_data.get(
                    "privilege_level",
                    "USER",
                )
            ]

            internet_level = InternetLevel[
                policy_data.get(
                    "internet_level",
                    "STANDARD",
                )
            ]

            confirmation_mode = ConfirmationMode[
                policy_data.get(
                    "confirmation_mode",
                    "RISK_BASED",
                )
            ]

        except KeyError as exc:
            raise ValueError(
                f"Invalid desktop policy value: {exc}"
            ) from exc

        policy = DesktopPolicy(
            privilege_level=privilege_level,
            internet_level=internet_level,
            confirmation_mode=confirmation_mode,
        )

        return voice, policy
