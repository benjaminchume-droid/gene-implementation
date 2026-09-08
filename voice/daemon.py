from __future__ import annotations

import threading
import time
from typing import Callable

from .config import VoiceConfig
from .stt import WindowsSTT
from .tts import WindowsTTS


class VoiceDaemon:

    def __init__(
        self,
        config: VoiceConfig | None = None,
        on_command: Callable[[str], str | None] | None = None,
    ) -> None:

        self.config = config or VoiceConfig()

        self.on_command = on_command

        self.stt = WindowsSTT(
            self.config.listening_timeout_seconds
        )

        self.tts = WindowsTTS()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._active = False

    def speak(self, text: str) -> dict:

        if not self.config.tts_enabled:
            return {
                "success": True,
                "disabled": True,
            }

        return self.tts.speak(text)

    def activate(self) -> dict:

        self._active = True

        result = self.speak(
            self.config.response_phrase
        )

        return {
            "success": True,
            "activated": True,
            "voice": result,
        }

    def process_text(
        self,
        text: str,
    ) -> dict:

        normalized = text.strip().lower()

        for phrase in (
            self.config
            .accepted_wake_phrases()
        ):

            if normalized == phrase:

                self.activate()

                return {
                    "activated": True,
                    "command": "",
                }

            prefix = phrase + " "

            if normalized.startswith(
                prefix
            ):

                command = text[
                    len(prefix):
                ].strip()

                self._active = True

                result = (
                    self.on_command(
                        command
                    )
                    if self.on_command
                    else None
                )

                if result:
                    self.speak(result)

                self._active = False

                return {
                    "activated": True,
                    "command": command,
                    "result": result,
                }

        if self._active:

            command = text.strip()

            if command:

                result = (
                    self.on_command(
                        command
                    )
                    if self.on_command
                    else None
                )

                if result:
                    self.speak(result)

                self._active = False

                return {
                    "activated": True,
                    "command": command,
                    "result": result,
                }

        return {
            "activated": False,
            "command": "",
        }

    def listen_once(self) -> dict:

        result = self.stt.listen_once()

        if not result.get("success"):
            return result

        processed = self.process_text(
            result.get("text", "")
        )

        return {
            **result,
            **processed,
        }

    def start(
        self,
        background: bool = True,
    ) -> None:

        if self.running:
            return

        self._stop.clear()

        def loop():

            while not self._stop.is_set():

                try:
                    self.listen_once()

                except Exception:
                    # Voice errors must never crash Gene.
                    pass

                self._stop.wait(0.15)

        self._thread = threading.Thread(
            target=loop,
            name="gene-voice-daemon",
            daemon=True,
        )

        self._thread.start()

        if not background:
            self._thread.join()

    def stop(self) -> None:

        self._stop.set()

        if self._thread:
            self._thread.join(
                timeout=3
            )

        self._thread = None
        self._active = False

    @property
    def running(self) -> bool:
        return bool(
            self._thread
            and self._thread.is_alive()
        )

    @property
    def active(self) -> bool:
        return self._active
