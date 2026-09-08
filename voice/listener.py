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
        on_activate: Callable[[], None] | None = None,
    ) -> None:

        self.config = config or VoiceConfig()
        self.on_command = on_command
        self.on_activate = on_activate

        self.stt = WindowsSTT(
            self.config.listening_timeout_seconds
        )
        self.tts = WindowsTTS()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _speak(self, text: str) -> None:
        if self.config.tts_enabled:
            self.tts.speak(text)

    def process(
        self,
        text: str,
    ) -> dict:

        normalized = text.strip().lower()

        for phrase in self.config.accepted_wake_phrases():

            if normalized == phrase:

                if self.on_activate:
                    self.on_activate()

                self._speak(
                    self.config.response_phrase
                )

                return {
                    "activated": True,
                    "command": "",
                }

            prefix = phrase + " "

            if normalized.startswith(prefix):

                command = text[
                    len(prefix):
                ].strip()

                if self.on_activate:
                    self.on_activate()

                result = (
                    self.on_command(command)
                    if self.on_command
                    else None
                )

                if result:
                    self._speak(result)

                return {
                    "activated": True,
                    "command": command,
                    "result": result,
                }

        return {
            "activated": False,
            "command": "",
        }

    def listen_forever(
        self,
        background: bool = True,
    ):

        self._stop.clear()

        def loop():

            while not self._stop.is_set():

                try:
                    result = self.stt.listen_once()

                    if result.get("success"):
                        self.process(
                            result.get("text", "")
                        )

                except Exception:
                    # Voice daemon must not crash Gene.
                    pass

                time.sleep(0.15)

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

    @property
    def running(self) -> bool:
        return bool(
            self._thread
            and self._thread.is_alive()
        )