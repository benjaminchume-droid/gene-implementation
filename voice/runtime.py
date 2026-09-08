from __future__ import annotations

from gene.voice.daemon import VoiceDaemon
from gene.voice.hotkey import HotkeyService


class VoiceRuntime:

    def __init__(
        self,
        config,
        on_command=None,
    ) -> None:

        self.config = config

        self.daemon = VoiceDaemon(
            config=config,
            on_command=on_command,
        )

        self.hotkey = None

        if config.hotkey_enabled:
            self.hotkey = HotkeyService(
                config.hotkey,
                self.daemon.activate,
            )

    def start(self) -> dict:

        if self.config.stt_enabled:
            self.daemon.start()

        if self.hotkey:
            self.hotkey.register()

        return self.status()

    def stop(self) -> dict:

        if self.hotkey:
            self.hotkey.unregister()

        self.daemon.stop()

        return self.status()

    def status(self) -> dict:

        return {
            "running":
                self.daemon.running,
            "active":
                self.daemon.active,
            "wake_phrase":
                self.config.wake_phrase,
            "custom_wake_phrases":
                self.config.custom_phrases,
            "hotkey":
                self.config.hotkey
                if self.config.hotkey_enabled
                else None,
            "hotkey_registered":
                self.hotkey.registered
                if self.hotkey
                else False,
            "stt_enabled":
                self.config.stt_enabled,
            "tts_enabled":
                self.config.tts_enabled,
        }
