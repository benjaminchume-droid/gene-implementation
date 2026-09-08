from __future__ import annotations

from typing import Callable


class HotkeyService:

    def __init__(
        self,
        hotkey: str,
        callback: Callable[[], None],
    ) -> None:

        try:
            import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "Global hotkey support requires keyboard."
            ) from exc

        self.keyboard = keyboard
        self.hotkey = hotkey
        self.callback = callback

        self._registered = False

    def register(self) -> None:

        if self._registered:
            return

        self.keyboard.add_hotkey(
            self.hotkey,
            self.callback,
        )

        self._registered = True

    def unregister(self) -> None:

        if not self._registered:
            return

        try:
            self.keyboard.remove_hotkey(
                self.hotkey
            )
        finally:
            self._registered = False

    @property
    def registered(self) -> bool:
        return self._registered
