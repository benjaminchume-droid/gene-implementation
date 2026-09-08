from __future__ import annotations


class ComputerControl:

    def __init__(self) -> None:
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError(
                "Computer control requires pyautogui."
            ) from exc

        self.pyautogui = pyautogui

        # Safety setting: PyAutoGUI raises an exception if the mouse
        # reaches the configured fail-safe corner.
        self.pyautogui.FAILSAFE = True

    def position(self) -> dict:
        x, y = self.pyautogui.position()

        return {
            "success": True,
            "x": x,
            "y": y,
        }

    def move(
        self,
        x: int,
        y: int,
        duration: float = 0.15,
    ) -> dict:

        self.pyautogui.moveTo(
            x,
            y,
            duration=duration,
        )

        return self.position()

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
    ) -> dict:

        if x is not None and y is not None:
            self.pyautogui.moveTo(
                x,
                y,
                duration=0.1,
            )

        self.pyautogui.click(
            button=button,
            clicks=clicks,
        )

        return {
            "success": True,
            "action": "click",
            "button": button,
            "clicks": clicks,
        }

    def type_text(
        self,
        text: str,
        interval: float = 0.01,
    ) -> dict:

        self.pyautogui.write(
            text,
            interval=interval,
        )

        return {
            "success": True,
            "action": "type",
        }

    def press(
        self,
        key: str,
    ) -> dict:

        self.pyautogui.press(key)

        return {
            "success": True,
            "action": "press",
            "key": key,
        }

    def hotkey(
        self,
        *keys: str,
    ) -> dict:

        self.pyautogui.hotkey(*keys)

        return {
            "success": True,
            "action": "hotkey",
            "keys": list(keys),
        }

    def scroll(
        self,
        amount: int,
    ) -> dict:

        self.pyautogui.scroll(amount)

        return {
            "success": True,
            "action": "scroll",
            "amount": amount,
        }
