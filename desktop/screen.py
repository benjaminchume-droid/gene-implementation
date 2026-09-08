from __future__ import annotations

from pathlib import Path


class ScreenService:
    """Desktop screen capture backend."""

    def __init__(self) -> None:
        try:
            import mss
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Screen capture requires mss and pillow."
            ) from exc

        self._mss = mss
        self._Image = Image

    def monitors(self) -> list[dict]:
        with self._mss.mss() as capture:
            return [
                {
                    "index": index,
                    "left": monitor["left"],
                    "top": monitor["top"],
                    "width": monitor["width"],
                    "height": monitor["height"],
                }
                for index, monitor in enumerate(capture.monitors)
            ]

    def capture(
        self,
        output: str = "gene/data/screens/latest.png",
        monitor: int = 1,
    ) -> dict:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._mss.mss() as capture:
            monitors = capture.monitors

            if monitor < 1 or monitor >= len(monitors):
                raise ValueError(f"Invalid monitor: {monitor}")

            shot = capture.grab(monitors[monitor])

            image = self._Image.frombytes(
                "RGB",
                shot.size,
                shot.rgb,
            )

            image.save(target)

        return {
            "success": True,
            "path": str(target.resolve()),
            "width": image.width,
            "height": image.height,
            "monitor": monitor,
        }

    def capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        output: str = "gene/data/screens/region.png",
    ) -> dict:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)

        region = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

        with self._mss.mss() as capture:
            shot = capture.grab(region)

            image = self._Image.frombytes(
                "RGB",
                shot.size,
                shot.rgb,
            )

            image.save(target)

        return {
            "success": True,
            "path": str(target.resolve()),
            "width": image.width,
            "height": image.height,
        }
