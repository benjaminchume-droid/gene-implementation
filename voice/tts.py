from __future__ import annotations

import base64
import subprocess


class WindowsTTS:

    def speak(self, text: str) -> dict:
        encoded = base64.b64encode(
            text.encode("utf-8")
        ).decode("ascii")

        script = (
            "$bytes = [Convert]::FromBase64String('"
            + encoded
            + "'); "
            "$text = [Text.Encoding]::UTF8.GetString($bytes); "
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object "
            "System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Speak($text); "
            "$speaker.Dispose();"
        )

        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
        )

        return {
            "success": result.returncode == 0,
            "stderr": result.stderr.strip(),
        }
