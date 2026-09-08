from __future__ import annotations

import subprocess


class WindowsSTT:

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds

    def listen_once(self) -> dict:
        script = f"""
Add-Type -AssemblyName System.Speech

$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine

$recognizer.SetInputToDefaultAudioDevice()

$grammar = New-Object System.Speech.Recognition.DictationGrammar
$recognizer.LoadGrammar($grammar)

$result = $recognizer.Recognize(
    [TimeSpan]::FromSeconds({self.timeout_seconds})
)

if ($null -eq $result) {{
    Write-Output ""
}} else {{
    Write-Output $result.Text
}}

$recognizer.Dispose()
"""

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
            timeout=self.timeout_seconds + 5,
        )

        return {
            "success": result.returncode == 0,
            "text": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
