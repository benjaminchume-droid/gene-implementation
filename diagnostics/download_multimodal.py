from __future__ import annotations

import time
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = [
    {
        "repo": "google/siglip-base-patch16-224",
        "target": "gene/models/vision/siglip-base",
    },
    {
        "repo": "openai/whisper-base",
        "target": "gene/models/audio/whisper-base",
    },
    {
        "repo": "microsoft/speecht5_tts",
        "target": "gene/models/audio/speecht5-tts",
    },
]


def download(
    repo: str,
    target: str,
    retries: int = 20,
) -> None:

    Path(target).mkdir(
        parents=True,
        exist_ok=True,
    )

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            print()
            print(
                "=" * 72
            )
            print(
                f"MODEL: {repo}"
            )
            print(
                f"TARGET: {target}"
            )
            print(
                f"ATTEMPT: {attempt}/{retries}"
            )
            print(
                "=" * 72
            )

            snapshot_download(
                repo_id=repo,
                local_dir=target,
            )

            print(
                f"COMPLETED: {repo}"
            )

            return

        except KeyboardInterrupt:
            raise

        except Exception as exc:

            print(
                f"DOWNLOAD INTERRUPTED: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt >= retries:
                raise

            delay = min(
                60,
                2 ** min(
                    attempt,
                    6,
                ),
            )

            print(
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)


def main():

    for model in MODELS:

        download(
            repo=model["repo"],
            target=model["target"],
        )

    print()
    print(
        "ALL MODEL DOWNLOADS COMPLETE"
    )


if __name__ == "__main__":
    main()
