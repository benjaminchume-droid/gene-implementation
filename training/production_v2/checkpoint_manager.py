from __future__ import annotations

import re
import shutil

from pathlib import Path


_STEP_PATTERN = re.compile(
    r"(?:weights|resume)-step-(\d+)\.pt$"
)


class CheckpointManager:

    def __init__(
        self,
        directory: str,
        *,
        keep_weights: int = 3,
        keep_resume: int = 1,
        minimum_free_gb: float = 6.0,
    ) -> None:

        self.directory = Path(
            directory
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.keep_weights = (
            max(1, keep_weights)
        )

        self.keep_resume = (
            max(1, keep_resume)
        )

        self.minimum_free_bytes = int(
            minimum_free_gb * 1024**3
        )

    def free_bytes(self) -> int:

        return shutil.disk_usage(
            self.directory
        ).free

    def free_gb(self) -> float:

        return (
            self.free_bytes()
            / 1024**3
        )

    def assert_space(
        self,
        estimated_bytes: int,
    ) -> None:

        required = (
            estimated_bytes
            + self.minimum_free_bytes
        )

        available = (
            self.free_bytes()
        )

        if available < required:

            raise RuntimeError(
                "Checkpoint denied because "
                "disk headroom is too low. "
                f"available={available / 1024**3:.2f}GB "
                f"required={(required) / 1024**3:.2f}GB "
                f"safety={self.minimum_free_bytes / 1024**3:.2f}GB"
            )

    def _files(
        self,
        prefix: str,
    ) -> list[tuple[int, Path]]:

        result = []

        for path in self.directory.glob(
            f"{prefix}-step-*.pt"
        ):

            match = _STEP_PATTERN.match(
                path.name
            )

            if not match:
                continue

            try:
                step = int(
                    match.group(1)
                )
            except ValueError:
                continue

            result.append(
                (
                    step,
                    path,
                )
            )

        return sorted(
            result,
            key=lambda item: item[0],
            reverse=True,
        )

    def prune(
        self,
    ) -> list[str]:

        deleted = []

        for (
            _,
            path,
        ) in self._files(
            "weights"
        )[self.keep_weights:]:

            try:

                path.unlink()

                deleted.append(
                    str(path)
                )

            except FileNotFoundError:
                pass

        for (
            _,
            path,
        ) in self._files(
            "resume"
        )[self.keep_resume:]:

            try:

                path.unlink()

                deleted.append(
                    str(path)
                )

            except FileNotFoundError:
                pass

        for path in self.directory.glob(
            "*.tmp"
        ):

            try:

                path.unlink()

                deleted.append(
                    str(path)
                )

            except FileNotFoundError:
                pass

        return deleted

    def status(self) -> dict:

        weights = [
            str(path)
            for _,
            path in self._files(
                "weights"
            )
        ]

        resumes = [
            str(path)
            for _,
            path in self._files(
                "resume"
            )
        ]

        return {
            "free_gb":
                self.free_gb(),
            "weights":
                weights,
            "resume":
                resumes,
            "keep_weights":
                self.keep_weights,
            "keep_resume":
                self.keep_resume,
            "minimum_free_gb":
                self.minimum_free_bytes
                / 1024**3,
        }
