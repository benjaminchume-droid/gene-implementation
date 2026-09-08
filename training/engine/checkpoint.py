from __future__ import annotations

from pathlib import Path
from typing import Any


class TrainingCheckpoint:

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch: int,
        step: int,
        best_validation_loss: float | None,
        config,
    ) -> str:

        import torch

        path = (
            self.directory
            / f"checkpoint-{step}.pt"
        )

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "scaler": (
                scaler.state_dict()
                if scaler is not None
                else None
            ),
            "epoch": epoch,
            "step": step,
            "best_validation_loss":
                best_validation_loss,
            "config": config.to_dict(),
        }

        torch.save(payload, path)

        latest = self.directory / "latest.pt"
        torch.save(payload, latest)

        return str(path)

    def load(
        self,
        path: str,
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        map_location="cpu",
    ) -> dict[str, Any]:

        import torch

        payload = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])

        if (
            scheduler is not None
            and payload.get("scheduler") is not None
        ):
            scheduler.load_state_dict(
                payload["scheduler"]
            )

        if (
            scaler is not None
            and payload.get("scaler") is not None
        ):
            scaler.load_state_dict(
                payload["scaler"]
            )

        return payload

    def list(self) -> list[str]:
        return [
            str(path)
            for path in sorted(
                self.directory.glob(
                    "checkpoint-*.pt"
                )
            )
        ]
