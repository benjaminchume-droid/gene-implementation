from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import TransformerConfig
from .transformer import GeneTransformer


def save_checkpoint(
    model: GeneTransformer,
    path: str,
) -> dict:

    target = Path(path)

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "config":
            model.config.to_dict(),
        "state_dict":
            model.state_dict(),
    }

    torch.save(
        payload,
        target,
    )

    return {
        "success": True,
        "path": str(target),
        "parameters":
            model.parameter_count(),
    }


def load_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> GeneTransformer:

    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    config = TransformerConfig(
        **payload["config"]
    )

    model = GeneTransformer(
        config
    )

    model.load_state_dict(
        payload["state_dict"]
    )

    return model


def model_manifest(
    model: GeneTransformer,
) -> dict:

    return {
        "model_name":
            model.config.model_name,
        "config":
            model.config.to_dict(),
        "parameters":
            model.parameter_count(),
        "trainable_parameters":
            model.trainable_parameter_count(),
    }
