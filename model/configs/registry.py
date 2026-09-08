from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gene.model.neural.config import TransformerConfig


@dataclass(frozen=True)
class GeneModelSpec:
    name: str
    config_factory: Callable[[], TransformerConfig]
    target_parameters: int


def _gene_200m() -> TransformerConfig:
    from .gene_200m import gene_200m_config
    return gene_200m_config()


def _gene_500m() -> TransformerConfig:
    from .gene_500m import gene_500m_config
    return gene_500m_config()


def _gene_700m() -> TransformerConfig:
    from .gene_700m import gene_700m_config
    return gene_700m_config()


GENE_MODEL_REGISTRY: dict[str, GeneModelSpec] = {
    "gene-200m": GeneModelSpec(
        name="gene-200m",
        config_factory=_gene_200m,
        target_parameters=195_551_712,
    ),
    "gene-500m": GeneModelSpec(
        name="gene-500m",
        config_factory=_gene_500m,
        target_parameters=503_373_824,
    ),
    "gene-700m": GeneModelSpec(
        name="gene-700m",
        config_factory=_gene_700m,
        target_parameters=697_367_040,
    ),
}


def get_model_spec(name: str) -> GeneModelSpec:
    try:
        return GENE_MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(GENE_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown Gene model '{name}'. Available models: {available}"
        ) from exc


def list_models() -> list[str]:
    return list(GENE_MODEL_REGISTRY)
