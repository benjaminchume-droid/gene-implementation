from __future__ import annotations

from gene.model.neural.config import (
    TransformerConfig,
)


def gene_200m_config() -> TransformerConfig:
    """
    Target configuration for the first large Gene model.

    The actual parameter count is always measured from
    the instantiated model; the name is a target scale,
    not a hard-coded parameter-count claim.
    """

    config = TransformerConfig(
        vocab_size=32768,
        hidden_size=864,
        intermediate_size=3456,
        num_layers=14,
        num_heads=12,
        max_position_embeddings=112000,
        attention_window=8192,
        dropout=0.0,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        model_name="gene-200m",
    )

    config.validate()

    return config


def build_gene_200m():
    from gene.model.neural import (
        GeneTransformer,
    )

    return GeneTransformer(
        gene_200m_config()
    )

