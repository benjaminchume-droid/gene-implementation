from __future__ import annotations

from gene.model.neural.config import TransformerConfig


def gene_500m_config() -> TransformerConfig:
    """
    Gene-500M model configuration.

    The actual parameter count is measured from the instantiated model.
    This configuration targets approximately 500M parameters.
    """

    config = TransformerConfig(
        vocab_size=32768,
        hidden_size=1024,
        intermediate_size=4096,
        num_layers=28,
        num_heads=16,
        max_position_embeddings=112000,
        attention_window=8192,
        dropout=0.0,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        model_name="gene-500m",
    )

    config.validate()
    return config


def build_gene_500m():
    from gene.model.neural import GeneTransformer

    return GeneTransformer(gene_500m_config())

