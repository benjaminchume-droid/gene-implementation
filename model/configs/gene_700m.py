from __future__ import annotations

from gene.model.neural.config import TransformerConfig


def gene_700m_config() -> TransformerConfig:
    """
    Gene-700M model configuration.

    The actual parameter count is measured from the instantiated model.
    This configuration targets approximately 700M parameters.
    """

    config = TransformerConfig(
        vocab_size=32768,
        hidden_size=1280,
        intermediate_size=5120,
        num_layers=25,
        num_heads=20,
        max_position_embeddings=112000,
        attention_window=8192,
        dropout=0.0,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        model_name="gene-700m",
    )

    config.validate()
    return config


def build_gene_700m():
    from gene.model.neural import GeneTransformer

    return GeneTransformer(gene_700m_config())

