from __future__ import annotations

from gene.model.configs import (
    gene_200m_config,
    gene_500m_config,
    gene_700m_config,
)
from gene.model.neural import GeneTransformer


EXPECTED = {
    "gene-200m": 195_551_712,
    "gene-500m": 503_374_848,
    "gene-700m": 697_368_320,
}


def build_and_check(name, config_factory):
    config = config_factory()
    model = GeneTransformer(config)

    actual = model.parameter_count()
    trainable = model.trainable_parameter_count()

    print(f"\n{name}")
    print("=" * len(name))
    print("Parameters:", actual)
    print("Trainable:", trainable)
    print("Config:", config.to_dict())

    assert actual == EXPECTED[name], (
        f"{name}: expected {EXPECTED[name]:,} parameters, "
        f"got {actual:,}"
    )

    assert trainable == actual, (
        f"{name}: trainable parameter count differs from total"
    )

    return model


def main():
    models = [
        ("gene-200m", gene_200m_config),
        ("gene-500m", gene_500m_config),
        ("gene-700m", gene_700m_config),
    ]

    for name, factory in models:
        build_and_check(name, factory)

    print("\nPHASE 1 MODEL FAMILY VALIDATION: PASSED")


if __name__ == "__main__":
    main()
