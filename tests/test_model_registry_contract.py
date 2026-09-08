from gene.model.configs import (
    gene_200m_config,
    gene_500m_config,
    gene_700m_config,
)

from gene.model.neural.transformer import (
    GeneTransformer,
)


EXPECTED = {
    "gene-200m": 195551712,
    "gene-500m": 503374848,
    "gene-700m": 697368320,
}


def test_registry_configs():

    configs = [
        gene_200m_config(),
        gene_500m_config(),
        gene_700m_config(),
    ]

    assert len(configs) == 3

    names = {
        config.model_name
        for config in configs
    }

    assert names == set(
        EXPECTED.keys()
    )


def test_parameter_contracts():

    configs = [
        gene_200m_config(),
        gene_500m_config(),
        gene_700m_config(),
    ]

    for config in configs:

        model = GeneTransformer(
            config
        )

        actual = (
            model.parameter_count()
        )

        expected = EXPECTED[
            config.model_name
        ]

        assert actual == expected


def test_shared_context_contract():

    configs = [
        gene_200m_config(),
        gene_500m_config(),
        gene_700m_config(),
    ]

    for config in configs:

        assert (
            config.max_position_embeddings
            == 112000
        )

        assert (
            config.attention_window
            == 8192
        )


if __name__ == "__main__":

    test_registry_configs()
    print(
        "MODEL REGISTRY CONFIGS: PASSED"
    )

    test_parameter_contracts()
    print(
        "PARAMETER CONTRACTS: PASSED"
    )

    test_shared_context_contract()
    print(
        "SHARED CONTEXT CONTRACT: PASSED"
    )

    print("")
    print(
        "PHASE 5 MODEL REGISTRY CONTRACT: PASSED"
    )
