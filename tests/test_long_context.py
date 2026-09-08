import torch

from gene.model.configs import (
    gene_200m_config,
    gene_500m_config,
    gene_700m_config,
)

from gene.model.neural.rope import (
    RotaryEmbedding,
)

from gene.model.neural.transformer import (
    GeneTransformer,
)


def test_configurations():

    configs = [
        gene_200m_config(),
        gene_500m_config(),
        gene_700m_config(),
    ]

    for config in configs:
        config.validate()

        assert (
            config.max_position_embeddings
            == 112000
        )

        assert (
            config.attention_window
            == 8192
        )


def test_dynamic_rope():

    rope = RotaryEmbedding(
        head_dim=72,
        max_position_embeddings=112000,
    )

    x = torch.zeros(
        1,
        12,
        9000,
        72,
    )

    cos, sin = rope(
        x,
        position_offset=0,
    )

    assert cos.shape == (
        1,
        1,
        9000,
        36,
    )

    assert sin.shape == (
        1,
        1,
        9000,
        36,
    )


def test_112k_rope_position():

    rope = RotaryEmbedding(
        head_dim=72,
        max_position_embeddings=112000,
    )

    x = torch.zeros(
        1,
        1,
        16,
        72,
    )

    cos, sin = rope(
        x,
        position_offset=111984,
    )

    assert cos.shape == (
        1,
        1,
        16,
        36,
    )

    assert sin.shape == (
        1,
        1,
        16,
        36,
    )


def test_200m_forward():

    config = gene_200m_config()

    model = GeneTransformer(
        config
    ).eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 16),
    )

    with torch.no_grad():
        output = model(
            input_ids
        )

    assert output["logits"].shape == (
        1,
        16,
        config.vocab_size,
    )


def test_kv_cache():

    config = gene_200m_config()

    model = GeneTransformer(
        config
    ).eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 16),
    )

    with torch.no_grad():

        cached = model(
            input_ids,
            use_cache=True,
        )

        decoded = model(
            input_ids[:, -1:],
            past_key_values=
                cached["past_key_values"],
            use_cache=True,
            position_offset=15,
        )

    assert cached["logits"].shape == (
        1,
        16,
        config.vocab_size,
    )

    assert decoded["logits"].shape == (
        1,
        1,
        config.vocab_size,
    )

    assert len(
        cached["past_key_values"]
    ) == config.num_layers


def test_model_family():

    configs = [
        gene_200m_config(),
        gene_500m_config(),
        gene_700m_config(),
    ]

    for config in configs:

        model = GeneTransformer(
            config
        ).eval()

        input_ids = torch.randint(
            0,
            config.vocab_size,
            (1, 4),
        )

        with torch.no_grad():
            output = model(
                input_ids
            )

        assert output["logits"].shape == (
            1,
            4,
            config.vocab_size,
        )


if __name__ == "__main__":

    test_configurations()
    print("CONFIGURATION: PASSED")

    test_dynamic_rope()
    print("DYNAMIC ROPE: PASSED")

    test_112k_rope_position()
    print("112K ROPE POSITION: PASSED")

    test_200m_forward()
    print("200M FORWARD: PASSED")

    test_kv_cache()
    print("KV CACHE: PASSED")

    test_model_family()
    print("MODEL FAMILY: PASSED")

    print("")
    print(
        "PHASE 2 LONG-CONTEXT ARCHITECTURE: PASSED"
    )
