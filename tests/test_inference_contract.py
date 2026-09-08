import torch

from gene.model.configs import (
    gene_200m_config,
    gene_500m_config,
    gene_700m_config,
)

from gene.model.neural.transformer import (
    GeneTransformer,
)

from gene.inference.limits import (
    limits_from_config,
)


def test_all_model_inference_limits():

    configs = [
        gene_200m_config(),
        gene_500m_config(),
        gene_700m_config(),
    ]

    for config in configs:

        limits = limits_from_config(
            config
        )

        assert limits.context_length == 112000
        assert limits.attention_window == 8192
        assert limits.max_new_tokens == 256


def test_inference_forward_contract():

    config = gene_200m_config()

    model = GeneTransformer(
        config
    ).eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 32),
    )

    with torch.no_grad():

        output = model(
            input_ids
        )

    assert "logits" in output
    assert "loss" in output

    assert output["logits"].shape == (
        1,
        32,
        config.vocab_size,
    )


def test_inference_generation_contract():

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

        output = model.generate(
            input_ids,
            max_new_tokens=8,
            temperature=0.0,
        )

    assert output.shape == (
        1,
        24,
    )

    assert torch.equal(
        output[:, :16],
        input_ids,
    )


if __name__ == "__main__":

    test_all_model_inference_limits()
    print(
        "INFERENCE LIMITS: PASSED"
    )

    test_inference_forward_contract()
    print(
        "INFERENCE FORWARD CONTRACT: PASSED"
    )

    test_inference_generation_contract()
    print(
        "INFERENCE GENERATION CONTRACT: PASSED"
    )

    print("")
    print(
        "PHASE 4 INFERENCE CONTRACT: PASSED"
    )
