import torch

from gene.model.configs import gene_200m_config
from gene.model.neural.transformer import GeneTransformer


def test_greedy_generation():

    torch.manual_seed(42)

    config = gene_200m_config()

    model = GeneTransformer(config).eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 8),
    )

    output = model.generate(
        input_ids,
        max_new_tokens=8,
        temperature=0.0,
    )

    assert output.shape == (
        1,
        16,
    )

    assert torch.equal(
        output[:, :8],
        input_ids,
    )


def test_sampling_generation():

    torch.manual_seed(42)

    config = gene_200m_config()

    model = GeneTransformer(config).eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 8),
    )

    output = model.generate(
        input_ids,
        max_new_tokens=8,
        temperature=1.0,
        top_k=20,
    )

    assert output.shape == (
        1,
        16,
    )


def test_generation_context_limit():

    config = gene_200m_config()

    model = GeneTransformer(config).eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 8),
    )

    remaining = (
        config.max_position_embeddings
        - input_ids.shape[1]
    )

    output = model.generate(
        input_ids,
        max_new_tokens=remaining + 100,
        temperature=0.0,
    )

    assert output.shape[1] <= (
        config.max_position_embeddings
    )


if __name__ == "__main__":

    test_greedy_generation()
    print("GREEDY GENERATION: PASSED")

    test_sampling_generation()
    print("SAMPLING GENERATION: PASSED")

    test_generation_context_limit()
    print("GENERATION CONTEXT LIMIT: PASSED")

    print("")
    print(
        "PHASE 4 GENERATION INTEGRATION: PASSED"
    )
