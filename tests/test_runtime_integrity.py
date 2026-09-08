import tempfile
from pathlib import Path

import torch

from gene.model.configs import gene_200m_config
from gene.model.neural.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from gene.model.neural.transformer import GeneTransformer


def test_cached_decode_matches_full_forward():

    torch.manual_seed(1234)

    config = gene_200m_config()

    model = GeneTransformer(config).eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 32),
    )

    with torch.no_grad():

        full = model(
            input_ids
        )["logits"][:, -1, :]

        prefix = input_ids[:, :-1]

        cached = model(
            prefix,
            use_cache=True,
        )

        decoded = model(
            input_ids[:, -1:],
            past_key_values=
                cached["past_key_values"],
            use_cache=True,
            position_offset=
                prefix.shape[1],
        )

        cached_logits = (
            decoded["logits"][:, -1, :]
        )

    torch.testing.assert_close(
        full,
        cached_logits,
        rtol=1e-4,
        atol=1e-4,
    )


def test_cached_decode_multiple_tokens():

    torch.manual_seed(5678)

    config = gene_200m_config()

    model = GeneTransformer(config).eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 24),
    )

    prefix = input_ids[:, :16]
    suffix = input_ids[:, 16:]

    with torch.no_grad():

        cached = model(
            prefix,
            use_cache=True,
        )

        current_cache = (
            cached["past_key_values"]
        )

        cached_outputs = []

        for index in range(
            suffix.shape[1]
        ):

            token = suffix[
                :,
                index:index + 1
            ]

            result = model(
                token,
                past_key_values=
                    current_cache,
                use_cache=True,
                position_offset=
                    16 + index,
            )

            cached_outputs.append(
                result["logits"]
            )

            current_cache = (
                result["past_key_values"]
            )

        cached_logits = torch.cat(
            cached_outputs,
            dim=1,
        )

        full_logits = model(
            input_ids
        )["logits"][:, 16:, :]

    torch.testing.assert_close(
        full_logits,
        cached_logits,
        rtol=1e-4,
        atol=1e-4,
    )


def test_checkpoint_roundtrip():

    torch.manual_seed(999)

    config = gene_200m_config()

    model = GeneTransformer(config).eval()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 8),
    )

    with torch.no_grad():
        before = model(
            input_ids
        )["logits"]

    with tempfile.TemporaryDirectory() as temp:

        path = Path(temp) / "roundtrip.pt"

        save_checkpoint(
            model,
            path,
        )

        restored = load_checkpoint(
            path,
            map_location="cpu",
        ).eval()

        with torch.no_grad():
            after = restored(
                input_ids
            )["logits"]

    torch.testing.assert_close(
        before,
        after,
        rtol=1e-5,
        atol=1e-5,
    )


def test_context_boundary():

    config = gene_200m_config()

    model = GeneTransformer(config).eval()

    token = torch.randint(
        0,
        config.vocab_size,
        (1, 1),
    )

    with torch.no_grad():

        result = model(
            token,
            position_offset=
                config.max_position_embeddings - 1,
        )

    assert result["logits"].shape == (
        1,
        1,
        config.vocab_size,
    )


def test_context_overflow():

    config = gene_200m_config()

    model = GeneTransformer(config).eval()

    token = torch.randint(
        0,
        config.vocab_size,
        (1, 1),
    )

    try:

        model(
            token,
            position_offset=
                config.max_position_embeddings,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected context overflow to raise ValueError."
    )


if __name__ == "__main__":

    test_cached_decode_matches_full_forward()
    print("CACHE EQUIVALENCE: PASSED")

    test_cached_decode_multiple_tokens()
    print("MULTI-TOKEN CACHE EQUIVALENCE: PASSED")

    test_checkpoint_roundtrip()
    print("CHECKPOINT ROUNDTRIP: PASSED")

    test_context_boundary()
    print("112K CONTEXT BOUNDARY: PASSED")

    test_context_overflow()
    print("CONTEXT OVERFLOW: PASSED")

    print("")
    print(
        "PHASE 3 CHECKPOINT + CACHE INTEGRITY: PASSED"
    )
