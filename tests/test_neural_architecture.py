from gene.model.neural import (
    GeneTransformer,
    TransformerConfig,
    load_checkpoint,
    model_manifest,
    save_checkpoint,
)


def test_config_validation():

    config = TransformerConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=128,
    )

    config.validate()

    assert config.head_dim == 16


def test_forward_pass():

    import torch

    config = TransformerConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=128,
    )

    model = GeneTransformer(
        config
    )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    outputs = model(
        input_ids,
        labels=input_ids,
    )

    assert outputs["logits"].shape == (
        2,
        16,
        128,
    )

    assert outputs["loss"] is not None
    assert outputs["loss"].item() >= 0


def test_generation():

    import torch

    config = TransformerConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=64,
    )

    model = GeneTransformer(
        config
    )

    input_ids = torch.randint(
        0,
        64,
        (1, 8),
    )

    output = model.generate(
        input_ids,
        max_new_tokens=4,
        temperature=0,
    )

    assert output.shape == (
        1,
        12,
    )


def test_checkpoint_roundtrip(
    tmp_path,
):

    import torch

    config = TransformerConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=64,
    )

    model = GeneTransformer(
        config
    )

    path = (
        tmp_path
        / "gene-test.pt"
    )

    result = save_checkpoint(
        model,
        str(path),
    )

    assert result["success"] is True

    restored = load_checkpoint(
        str(path)
    )

    inputs = torch.randint(
        0,
        64,
        (1, 8),
    )

    original = model(
        inputs
    )["logits"]

    reloaded = restored(
        inputs
    )["logits"]

    assert torch.allclose(
        original,
        reloaded,
        atol=1e-5,
    )


def test_manifest():

    config = TransformerConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=64,
    )

    model = GeneTransformer(
        config
    )

    manifest = model_manifest(
        model
    )

    assert manifest["parameters"] > 0
    assert (
        manifest["model_name"]
        == "gene-transformer"
    )
