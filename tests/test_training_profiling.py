import torch

from gene.model.neural import (
    GeneTransformer,
    TransformerConfig,
)

from gene.training.profiling import (
    HardwareProfiler,
    hardware_report,
)


def test_hardware_report():

    report = hardware_report()

    assert "platform" in report
    assert "python" in report
    assert "torch" in report


def test_training_profile():

    config = TransformerConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=64,
        dropout=0.0,
        model_name="profiling-test",
    )

    model = GeneTransformer(
        config
    )

    profiler = HardwareProfiler(
        model,
        device="cpu",
    )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 32),
    )

    profile = profiler.profile_step(
        input_ids
    )

    assert profile.parameter_count > 0
    assert profile.trainable_parameter_count > 0
    assert profile.forward_seconds >= 0
    assert profile.backward_seconds >= 0
    assert profile.optimizer_seconds >= 0
    assert profile.total_step_seconds > 0
    assert profile.tokens_per_second > 0
    assert profile.loss > 0
