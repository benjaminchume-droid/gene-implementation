from gene.training.pilot import (
    run_pilot,
)


def test_pilot_profiler():

    result = run_pilot(
        packed_path=(
            "gene/data/training/packed/"
            "train.bin"
        ),
        steps=1,
        sequence_length=64,
    )

    assert result[
        "parameter_count"
    ] > 0

    assert result[
        "trainable_parameter_count"
    ] > 0

    assert result[
        "average_step_seconds"
    ] > 0

    assert result[
        "average_tokens_per_second"
    ] > 0

    assert result[
        "first_loss"
    ] > 0

    assert result[
        "last_loss"
    ] > 0
