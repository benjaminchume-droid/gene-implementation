from gene.training.tests.overfit import (
    run_test,
)


def test_tiny_model_overfit():

    result = run_test()

    assert result["parameters"] > 0

    assert result["first_loss"] > 0

    assert result["last_loss"] > 0

    assert result["loss_reduced"] is True
