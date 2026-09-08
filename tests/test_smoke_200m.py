from gene.training.tests.smoke_200m import (
    run_smoke_test,
)


def test_gene_200m_smoke():

    result = run_smoke_test()

    assert result["parameter_count"] > 0
    assert result["trainable_parameter_count"] > 0

    assert result["forward"] is True
    assert result["backward"] is True
    assert result["optimizer_step"] is True

    assert result["gradient_tensors"] > 0
    assert result["changed_parameters"] > 0

    assert result["loss"] > 0
