import json

from gene.training.normalization import (
    DatasetNormalizer,
)


def test_normalization():

    record = {
        "prompt":
            "Explain the concept.",
        "text":
            "This is a detailed explanation "
            "of an arbitrary subject.",
        "token_length":
            25,
        "audience":
            "general",
        "format":
            "text",
        "seed_data":
            "unknown_source",
    }

    normalizer = DatasetNormalizer()

    result = normalizer.normalize_record(
        record,
        source_type="dataset",
        source_id="test",
        line_number=1,
    )

    assert result is not None
    assert result.text
    assert result.competency
    assert result.competency["stage"]


def test_no_subject_taxonomy():

    normalizer = DatasetNormalizer()

    record = {
        "text":
            "Develop a complex reusable workflow "
            "and verify each step.",
    }

    result = normalizer.normalize_record(
        record,
        source_type="unknown",
        source_id="unknown",
        line_number=1,
    )

    assert result is not None

    # No domain whitelist is required.
    assert (
        result.competency["domain"]
        is None
    )
