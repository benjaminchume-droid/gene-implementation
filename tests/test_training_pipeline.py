from gene.training.pipeline.registry import (
    DatasetRegistry,
)
from gene.training.pipeline.normalize import (
    DatasetNormalizer,
)
from gene.training.pipeline.deduplicate import (
    Deduplicator,
)
from gene.training.pipeline.split import (
    DatasetSplitter,
)


def test_registry(tmp_path):

    registry = DatasetRegistry(
        str(
            tmp_path / "datasets.json"
        )
    )

    spec = registry.register(
        name="test",
        source="local",
        path="data",
        license="test",
    )

    assert spec.name == "test"
    assert len(
        registry.list()
    ) == 1


def test_normalization():

    normalizer = DatasetNormalizer()

    record = normalizer.normalize(
        {
            "title":
                "Example",
            "text":
                "Hello   world.",
            "extra":
                "metadata",
        },
        dataset_id="dataset",
        source="source",
    )

    assert record is not None
    assert record.text == (
        "Hello world."
    )
    assert (
        record.dataset_id
        == "dataset"
    )


def test_generic_schema_fallback():

    normalizer = DatasetNormalizer()

    record = normalizer.normalize(
        {
            "arbitrary":
                {
                    "nested":
                        "Some training text."
                }
        },
        dataset_id="dataset",
        source="source",
    )

    assert record is not None
    assert "training text" in (
        record.text
    )


def test_deduplication():

    deduplicator = Deduplicator()

    assert deduplicator.accept(
        "same text"
    )

    assert not deduplicator.accept(
        "Same   text"
    )


def test_deterministic_split():

    splitter = DatasetSplitter(
        validation_ratio=0.1,
        seed=42,
    )

    first = splitter.assign(
        "stable-record-id"
    )

    second = splitter.assign(
        "stable-record-id"
    )

    assert first == second
