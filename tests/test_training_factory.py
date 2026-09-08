from gene.training.factory import (
    TrainingDataFactory,
    TrainingManifest,
    TrainingRecord,
)


def test_add_and_deduplicate():

    factory = TrainingDataFactory()

    record = TrainingRecord(
        text="Gene learns from verified evidence.",
        source_type="experience",
        source_id="exp-1",
    )

    assert factory.add(record)
    assert not factory.add(record)

    assert len(
        factory.records
    ) == 1


def test_quality_filter():

    factory = TrainingDataFactory()

    accepted = factory.add(
        TrainingRecord(
            text="High quality training example.",
            source_type="teacher",
            source_id="x",
            quality=0.95,
            confidence=0.95,
        )
    )

    rejected = factory.add(
        TrainingRecord(
            text="Low quality.",
            source_type="teacher",
            source_id="y",
            quality=0.10,
            confidence=0.10,
        )
    )

    assert accepted is True
    assert rejected is False


def test_split(
    tmp_path,
):

    factory = TrainingDataFactory()

    for index in range(100):

        factory.add(
            TrainingRecord(
                text=f"training example {index}",
                source_type="test",
                source_id=str(index),
            )
        )

    train, validation = (
        factory.split(
            validation_ratio=0.10
        )
    )

    assert len(train) == 90
    assert len(validation) == 10


def test_build_and_manifest(
    tmp_path,
):

    factory = TrainingDataFactory()

    factory.add_text(
        "Verified learning example.",
        source_type="experience",
        source_id="exp",
    )

    train_path = (
        tmp_path
        / "train.jsonl"
    )

    validation_path = (
        tmp_path
        / "validation.jsonl"
    )

    result = factory.build(
        str(train_path),
        str(validation_path),
        validation_ratio=0.0,
    )

    assert result["success"] is True
    assert train_path.exists()
    assert validation_path.exists()

    manifest = TrainingManifest(
        str(
            tmp_path
            / "manifest.json"
        )
    )

    saved = manifest.save(
        factory,
        metadata={
            "model":
                "gene-200m",
        },
    )

    assert saved["version"] == 1
    assert (
        saved["metadata"]["model"]
        == "gene-200m"
    )
