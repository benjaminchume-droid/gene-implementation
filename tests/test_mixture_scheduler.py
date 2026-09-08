from gene.training.mixture import (
    MixtureManifest,
    MixtureScheduler,
    MixtureSource,
)


def test_normalized_weights():

    scheduler = MixtureScheduler(
        [
            MixtureSource(
                "a",
                "a.jsonl",
                1.0,
            ),
            MixtureSource(
                "b",
                "b.jsonl",
                3.0,
            ),
        ],
        seed=42,
    )

    probabilities = scheduler.probabilities()

    assert abs(
        sum(
            probabilities.values()
        ) - 1.0
    ) < 1e-8

    assert probabilities["b"] > (
        probabilities["a"]
    )


def test_deterministic_schedule():

    sources = [
        MixtureSource(
            "a",
            "a.jsonl",
            1.0,
        ),
        MixtureSource(
            "b",
            "b.jsonl",
            2.0,
        ),
    ]

    first = MixtureScheduler(
        sources,
        seed=42,
    ).deterministic_schedule(30)

    second = MixtureScheduler(
        sources,
        seed=42,
    ).deterministic_schedule(30)

    assert [
        item.name
        for item in first
    ] == [
        item.name
        for item in second
    ]

    assert len(first) == 30


def test_weight_update():

    scheduler = MixtureScheduler(
        [
            MixtureSource(
                "a",
                "a.jsonl",
                1.0,
            ),
            MixtureSource(
                "b",
                "b.jsonl",
                1.0,
            ),
        ]
    )

    scheduler.update_weight(
        "a",
        4.0,
    )

    probabilities = scheduler.probabilities()

    assert probabilities["a"] == 0.8
    assert probabilities["b"] == 0.2


def test_manifest(
    tmp_path,
):

    scheduler = MixtureScheduler(
        [
            MixtureSource(
                "train",
                "train.jsonl",
                1.0,
            )
        ]
    )

    manifest = MixtureManifest(
        str(
            tmp_path
            / "manifest.json"
        )
    )

    manifest.save(
        scheduler,
        metadata={
            "purpose":
                "gene 200m training",
        },
    )

    loaded = manifest.load()

    assert (
        loaded["version"]
        == 1
    )

    assert (
        loaded["metadata"]["purpose"]
        == "gene 200m training"
    )
