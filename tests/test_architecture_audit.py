from pathlib import Path


ROOT = Path("gene")

FORBIDDEN_PATTERNS = [
    "max_position_embeddings=8192",
    "max_position_embeddings = 8192",
    "attention_window=2048",
    "attention_window = 2048",
]


def test_no_stale_model_context_constants():

    matches = []

    for path in ROOT.rglob("*.py"):

        try:
            text = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            continue

        for pattern in FORBIDDEN_PATTERNS:

            if pattern in text:
                matches.append(
                    f"{path}: {pattern}"
                )

    assert not matches, (
        "Stale context constants found:\n"
        + "\n".join(matches)
    )


def test_all_model_configs_use_112k():

    config_files = [
        ROOT
        / "model"
        / "configs"
        / "gene_200m.py",

        ROOT
        / "model"
        / "configs"
        / "gene_500m.py",

        ROOT
        / "model"
        / "configs"
        / "gene_700m.py",
    ]

    for path in config_files:

        text = path.read_text(
            encoding="utf-8"
        )

        assert (
            "max_position_embeddings=112000"
            in text
        )

        assert (
            "attention_window=8192"
            in text
        )


if __name__ == "__main__":

    test_no_stale_model_context_constants()
    print(
        "STALE CONTEXT CONSTANT AUDIT: PASSED"
    )

    test_all_model_configs_use_112k()
    print(
        "MODEL CONFIG AUDIT: PASSED"
    )

    print("")
    print(
        "PHASE 5 ARCHITECTURE AUDIT: PASSED"
    )
