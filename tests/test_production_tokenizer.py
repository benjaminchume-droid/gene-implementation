import json

from pathlib import Path

from gene.training.tokenizer import (
    GeneTokenizer,
)


def test_tokenizer_training(
    tmp_path,
):

    source = (
        tmp_path
        / "train.jsonl"
    )

    records = [
        {
            "text":
                "Gene is a modular "
                "intelligence system."
        },
        {
            "text":
                "Software engineering "
                "requires testing."
        },
        {
            "text":
                "Knowledge must be "
                "verified before learning."
        },
        {
            "text":
                "Natural language contains "
                "symbols, numbers, and punctuation."
        },
    ]

    source.write_text(
        "\n".join(
            json.dumps(record)
            for record in records
        ),
        encoding="utf-8",
    )

    tokenizer = GeneTokenizer(
        vocab_size=128,
        min_frequency=1,
    )

    metadata = tokenizer.train_jsonl(
        [str(source)]
    )

    assert (
        metadata["actual_vocab_size"]
        > 4
    )

    ids = tokenizer.encode(
        "Gene learns."
    )

    assert ids

    decoded = tokenizer.decode(
        ids
    )

    assert decoded.strip()


def test_tokenizer_save_reload(
    tmp_path,
):

    source = (
        tmp_path
        / "train.jsonl"
    )

    source.write_text(
        '{"text":"hello world"}\n'
        '{"text":"hello Gene"}\n',
        encoding="utf-8",
    )

    output = (
        tmp_path
        / "tokenizer.json"
    )

    tokenizer = GeneTokenizer(
        vocab_size=64,
        min_frequency=1,
    )

    metadata = tokenizer.train_jsonl(
        [str(source)]
    )

    result = tokenizer.save(
        str(output),
        metadata=metadata,
    )

    assert result["success"] is True

    restored = GeneTokenizer.load(
        str(output)
    )

    original_ids = tokenizer.encode(
        "hello Gene"
    )

    restored_ids = restored.encode(
        "hello Gene"
    )

    assert original_ids == restored_ids


def test_tokenizer_special_tokens(
    tmp_path,
):

    source = (
        tmp_path
        / "train.jsonl"
    )

    source.write_text(
        '{"text":"example training text"}\n',
        encoding="utf-8",
    )

    tokenizer = GeneTokenizer(
        vocab_size=64,
        min_frequency=1,
    )

    tokenizer.train_jsonl(
        [str(source)]
    )

    assert (
        tokenizer.token_id("<pad>")
        is not None
    )

    assert (
        tokenizer.token_id("<unk>")
        is not None
    )

    assert (
        tokenizer.token_id("<bos>")
        is not None
    )

    assert (
        tokenizer.token_id("<eos>")
        is not None
    )
