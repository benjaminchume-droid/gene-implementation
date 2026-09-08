from __future__ import annotations

import json

from pathlib import Path

from gene.training.tokenizer import (
    GeneTokenizer,
)
from gene.training.packing import (
    CorpusPacker,
)


def test_packing(
    tmp_path,
):

    source = (
        tmp_path
        / "train.jsonl"
    )

    source.write_text(
        "\n".join([
            json.dumps(
                {
                    "text":
                    "hello world " * 20
                }
            ),
            json.dumps(
                {
                    "text":
                    "Gene learns from "
                    "training data. " * 20
                }
            ),
        ]),
        encoding="utf-8",
    )

    tokenizer_path = (
        tmp_path
        / "tokenizer.json"
    )

    tokenizer = GeneTokenizer(
        vocab_size=128,
        min_frequency=1,
    )

    tokenizer.train_jsonl(
        [str(source)]
    )

    tokenizer.save(
        str(tokenizer_path)
    )

    packer = CorpusPacker(
        tokenizer=tokenizer,
        block_size=32,
    )

    output = (
        tmp_path
        / "train.bin"
    )

    result = packer.pack(
        [str(source)],
        str(output),
    )

    assert result["success"] is True
    assert result["blocks"] > 0
    assert output.exists()

    blocks = list(
        packer.read_blocks(
            str(output),
            32,
        )
    )

    assert blocks
    assert all(
        len(block) == 32
        for block in blocks
    )


def test_trailing_tokens(
    tmp_path,
):

    source = (
        tmp_path
        / "train.jsonl"
    )

    source.write_text(
        '{"text":"hello world"}\n',
        encoding="utf-8",
    )

    tokenizer = GeneTokenizer(
        vocab_size=64,
        min_frequency=1,
    )

    tokenizer.train_jsonl(
        [str(source)]
    )

    packer = CorpusPacker(
        tokenizer=tokenizer,
        block_size=1000,
    )

    result = packer.pack(
        [str(source)],
        str(
            tmp_path
            / "data.bin"
        ),
    )

    assert result["success"] is True
    assert result["blocks"] == 0
    assert result["trailing_tokens"] > 0


def test_hash(
    tmp_path,
):

    path = (
        tmp_path
        / "file.bin"
    )

    path.write_bytes(
        b"gene"
    )

    digest = CorpusPacker.sha256(
        str(path)
    )

    assert len(digest) == 64
