import json

from gene.training.conversation import (
    ConversationDatasetBuilder,
)


def test_dataset_builder(
    tmp_path,
):

    output = (
        tmp_path
        / "conversation.jsonl"
    )

    result = (
        ConversationDatasetBuilder()
        .build(
            str(output)
        )
    )

    assert result["success"] is True
    assert result["records"] > 0
    assert output.exists()

    lines = output.read_text(
        encoding="utf-8"
    ).splitlines()

    assert lines

    record = json.loads(
        lines[0]
    )

    assert record["messages"]
    assert record["text"]
