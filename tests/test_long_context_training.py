from gene.training.long_context import (
    chunk_token_sequence,
    default_long_context_config,
)


def test_long_context_training_config():

    config = default_long_context_config()

    assert config.context_length == 112000
    assert config.attention_window == 8192
    assert config.chunk_size == 8192
    assert config.overlap == 1024

    assert config.stride == 7168


def test_chunking():

    tokens = list(range(20000))

    chunks = chunk_token_sequence(
        tokens,
        chunk_size=8192,
        overlap=1024,
    )

    assert len(chunks) > 1

    assert chunks[0].start == 0
    assert chunks[0].end == 8192

    assert chunks[1].start == 7168


def test_chunk_reconstruction_coverage():

    tokens = list(range(20000))

    chunks = chunk_token_sequence(
        tokens,
        chunk_size=8192,
        overlap=1024,
    )

    covered = set()

    for chunk in chunks:

        covered.update(
            range(
                chunk.start,
                chunk.end,
            )
        )

    assert covered == set(
        range(len(tokens))
    )


if __name__ == "__main__":

    test_long_context_training_config()
    print(
        "LONG-CONTEXT TRAINING CONFIG: PASSED"
    )

    test_chunking()
    print(
        "LONG-CONTEXT CHUNKING: PASSED"
    )

    test_chunk_reconstruction_coverage()
    print(
        "CHUNK COVERAGE: PASSED"
    )

    print("")
    print(
        "PHASE 6 TRAINING CONTRACT: PASSED"
    )
