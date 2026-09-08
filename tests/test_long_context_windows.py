from gene.training.long_context.windows import (
    build_training_windows,
)


def test_short_sequence():

    tokens = list(
        range(1000)
    )

    windows = build_training_windows(
        tokens
    )

    assert len(windows) == 1

    assert windows[0].start == 0
    assert windows[0].end == 1000
    assert windows[0].loss_start == 0
    assert windows[0].loss_end == 1000


def test_long_sequence():

    tokens = list(
        range(20000)
    )

    windows = build_training_windows(
        tokens,
        chunk_size=8192,
        overlap=1024,
    )

    assert len(windows) > 1

    assert windows[0].start == 0
    assert windows[0].end == 8192
    assert windows[0].loss_start == 0

    assert windows[1].start == 7168
    assert windows[1].loss_start == 8192


def test_window_coverage():

    tokens = list(
        range(30000)
    )

    windows = build_training_windows(
        tokens,
        chunk_size=8192,
        overlap=1024,
    )

    covered = set()

    for window in windows:

        covered.update(
            range(
                window.start,
                window.end,
            )
        )

    assert covered == set(
        range(len(tokens))
    )


def test_112k_boundary():

    tokens = list(
        range(112000)
    )

    windows = build_training_windows(
        tokens,
        context_length=112000,
        chunk_size=8192,
        overlap=1024,
    )

    assert windows

    assert windows[-1].end == 112000


def test_overflow_rejected():

    tokens = list(
        range(112001)
    )

    try:

        build_training_windows(
            tokens,
            context_length=112000,
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected context overflow."
    )


if __name__ == "__main__":

    test_short_sequence()
    print(
        "SHORT TRAINING WINDOW: PASSED"
    )

    test_long_sequence()
    print(
        "LONG TRAINING WINDOW: PASSED"
    )

    test_window_coverage()
    print(
        "TRAINING WINDOW COVERAGE: PASSED"
    )

    test_112k_boundary()
    print(
        "112K TRAINING BOUNDARY: PASSED"
    )

    test_overflow_rejected()
    print(
        "TRAINING OVERFLOW: PASSED"
    )

    print("")
    print(
        "PHASE 6 LONG-CONTEXT WINDOWS: PASSED"
    )
