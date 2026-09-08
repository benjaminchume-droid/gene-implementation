from gene.cli.main import build_parser


def test_cli_parser():

    parser = build_parser()

    args = parser.parse_args(
        ["status"]
    )

    assert args.command == "status"


def test_cli_arbitrary_task():

    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "do whatever the user asks",
        ]
    )

    assert args.command == "run"
    assert args.task == (
        "do whatever the user asks"
    )


def test_cli_datasets():

    parser = build_parser()

    args = parser.parse_args(
        [
            "datasets",
            "--path",
            "gene/datasets",
        ]
    )

    assert args.command == "datasets"
    assert args.path == "gene/datasets"
