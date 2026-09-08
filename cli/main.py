from __future__ import annotations

import argparse
import json
import sys

from gene.agent import GeneAgent


def _print(
    value,
) -> None:

    if isinstance(
        value,
        (dict, list),
    ):
        print(
            json.dumps(
                value,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    else:
        print(value)


def _agent() -> GeneAgent:
    return GeneAgent()


def command_status(args) -> int:
    agent = _agent()
    _print(agent.status())
    return 0


def command_run(args) -> int:

    agent = _agent()

    result = agent.run(
        args.task
    )

    _print(result)

    return (
        0
        if result.get("status")
        == "completed"
        else 1
    )


def command_models(args) -> int:

    from gene.model import ModelRegistry

    registry = ModelRegistry()

    _print(
        registry.status()
    )

    return 0


def command_tools(args) -> int:

    from gene.tools.runtime import ToolRuntime

    runtime = ToolRuntime()

    _print(
        runtime.describe_tools()
    )

    return 0


def command_workers(args) -> int:

    from gene.workers import WorkerRuntime

    runtime = WorkerRuntime()

    _print(
        runtime.status()
    )

    return 0


def command_memory(args) -> int:

    from gene.knowledge.v2 import (
        LongTermIntelligence,
    )

    system = LongTermIntelligence()

    _print(
        system.status()
    )

    return 0


def command_evolution(args) -> int:

    from gene.evolution import (
        ExternalParameterStore,
    )

    store = ExternalParameterStore()

    _print(
        store.status()
    )

    return 0


def command_datasets(args) -> int:

    from pathlib import Path

    root = Path(
        args.path
        or "gene/datasets"
    )

    if not root.exists():
        _print({
            "exists": False,
            "path": str(root),
        })
        return 1

    files = []

    for path in root.rglob("*"):

        if path.is_file():

            files.append({
                "path": str(path),
                "size_bytes":
                    path.stat().st_size,
            })

    _print({
        "exists": True,
        "path": str(root),
        "files": len(files),
        "entries": files,
    })

    return 0


def command_doctor(args) -> int:

    checks = []

    def check(
        name,
        callback,
    ):

        try:
            result = callback()

            checks.append({
                "name": name,
                "success": True,
                "result": result,
            })

        except Exception as exc:

            checks.append({
                "name": name,
                "success": False,
                "error": str(exc),
            })

    check(
        "agent",
        lambda: type(
            _agent()
        ).__name__,
    )

    check(
        "tools",
        lambda: len(
            __import__(
                "gene.tools.runtime",
                fromlist=[
                    "ToolRuntime"
                ],
            )
            .ToolRuntime()
            .capability_status()
        ),
    )

    check(
        "model_registry",
        lambda: __import__(
            "gene.model",
            fromlist=[
                "ModelRegistry"
            ],
        )
        .ModelRegistry()
        .status(),
    )

    check(
        "workers",
        lambda: __import__(
            "gene.workers",
            fromlist=[
                "WorkerRuntime"
            ],
        )
        .WorkerRuntime()
        .status(),
    )

    check(
        "memory",
        lambda: __import__(
            "gene.knowledge.v2",
            fromlist=[
                "LongTermIntelligence"
            ],
        )
        .LongTermIntelligence()
        .status(),
    )

    passed = all(
        item["success"]
        for item in checks
    )

    _print({
        "healthy": passed,
        "checks": checks,
    })

    return 0 if passed else 1


def build_parser():

    parser = argparse.ArgumentParser(
        prog="gene",
        description=(
            "Gene modular agent runtime."
        ),
    )

    sub = parser.add_subparsers(
        dest="command"
    )

    status = sub.add_parser(
        "status",
        help="Show Gene runtime status.",
    )
    status.set_defaults(
        handler=command_status
    )

    run = sub.add_parser(
        "run",
        help="Execute an arbitrary user task.",
    )
    run.add_argument(
        "task",
        help="The task to give Gene.",
    )
    run.set_defaults(
        handler=command_run
    )

    models = sub.add_parser(
        "models",
        help="Show registered model backends.",
    )
    models.set_defaults(
        handler=command_models
    )

    tools = sub.add_parser(
        "tools",
        help="Show available tools.",
    )
    tools.set_defaults(
        handler=command_tools
    )

    workers = sub.add_parser(
        "workers",
        help="Show specialist workers.",
    )
    workers.set_defaults(
        handler=command_workers
    )

    memory = sub.add_parser(
        "memory",
        help="Show memory/knowledge status.",
    )
    memory.set_defaults(
        handler=command_memory
    )

    evolution = sub.add_parser(
        "evolution",
        help="Show external parameter evolution state.",
    )
    evolution.set_defaults(
        handler=command_evolution
    )

    datasets = sub.add_parser(
        "datasets",
        help="Inspect datasets.",
    )
    datasets.add_argument(
        "--path",
        default=None,
    )
    datasets.set_defaults(
        handler=command_datasets
    )

    doctor = sub.add_parser(
        "doctor",
        help="Run runtime diagnostics.",
    )
    doctor.set_defaults(
        handler=command_doctor
    )

    return parser


def main(
    argv=None,
) -> int:

    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    handler = getattr(
        args,
        "handler",
        None,
    )

    if handler is None:
        parser.print_help()
        return 0

    try:
        return handler(args)

    except KeyboardInterrupt:
        print(
            "Interrupted.",
            file=sys.stderr,
        )
        return 130

    except Exception as exc:

        print(
            f"Gene error: {exc}",
            file=sys.stderr,
        )

        return 1
