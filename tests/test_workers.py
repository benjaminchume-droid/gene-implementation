from gene.workers import (
    WorkerRegistry,
    WorkerRuntime,
)


def test_worker_creation(
    tmp_path,
):

    registry = WorkerRegistry(
        str(
            tmp_path / "workers.json"
        )
    )

    runtime = WorkerRuntime(
        registry
    )

    worker = runtime.create(
        name="coding",
        purpose="Software development.",
        capabilities=[
            "coding"
        ],
    )

    assert worker.name == "coding"
    assert worker.enabled is True
    assert worker.state.value == "ready"


def test_worker_execution(
    tmp_path,
):

    registry = WorkerRegistry(
        str(
            tmp_path / "workers.json"
        )
    )

    runtime = WorkerRuntime(
        registry
    )

    runtime.create(
        name="research",
        purpose="Research.",
        capabilities=[
            "research"
        ],
    )

    result = runtime.run(
        "research",
        "Find information about X.",
        {
            "source":
                "test"
        },
    )

    assert result.success is True
    assert result.worker == "research"


def test_worker_capability_filter(
    tmp_path,
):

    registry = WorkerRegistry(
        str(
            tmp_path / "workers.json"
        )
    )

    runtime = WorkerRuntime(
        registry
    )

    runtime.create(
        name="coding",
        purpose="Coding.",
        capabilities=[
            "coding"
        ],
    )

    runtime.create(
        name="math",
        purpose="Mathematics.",
        capabilities=[
            "math"
        ],
    )

    workers = (
        runtime.compatible_workers(
            "coding"
        )
    )

    assert len(workers) == 1
    assert workers[0].name == "coding"


def test_worker_disable(
    tmp_path,
):

    registry = WorkerRegistry(
        str(
            tmp_path / "workers.json"
        )
    )

    runtime = WorkerRuntime(
        registry
    )

    runtime.create(
        name="test",
        purpose="Test.",
    )

    runtime.disable(
        "test"
    )

    result = runtime.run(
        "test",
        "do something",
    )

    assert result.success is False
    assert "disabled" in (
        result.error or ""
    )
