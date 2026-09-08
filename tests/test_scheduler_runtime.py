import time
from datetime import datetime, timezone, timedelta

from gene.scheduler import TaskScheduler


def test_scheduler_persists_task(tmp_path):

    path = tmp_path / "tasks.json"

    scheduler = TaskScheduler(
        path=str(path)
    )

    task = scheduler.schedule(
        name="test",
        action="test.action",
        run_at=(
            datetime.now(timezone.utc)
            + timedelta(seconds=60)
        ).isoformat(),
    )

    assert task.status == "scheduled"
    assert path.exists()

    second = TaskScheduler(
        path=str(path)
    )

    loaded = second.get(task.id)

    assert loaded is not None
    assert loaded.name == "test"


def test_scheduler_executes_task(tmp_path):

    path = tmp_path / "tasks.json"

    scheduler = TaskScheduler(
        path=str(path),
        poll_seconds=0.05,
    )

    executed = []

    def executor(task):

        executed.append(task.id)

        return {
            "success": True,
            "value": "done",
        }

    scheduler.set_executor(
        executor
    )

    task = scheduler.schedule_in(
        name="immediate",
        action="test.action",
        delay_seconds=0,
    )

    scheduler.start()

    try:

        deadline = time.time() + 3

        while (
            time.time() < deadline
            and not executed
        ):
            time.sleep(0.05)

        stored = scheduler.get(
            task.id
        )

        assert executed
        assert stored is not None
        assert stored.status == "completed"

    finally:
        scheduler.stop()


def test_scheduler_retry(tmp_path):

    path = tmp_path / "tasks.json"

    scheduler = TaskScheduler(
        path=str(path),
        poll_seconds=0.05,
    )

    attempts = []

    def executor(task):

        attempts.append(
            task.retries
        )

        if len(attempts) == 1:
            return {
                "success": False,
                "error": "temporary failure",
            }

        return {
            "success": True,
        }

    scheduler.set_executor(
        executor
    )

    task = scheduler.schedule_in(
        name="retry",
        action="test.action",
        delay_seconds=0,
        retry_delay_seconds=0,
        max_retries=2,
    )

    scheduler.start()

    try:

        deadline = time.time() + 3

        while time.time() < deadline:

            stored = scheduler.get(
                task.id
            )

            if (
                stored
                and stored.status
                == "completed"
            ):
                break

            time.sleep(0.05)

        stored = scheduler.get(task.id)

        assert stored is not None
        assert stored.status == "completed"
        assert len(attempts) >= 2

    finally:
        scheduler.stop()


def test_scheduler_cancel(tmp_path):

    scheduler = TaskScheduler(
        path=str(
            tmp_path / "tasks.json"
        )
    )

    task = scheduler.schedule_in(
        name="cancel",
        action="test.action",
        delay_seconds=60,
    )

    assert scheduler.cancel(
        task.id
    )

    assert (
        scheduler.get(task.id).status
        == "cancelled"
    )
