from __future__ import annotations

import json
import threading
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable


@dataclass
class ScheduledTask:
    id: str
    name: str
    action: str
    run_at: str

    payload: dict[str, Any] = field(
        default_factory=dict
    )

    status: str = "scheduled"

    max_retries: int = 3
    retries: int = 0

    retry_delay_seconds: int = 30

    created_at: str = field(
        default_factory=lambda:
        datetime.now(timezone.utc).isoformat()
    )

    started_at: str | None = None
    completed_at: str | None = None

    last_error: str | None = None
    result: dict[str, Any] | None = None

    worker_id: str | None = None


class TaskScheduler:

    def __init__(
        self,
        path: str = (
            "gene/data/scheduler/tasks.json"
        ),
        poll_seconds: float = 2.0,
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.poll_seconds = poll_seconds

        self.tasks: dict[str, ScheduledTask] = {}

        self.executor: Callable[
            [ScheduledTask],
            dict[str, Any]
        ] | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

        self._load()

        self._recover_interrupted_tasks()

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def _load(self) -> None:

        if not self.path.exists():
            return

        raw = self.path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not raw:
            return

        data = json.loads(raw)

        for item in data:
            task = ScheduledTask(**item)
            self.tasks[task.id] = task

    def _save(self) -> None:

        with self._lock:

            tmp = self.path.with_suffix(
                ".tmp"
            )

            tmp.write_text(
                json.dumps(
                    [
                        asdict(task)
                        for task
                        in self.tasks.values()
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            tmp.replace(self.path)

    # ---------------------------------------------------------
    # Recovery
    # ---------------------------------------------------------

    def _recover_interrupted_tasks(self) -> None:

        changed = False

        for task in self.tasks.values():

            if task.status == "running":

                task.status = "scheduled"
                task.worker_id = None
                task.last_error = (
                    "Recovered after Gene restart."
                )

                changed = True

        if changed:
            self._save()

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    def set_executor(
        self,
        executor: Callable[
            [ScheduledTask],
            dict[str, Any]
        ],
    ) -> None:

        self.executor = executor

    # ---------------------------------------------------------
    # Scheduling
    # ---------------------------------------------------------

    def schedule(
        self,
        name: str,
        action: str,
        run_at: str,
        payload: dict[str, Any] | None = None,
        max_retries: int = 3,
        retry_delay_seconds: int = 30,
    ) -> ScheduledTask:

        task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name,
            action=action,
            run_at=run_at,
            payload=payload or {},
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )

        with self._lock:
            self.tasks[task.id] = task

        self._save()

        return task

    def schedule_in(
        self,
        name: str,
        action: str,
        delay_seconds: int,
        payload: dict[str, Any] | None = None,
        max_retries: int = 3,
        retry_delay_seconds: int = 30,
    ) -> ScheduledTask:

        run_at = (
            datetime.now(timezone.utc)
            + timedelta(
                seconds=delay_seconds
            )
        ).isoformat()

        return self.schedule(
            name=name,
            action=action,
            run_at=run_at,
            payload=payload,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )

    # ---------------------------------------------------------
    # Task lifecycle
    # ---------------------------------------------------------

    def cancel(
        self,
        task_id: str,
    ) -> bool:

        with self._lock:

            task = self.tasks.get(task_id)

            if task is None:
                return False

            if task.status in {
                "completed",
                "cancelled",
            }:
                return False

            task.status = "cancelled"

        self._save()

        return True

    def get(
        self,
        task_id: str,
    ) -> ScheduledTask | None:

        with self._lock:
            return self.tasks.get(task_id)

    def list(
        self,
        status: str | None = None,
    ) -> list[ScheduledTask]:

        with self._lock:

            values = list(
                self.tasks.values()
            )

        if status is not None:
            values = [
                task
                for task in values
                if task.status == status
            ]

        return values

    # ---------------------------------------------------------
    # Due-task logic
    # ---------------------------------------------------------

    @staticmethod
    def _is_due(
        run_at: str,
    ) -> bool:

        try:
            target = datetime.fromisoformat(
                run_at
            )

            if target.tzinfo is None:
                target = target.replace(
                    tzinfo=timezone.utc
                )

            return (
                datetime.now(timezone.utc)
                >= target
            )

        except ValueError:
            return False

    # ---------------------------------------------------------
    # Execution
    # ---------------------------------------------------------

    def _claim_due_task(
        self,
    ) -> ScheduledTask | None:

        with self._lock:

            for task in self.tasks.values():

                if task.status != "scheduled":
                    continue

                if not self._is_due(
                    task.run_at
                ):
                    continue

                task.status = "running"

                task.started_at = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                task.worker_id = str(
                    uuid.uuid4()
                )

                self._save()

                return task

        return None

    def _execute_task(
        self,
        task: ScheduledTask,
    ) -> None:

        if self.executor is None:

            task.status = "failed"
            task.last_error = (
                "No scheduler executor configured."
            )

            self._save()

            return

        try:

            result = self.executor(task)

            if result.get("success"):

                task.status = "completed"

                task.result = result

                task.completed_at = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                task.last_error = None

            else:

                self._handle_failure(
                    task,
                    result.get(
                        "error",
                        "Task execution failed.",
                    ),
                )

        except Exception as exc:

            self._handle_failure(
                task,
                str(exc),
            )

        finally:

            task.worker_id = None

            self._save()

    def _handle_failure(
        self,
        task: ScheduledTask,
        error: str,
    ) -> None:

        task.last_error = error

        if task.retries < task.max_retries:

            task.retries += 1

            retry_at = (
                datetime.now(
                    timezone.utc
                )
                + timedelta(
                    seconds=task.retry_delay_seconds
                    * task.retries
                )
            )

            task.run_at = retry_at.isoformat()
            task.status = "scheduled"

        else:

            task.status = "failed"

            task.completed_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

    # ---------------------------------------------------------
    # Background worker
    # ---------------------------------------------------------

    def _loop(self) -> None:

        while not self._stop.is_set():

            try:

                task = (
                    self._claim_due_task()
                )

                if task is not None:

                    self._execute_task(
                        task
                    )

                else:

                    self._stop.wait(
                        self.poll_seconds
                    )

            except Exception:

                self._stop.wait(
                    self.poll_seconds
                )

    def start(self) -> None:

        if (
            self._thread is not None
            and self._thread.is_alive()
        ):
            return

        self._stop.clear()

        self._thread = threading.Thread(
            target=self._loop,
            name="gene-scheduler",
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> None:

        self._stop.set()

        if self._thread is not None:

            self._thread.join(
                timeout=5
            )

        self._thread = None

    @property
    def running(self) -> bool:

        return bool(
            self._thread
            and self._thread.is_alive()
        )

    def status(self) -> dict:

        with self._lock:

            counts: dict[str, int] = {}

            for task in self.tasks.values():

                counts[
                    task.status
                ] = counts.get(
                    task.status,
                    0,
                ) + 1

        return {
            "running": self.running,
            "total": len(
                self.tasks
            ),
            "statuses": counts,
            "storage": str(
                self.path
            ),
        }
