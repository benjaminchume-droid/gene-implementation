from __future__ import annotations

import threading
import time
from datetime import datetime, timezone


class SchedulerRunner:

    def __init__(
        self,
        scheduler,
        executor,
        poll_seconds: int = 2,
    ) -> None:

        self.scheduler = scheduler
        self.executor = executor
        self.poll_seconds = poll_seconds

        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _due(run_at: str) -> bool:
        try:
            target = datetime.fromisoformat(run_at)

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

    def _run_pending(self) -> None:

        for task in self.scheduler.list():

            if task.status != "scheduled":
                continue

            if not self._due(task.run_at):
                continue

            result = self.executor.execute(
                task.action,
                str(task.payload),
            )

            if result.get("success"):
                task.status = "completed"
            else:
                task.status = "failed"

            self.scheduler._save()

    def _loop(self) -> None:

        while not self._stop.is_set():
            try:
                self._run_pending()
            except Exception:
                pass

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
                timeout=3
            )