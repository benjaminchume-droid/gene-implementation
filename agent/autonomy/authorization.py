from __future__ import annotations

from typing import Any, Callable


class MissionAuthorization:

    def __init__(
        self,
        evaluator: Callable[
            [dict[str, Any]],
            bool,
        ] | None = None,
    ) -> None:

        self.evaluator = (
            evaluator
            or self._default
        )

    def authorize(
        self,
        action: dict[str, Any],
    ) -> bool:

        return bool(
            self.evaluator(
                action
            )
        )

    @staticmethod
    def _default(
        action: dict[str, Any],
    ) -> bool:

        return True
