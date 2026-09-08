from __future__ import annotations

from typing import Any, Callable

from .base import (
    LearningExchange,
    LearningSource,
    LearningSourceInfo,
)


class CallableLearningSource(
    LearningSource
):

    def __init__(
        self,
        source_id: str,
        name: str,
        function: Callable,
        *,
        kind: str = "model",
        capabilities: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:

        self._info = LearningSourceInfo(
            id=source_id,
            kind=kind,
            name=name,
            capabilities=capabilities,
            metadata=metadata or {},
        )

        self.function = function

    @property
    def info(self) -> LearningSourceInfo:
        return self._info

    def learn(
        self,
        request: str,
        context: dict[str, Any] | None = None,
    ) -> LearningExchange:

        result = self.function(
            request,
            context or {},
        )

        if isinstance(
            result,
            LearningExchange,
        ):
            return result

        if isinstance(
            result,
            dict,
        ):

            return LearningExchange(
                source_id=self.info.id,
                request=request,
                response=str(
                    result.get(
                        "response",
                        result.get(
                            "text",
                            result.get(
                                "content",
                                "",
                            ),
                        ),
                    )
                ),
                observations=result.get(
                    "observations",
                    [],
                ),
                actions=result.get(
                    "actions",
                    [],
                ),
                metadata=result.get(
                    "metadata",
                    {},
                ),
            )

        return LearningExchange(
            source_id=self.info.id,
            request=request,
            response=str(result),
        )
