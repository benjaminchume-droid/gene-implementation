from __future__ import annotations

from .assembler import ContextAssembler
from .budget import ContextBudget
from .models import ContextBundle, ContextItem


class ContextManager:

    def __init__(
        self,
        budget: ContextBudget | None = None,
    ) -> None:

        self.budget = (
            budget
            or ContextBudget()
        )

        errors = self.budget.validate()

        if errors:
            raise ValueError(
                "Invalid context budget: "
                + "; ".join(errors)
            )

        self.assembler = (
            ContextAssembler(
                self.budget
            )
        )

    def build(
        self,
        query: str,
        *,
        system: list[str] | None = None,
        memories: list[str] | None = None,
        knowledge: list[str] | None = None,
        skills: list[str] | None = None,
        tools: list[str] | None = None,
        conversation: list[str] | None = None,
    ) -> str:

        bundle = ContextBundle(
            system=self._items(
                system or [],
                "system",
            ),
            memories=self._items(
                memories or [],
                "memory",
            ),
            knowledge=self._items(
                knowledge or [],
                "knowledge",
            ),
            skills=self._items(
                skills or [],
                "skill",
            ),
            tools=self._items(
                tools or [],
                "tool",
            ),
            conversation=self._items(
                conversation or [],
                "conversation",
            ),
        )

        return self.assembler.render(
            query,
            bundle,
        )

    @staticmethod
    def _items(
        values: list[str],
        source: str,
    ) -> list[ContextItem]:

        return [
            ContextItem(
                content=value,
                source=source,
            )
            for value in values
            if value.strip()
        ]

    def configuration(self) -> dict:
        return {
            "native_model_context":
                self.budget.native_model_context,
            "effective_context_target":
                self.budget.effective_context_target,
            "working_budget":
                self.budget.working_budget,
        }
