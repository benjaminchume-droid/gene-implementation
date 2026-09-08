from __future__ import annotations

from .models import ContextBundle, ContextItem
from .budget import ContextBudget
from .selector import ContextSelector


class ContextAssembler:

    def __init__(
        self,
        budget: ContextBudget,
        selector: ContextSelector | None = None,
    ) -> None:

        self.budget = budget
        self.selector = (
            selector
            or ContextSelector()
        )

    def assemble(
        self,
        query: str,
        bundle: ContextBundle,
    ) -> list[ContextItem]:

        selected = []

        selected.extend(
            self.selector.select(
                query,
                bundle.system,
                self.budget.system_budget,
            )
        )

        selected.extend(
            self.selector.select(
                query,
                bundle.memories,
                self.budget.memory_budget,
            )
        )

        selected.extend(
            self.selector.select(
                query,
                bundle.knowledge,
                self.budget.knowledge_budget,
            )
        )

        selected.extend(
            self.selector.select(
                query,
                bundle.skills,
                self.budget.skills_budget,
            )
        )

        selected.extend(
            self.selector.select(
                query,
                bundle.tools,
                self.budget.tools_budget,
            )
        )

        selected.extend(
            self.selector.select(
                query,
                bundle.conversation,
                self.budget.conversation_budget,
            )
        )

        return selected

    def render(
        self,
        query: str,
        bundle: ContextBundle,
    ) -> str:

        items = self.assemble(
            query,
            bundle,
        )

        sections = []

        for item in items:
            sections.append(
                f"[{item.source}]\n{item.content}"
            )

        return "\n\n".join(sections)
