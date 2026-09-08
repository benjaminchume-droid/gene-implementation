from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextBudget:
    native_model_context: int = 32_768
    effective_context_target: int = 500_000

    memory_budget: int = 12_000
    knowledge_budget: int = 12_000
    skills_budget: int = 6_000
    tools_budget: int = 8_000
    conversation_budget: int = 24_000
    system_budget: int = 4_000

    @property
    def working_budget(self) -> int:
        return (
            self.memory_budget
            + self.knowledge_budget
            + self.skills_budget
            + self.tools_budget
            + self.conversation_budget
            + self.system_budget
        )

    def validate(self) -> list[str]:
        errors = []

        if self.native_model_context <= 0:
            errors.append(
                "native_model_context must be positive."
            )

        if self.effective_context_target < self.native_model_context:
            errors.append(
                "effective_context_target must be "
                "greater than or equal to native_model_context."
            )

        budgets = {
            "memory_budget": self.memory_budget,
            "knowledge_budget": self.knowledge_budget,
            "skills_budget": self.skills_budget,
            "tools_budget": self.tools_budget,
            "conversation_budget": self.conversation_budget,
            "system_budget": self.system_budget,
        }

        for name, value in budgets.items():
            if value < 0:
                errors.append(
                    f"{name} cannot be negative."
                )

        return errors
