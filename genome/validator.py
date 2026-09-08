from __future__ import annotations

from .schema import Genome


class GenomeValidator:

    REQUIRED_MODULES = {
        "reasoning",
        "language",
        "memory",
        "tool_use",
    }

    @classmethod
    def validate(cls, genome: Genome) -> list[str]:
        errors: list[str] = []

        if not genome.name.strip():
            errors.append("Genome name is empty.")

        if not genome.version.strip():
            errors.append("Genome version is empty.")

        module_names = {
            module.name
            for module in genome.modules
        }

        missing = cls.REQUIRED_MODULES - module_names

        for module in sorted(missing):
            errors.append(
                f"Required genome module missing: {module}"
            )

        duplicate_names = (
            len(module_names) != len(genome.modules)
        )

        if duplicate_names:
            errors.append(
                "Genome contains duplicate module names."
            )

        return errors

    @classmethod
    def is_valid(cls, genome: Genome) -> bool:
        return not cls.validate(genome)
