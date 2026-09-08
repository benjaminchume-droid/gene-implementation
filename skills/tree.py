from __future__ import annotations

from .schema import Skill


class SkillTree:

    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}

    def add(self, skill: Skill) -> None:

        if skill.name in self.skills:
            raise ValueError(
                f"Skill already exists: {skill.name}"
            )

        if skill.parent:
            if skill.parent not in self.skills:
                raise ValueError(
                    f"Parent skill does not exist: "
                    f"{skill.parent}"
                )

        self.skills[skill.name] = skill

    def get(
        self,
        name: str,
    ) -> Skill | None:

        return self.skills.get(name)

    def children(
        self,
        parent: str,
    ) -> list[Skill]:

        return [
            skill
            for skill in self.skills.values()
            if skill.parent == parent
        ]

    def descendants(
        self,
        parent: str,
    ) -> list[Skill]:

        result = []

        queue = list(
            self.children(parent)
        )

        while queue:

            current = queue.pop(0)

            result.append(current)

            queue.extend(
                self.children(current.name)
            )

        return result

    def roots(self) -> list[Skill]:

        return [
            skill
            for skill in self.skills.values()
            if skill.parent is None
        ]

    def validated(self) -> list[Skill]:

        return [
            skill
            for skill in self.skills.values()
            if skill.validated
        ]
