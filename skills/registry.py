from __future__ import annotations

from .schema import Skill
from .tree import SkillTree


class SkillRegistry:

    def __init__(self) -> None:
        self.tree = SkillTree()

    def register(
        self,
        skill: Skill,
    ) -> None:

        self.tree.add(skill)

    def get(
        self,
        name: str,
    ) -> Skill | None:

        return self.tree.get(name)

    def exists(
        self,
        name: str,
    ) -> bool:

        return self.get(name) is not None

    def list(
        self,
    ) -> list[Skill]:

        return list(
            self.tree.skills.values()
        )

    def validated(
        self,
    ) -> list[Skill]:

        return self.tree.validated()
