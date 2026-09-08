from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .registry import SkillRegistry
from .schema import Skill, SkillVersion


class SkillManager:

    def __init__(
        self,
        path: str = "gene/data/skills/skills.json",
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.registry = SkillRegistry()

        self._load()

    def _load(self) -> None:

        if not self.path.exists():
            return

        data = json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )

        for skill_data in data.get(
            "skills",
            []
        ):

            skill = Skill(
                name=skill_data["name"],
                domain=skill_data["domain"],
                description=skill_data[
                    "description"
                ],
                parent=skill_data.get(
                    "parent"
                ),
                enabled=skill_data.get(
                    "enabled",
                    True,
                ),
                level=skill_data.get(
                    "level",
                    1,
                ),
                tags=skill_data.get(
                    "tags",
                    [],
                ),
                dependencies=skill_data.get(
                    "dependencies",
                    [],
                ),
                metadata=skill_data.get(
                    "metadata",
                    {},
                ),
            )

            for version_data in skill_data.get(
                "versions",
                []
            ):

                skill.add_version(
                    SkillVersion(
                        version=version_data[
                            "version"
                        ],
                        instructions=version_data[
                            "instructions"
                        ],
                        created_at=version_data.get(
                            "created_at",
                            "",
                        ),
                        validated=version_data.get(
                            "validated",
                            False,
                        ),
                        score=version_data.get(
                            "score",
                            0.0,
                        ),
                        tests_passed=version_data.get(
                            "tests_passed",
                            0,
                        ),
                        tests_failed=version_data.get(
                            "tests_failed",
                            0,
                        ),
                    )
                )

            self.registry.register(skill)

    def save(self) -> None:

        skills = []

        for skill in self.registry.list():

            skills.append({
                "name": skill.name,
                "domain": skill.domain,
                "description":
                    skill.description,
                "parent": skill.parent,
                "enabled": skill.enabled,
                "level": skill.level,
                "tags": skill.tags,
                "dependencies":
                    skill.dependencies,
                "metadata":
                    skill.metadata,
                "versions": [
                    {
                        "version":
                            version.version,
                        "instructions":
                            version.instructions,
                        "created_at":
                            version.created_at,
                        "validated":
                            version.validated,
                        "score":
                            version.score,
                        "tests_passed":
                            version.tests_passed,
                        "tests_failed":
                            version.tests_failed,
                    }
                    for version in skill.versions
                ],
            })

        self.path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "skills": skills,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def create(
        self,
        name: str,
        domain: str,
        description: str,
        instructions: str,
        *,
        parent: str | None = None,
        level: int = 1,
        tags: list[str] | None = None,
        dependencies: list[str] | None = None,
    ) -> Skill:

        skill = Skill(
            name=name,
            domain=domain,
            description=description,
            parent=parent,
            level=level,
            tags=tags or [],
            dependencies=dependencies or [],
        )

        skill.add_version(
            SkillVersion(
                version="1.0.0",
                instructions=instructions,
            )
        )

        existing = self.registry.get(name)

        if existing is not None:
            return existing

        self.registry.register(skill)

        self.save()

        return skill

    def branch(
        self,
        parent: str,
        name: str,
        description: str,
        instructions: str,
        *,
        level: int | None = None,
        tags: list[str] | None = None,
        dependencies: list[str] | None = None,
    ) -> Skill:

        parent_skill = self.registry.get(parent)

        if parent_skill is None:
            raise KeyError(
                f"Parent skill does not exist: {parent}"
            )

        child_level = (
            level
            if level is not None
            else parent_skill.level + 1
        )

        return self.create(
            name=name,
            domain=parent_skill.domain,
            description=description,
            instructions=instructions,
            parent=parent,
            level=child_level,
            tags=tags or [],
            dependencies=dependencies or [],
        )

    def add_version(
        self,
        skill_name: str,
        instructions: str,
    ) -> SkillVersion:

        skill = self.registry.get(
            skill_name
        )

        if skill is None:
            raise KeyError(
                f"Skill does not exist: "
                f"{skill_name}"
            )

        latest = skill.latest

        if latest is None:
            version_number = "1.0.0"
        else:
            parts = latest.version.split(".")
            version_number = (
                f"{parts[0]}.{int(parts[1]) + 1}.0"
            )

        version = SkillVersion(
            version=version_number,
            instructions=instructions,
        )

        skill.add_version(version)

        self.save()

        return version

    def validate(
        self,
        skill_name: str,
        score: float,
        tests_passed: int,
        tests_failed: int,
    ) -> Skill:

        skill = self.registry.get(
            skill_name
        )

        if skill is None:
            raise KeyError(
                f"Skill does not exist: "
                f"{skill_name}"
            )

        version = skill.latest

        if version is None:
            raise ValueError(
                f"Skill has no version: "
                f"{skill_name}"
            )

        version.validate(
            score,
            tests_passed,
            tests_failed,
        )

        self.save()

        return skill

    def promote(
        self,
        skill_name: str,
    ) -> Skill:

        skill = self.registry.get(
            skill_name
        )

        if skill is None:
            raise KeyError(
                f"Skill does not exist: "
                f"{skill_name}"
            )

        if not skill.validated:
            raise ValueError(
                "Only validated skills "
                "can be promoted."
            )

        skill.level += 1

        self.save()

        return skill

    def status(self) -> dict[str, Any]:

        skills = self.registry.list()

        return {
            "total": len(skills),
            "validated": len(
                self.registry.validated()
            ),
            "roots": len(
                self.registry.tree.roots()
            ),
            "storage": str(self.path),
        }
