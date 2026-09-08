from __future__ import annotations

import json
from pathlib import Path

from .identity import Identity
from .profile import SoulProfile
from .spark import DEFAULT_SPARKS


class SoulManager:

    def __init__(
        self,
        path: str = "gene/data/soul/profile.json",
    ) -> None:

        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.profile = self._load()

    def _create_default(self) -> SoulProfile:
        profile = SoulProfile(
            identity=Identity(
                name="Gene",
                age="1",
                role="AI",
                description="Modular intelligence system.",
            )
        )

        for spark in DEFAULT_SPARKS.values():
            profile.add_spark(spark)

        profile.add_guideline(
            "Prioritize accuracy over unsupported claims."
        )

        profile.add_guideline(
            "Use available tools only within the active permission policy."
        )

        profile.add_guideline(
            "Do not claim an action succeeded unless it was verified."
        )

        profile.add_characteristic(
            "modular"
        )

        profile.add_characteristic(
            "learning-oriented"
        )

        profile.add_characteristic(
            "tool-capable"
        )

        return profile

    def _load(self) -> SoulProfile:
        if not self.path.exists():
            profile = self._create_default()
            self.save(profile)
            return profile

        data = json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )

        identity_data = data["identity"]

        profile = SoulProfile(
            identity=Identity(
                **identity_data
            )
        )

        for name, spark_data in data.get(
            "sparks",
            {}
        ).items():

            from .spark import SoulSpark

            profile.add_spark(
                SoulSpark(
                    **spark_data
                )
            )

        profile.active_sparks = data.get(
            "active_sparks",
            ["assistant"],
        )

        profile.guidelines = data.get(
            "guidelines",
            [],
        )

        profile.characteristics = data.get(
            "characteristics",
            [],
        )

        return profile

    def save(
        self,
        profile: SoulProfile | None = None,
    ) -> None:

        profile = profile or self.profile

        self.path.write_text(
            json.dumps(
                profile.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def activate_spark(
        self,
        name: str,
    ) -> None:

        self.profile.activate(name)
        self.save()

    def deactivate_spark(
        self,
        name: str,
    ) -> None:

        self.profile.deactivate(name)
        self.save()

    def identity(self) -> dict:
        return self.profile.identity.to_dict()

    def active_sparks(self) -> list[str]:
        return list(
            self.profile.active_sparks
        )

    def configuration(self) -> dict:
        return self.profile.to_dict()
