from __future__ import annotations

from .models import DiscoveryEnvironment


class DiscoveryRegistry:

    def __init__(self) -> None:
        self.environments = {}

    def register(
        self,
        environment: DiscoveryEnvironment,
        *,
        replace: bool = False,
    ) -> None:

        name = environment.name

        if (
            name in self.environments
            and not replace
        ):
            raise ValueError(
                f"Discovery environment already exists: {name}"
            )

        self.environments[name] = environment

    def remove(
        self,
        name: str,
    ) -> bool:

        return (
            self.environments.pop(
                name,
                None,
            )
            is not None
        )

    def get(
        self,
        name: str,
    ) -> DiscoveryEnvironment:

        try:
            return self.environments[name]
        except KeyError:
            raise KeyError(
                f"Unknown discovery environment: {name}"
            ) from None

    def environments_list(self):
        return list(
            self.environments.values()
        )

    def discover_all(self):
        resources = []

        for environment in (
            self.environments.values()
        ):
            try:
                resources.extend(
                    environment.discover()
                )
            except Exception:
                continue

        return resources

    def status(self) -> dict:

        return {
            "environment_count":
                len(self.environments),
            "environments":
                list(
                    self.environments.keys()
                ),
        }
