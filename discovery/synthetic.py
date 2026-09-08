from .models import (
    DiscoveryEnvironment,
    Observation,
    Resource,
)

class SyntheticEnvironment(
    DiscoveryEnvironment
):

    def __init__(
        self,
        name: str,
        resources: list[Resource],
    ) -> None:

        self._name = name
        self.resources = resources

    @property
    def name(self) -> str:
        return self._name

    def discover(self) -> list[Resource]:
        return list(self.resources)

    def observe(
        self,
        resource: Resource,
    ) -> Observation:

        return Observation(
            resource_id=resource.id,
            data={
                "discovered": True,
                "actions": resource.actions,
                "metadata": resource.metadata,
            },
            source=self.name,
            confidence=1.0,
        )

    def execute(
        self,
        resource: Resource,
        action: dict,
    ) -> dict:

        return {
            "success": True,
            "resource": resource.id,
            "action": action,
        }
