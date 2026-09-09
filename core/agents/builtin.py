from .models import SubAgentSpec
from .registry import SubAgentRegistry


def register_builtin_subagents(
    registry: SubAgentRegistry,
) -> None:
    """
    Gene does not require predefined sub-agent roles.

    Applications may register their own SubAgentSpec instances.
    This function remains as a compatibility entry point.
    """
    return None


def register_subagent(
    registry: SubAgentRegistry,
    spec: SubAgentSpec,
) -> None:
    registry.register(spec)
