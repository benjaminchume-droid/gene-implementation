from __future__ import annotations

from .models import SubAgentSpec
from .registry import SubAgentRegistry


def register_builtin_subagents(
    registry: SubAgentRegistry,
) -> None:
    agents = [
        SubAgentSpec(
            role="researcher",
            description="Researches a delegated subject and produces structured findings.",
            system_instructions=(
                "Research carefully, distinguish facts from assumptions, "
                "and return evidence-oriented findings."
            ),
            capabilities=["research", "analysis"],
            allowed_tools=["web_search", "filesystem"],
            max_steps=20,
        ),
        SubAgentSpec(
            role="software_engineer",
            description="Designs and implements software components.",
            system_instructions=(
                "Work as a production software engineer. "
                "Prefer maintainable architecture, tests, and explicit interfaces."
            ),
            capabilities=["coding", "architecture", "debugging"],
            allowed_tools=["filesystem", "process"],
            max_steps=30,
        ),
        SubAgentSpec(
            role="ui_engineer",
            description="Designs user interfaces and interaction systems.",
            system_instructions=(
                "Focus on usability, information hierarchy, responsive layout, "
                "interaction states, accessibility, and implementation feasibility."
            ),
            capabilities=["ui", "ux", "frontend"],
            allowed_tools=["filesystem"],
            max_steps=20,
        ),
        SubAgentSpec(
            role="tester",
            description="Tests implementations and identifies defects.",
            system_instructions=(
                "Attempt to falsify assumptions. Identify regressions, edge cases, "
                "integration failures, and incomplete behavior."
            ),
            capabilities=["testing", "verification", "debugging"],
            allowed_tools=["filesystem", "process"],
            max_steps=25,
        ),
        SubAgentSpec(
            role="planner",
            description="Breaks large objectives into executable work packages.",
            system_instructions=(
                "Decompose objectives into dependency-aware tasks. "
                "Do not perform unauthorized execution."
            ),
            capabilities=["planning", "decomposition"],
            allowed_tools=[],
            max_steps=15,
        ),
        SubAgentSpec(
            role="general",
            description="General-purpose delegated Gene worker.",
            system_instructions=(
                "Complete the assigned objective accurately and report "
                "limitations instead of inventing results."
            ),
            capabilities=["general"],
            allowed_tools=[],
            max_steps=20,
        ),
    ]

    for spec in agents:
        if not registry.exists(spec.role):
            registry.register(spec)
