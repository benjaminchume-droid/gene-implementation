import time
import uuid
from typing import Any, Callable

from .models import (
    SubAgentResult,
    SubAgentSpec,
    SubAgentStatus,
    SubAgentTask,
)
from .registry import SubAgentRegistry


ModelRunner = Callable[[dict[str, Any]], Any]


class SubAgentRuntime:

    def __init__(
        self,
        registry: SubAgentRegistry | None = None,
        model_runner: ModelRunner | None = None,
    ) -> None:
        self.registry = registry or SubAgentRegistry()
        self.model_runner = model_runner

    def register(
        self,
        spec: SubAgentSpec,
        *,
        replace: bool = False,
    ) -> None:
        self.registry.register(spec, replace=replace)

    def create(
        self,
        *,
        name: str,
        objective: str,
        description: str = "",
        instructions: str = "",
        capabilities: list[str] | None = None,
        tools: list[str] | None = None,
        context: dict[str, Any] | None = None,
        model_id: str | None = None,
        max_steps: int = 20,
        timeout_seconds: float = 300.0,
        metadata: dict[str, Any] | None = None,
    ) -> SubAgentTask:

        if not name.strip():
            raise ValueError("Sub-agent name cannot be empty.")

        if not objective.strip():
            raise ValueError("Sub-agent objective cannot be empty.")

        spec = SubAgentSpec(
            name=name,
            description=description,
            instructions=instructions,
            capabilities=list(capabilities or []),
            allowed_tools=list(tools or []),
            model_id=model_id,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            metadata=dict(metadata or {}),
        )

        self.register(spec, replace=True)

        return SubAgentTask(
            id=uuid.uuid4().hex,
            agent_name=name,
            objective=objective,
            context=dict(context or {}),
            instructions=instructions,
            tools=list(tools or []),
            model_id=model_id,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            metadata=dict(metadata or {}),
        )

    def create_from_spec(
        self,
        spec: SubAgentSpec,
        objective: str,
        *,
        context: dict[str, Any] | None = None,
        tools: list[str] | None = None,
    ) -> SubAgentTask:

        self.register(spec, replace=True)

        requested_tools = list(tools or spec.allowed_tools)

        unauthorized = [
            tool
            for tool in requested_tools
            if tool not in spec.allowed_tools
        ]

        if unauthorized:
            raise PermissionError(
                f"Unauthorized tools for '{spec.name}': {unauthorized}"
            )

        return SubAgentTask(
            id=uuid.uuid4().hex,
            agent_name=spec.name,
            objective=objective,
            context=dict(context or {}),
            instructions=spec.instructions,
            tools=requested_tools,
            model_id=spec.model_id,
            max_steps=spec.max_steps,
            timeout_seconds=spec.timeout_seconds,
            metadata=dict(spec.metadata),
        )

    def run(self, task: SubAgentTask) -> SubAgentResult:

        spec = self.registry.get(task.agent_name)

        if spec is None:
            return SubAgentResult(
                task_id=task.id,
                status=SubAgentStatus.FAILED,
                errors=[f"Unknown sub-agent: {task.agent_name}"],
            )

        if self.model_runner is None:
            return SubAgentResult(
                task_id=task.id,
                status=SubAgentStatus.FAILED,
                model_id=task.model_id,
                errors=["No model runner is connected."],
            )

        started = time.monotonic()
        observations: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []

        state = {
            "agent": {
                "name": spec.name,
                "description": spec.description,
                "instructions": spec.instructions,
                "capabilities": list(spec.capabilities),
            },
            "objective": task.objective,
            "context": dict(task.context),
            "tools": list(task.tools),
            "model_id": task.model_id,
            "observations": observations,
            "artifacts": artifacts,
            "metadata": dict(task.metadata),
        }

        try:
            for step in range(1, task.max_steps + 1):

                if time.monotonic() - started >= task.timeout_seconds:
                    return SubAgentResult(
                        task_id=task.id,
                        status=SubAgentStatus.CANCELLED,
                        model_id=task.model_id,
                        observations=observations,
                        artifacts=artifacts,
                        steps=step - 1,
                        errors=["Sub-agent timeout reached."],
                    )

                state["step"] = step

                response = self.model_runner(state)

                if response is None:
                    return SubAgentResult(
                        task_id=task.id,
                        status=SubAgentStatus.FAILED,
                        model_id=task.model_id,
                        observations=observations,
                        artifacts=artifacts,
                        steps=step,
                        errors=["Model runner returned no response."],
                    )

                if isinstance(response, str):
                    return SubAgentResult(
                        task_id=task.id,
                        status=SubAgentStatus.COMPLETED,
                        output=response,
                        model_id=task.model_id,
                        observations=observations,
                        artifacts=artifacts,
                        steps=step,
                    )

                if not isinstance(response, dict):
                    response = {
                        "type": "final",
                        "output": str(response),
                    }

                decision_type = response.get("type", "final")

                if decision_type == "final":
                    return SubAgentResult(
                        task_id=task.id,
                        status=SubAgentStatus.COMPLETED,
                        output=str(response.get("output", "")),
                        model_id=task.model_id,
                        observations=observations,
                        artifacts=artifacts,
                        steps=step,
                    )

                if decision_type == "observation":
                    observation = response.get("observation")

                    if observation is not None:
                        observations.append({
                            "step": step,
                            "value": observation,
                        })

                    state["observations"] = observations
                    continue

                if decision_type == "artifact":
                    artifact = response.get("artifact")

                    if artifact is not None:
                        artifacts.append({
                            "step": step,
                            "value": artifact,
                        })

                    state["artifacts"] = artifacts
                    continue

                if decision_type == "continue":
                    continue

                return SubAgentResult(
                    task_id=task.id,
                    status=SubAgentStatus.FAILED,
                    model_id=task.model_id,
                    observations=observations,
                    artifacts=artifacts,
                    steps=step,
                    errors=[
                        f"Unsupported sub-agent decision: {decision_type}"
                    ],
                )

            return SubAgentResult(
                task_id=task.id,
                status=SubAgentStatus.CANCELLED,
                model_id=task.model_id,
                observations=observations,
                artifacts=artifacts,
                steps=task.max_steps,
                errors=["Sub-agent maximum step limit reached."],
            )

        except Exception as exc:
            return SubAgentResult(
                task_id=task.id,
                status=SubAgentStatus.FAILED,
                model_id=task.model_id,
                observations=observations,
                artifacts=artifacts,
                errors=[f"{type(exc).__name__}: {exc}"],
            )
