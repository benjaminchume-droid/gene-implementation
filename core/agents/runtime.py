from __future__ import annotations

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


ModelCallable = Callable[[str, dict[str, Any]], Any]


class SubAgentRuntime:
    """
    Isolated execution runtime for delegated Gene tasks.

    A sub-agent receives:
      - one explicit objective
      - isolated context
      - an explicit role
      - an explicit tool allow-list
      - an optional model identifier

    It returns a structured SubAgentResult.

    The runtime deliberately does not grant unrestricted access to the
    parent Gene runtime.
    """

    def __init__(
        self,
        registry: SubAgentRegistry | None = None,
        model_runner: ModelCallable | None = None,
    ) -> None:
        self.registry = registry or SubAgentRegistry()
        self.model_runner = model_runner

    def register(self, spec: SubAgentSpec) -> None:
        self.registry.register(spec)

    def create_task(
        self,
        objective: str,
        role: str = "general",
        *,
        instructions: str = "",
        context: dict[str, Any] | None = None,
        tools: list[str] | None = None,
        model_id: str | None = None,
        max_steps: int = 20,
        timeout_seconds: float = 300.0,
        metadata: dict[str, Any] | None = None,
    ) -> SubAgentTask:
        if not objective.strip():
            raise ValueError("Sub-agent objective cannot be empty.")

        spec = self.registry.get(role)

        if spec is None:
            raise ValueError(
                f"Unknown sub-agent role: {role}"
            )

        requested_tools = tools or []

        unauthorized = [
            tool
            for tool in requested_tools
            if tool not in spec.allowed_tools
        ]

        if unauthorized:
            raise PermissionError(
                f"Sub-agent '{role}' requested unauthorized tools: "
                f"{unauthorized}"
            )

        return SubAgentTask(
            id=uuid.uuid4().hex,
            objective=objective,
            role=role,
            instructions=instructions,
            context=context or {},
            tools=requested_tools,
            model_id=model_id or spec.model_id,
            max_steps=min(max_steps, spec.max_steps),
            timeout_seconds=timeout_seconds,
            metadata=metadata or {},
        )

    def run(self, task: SubAgentTask) -> SubAgentResult:
        spec = self.registry.get(task.role)

        if spec is None:
            return SubAgentResult(
                task_id=task.id,
                status=SubAgentStatus.FAILED,
                errors=[f"Unknown sub-agent role: {task.role}"],
            )

        started = time.monotonic()

        if self.model_runner is None:
            return SubAgentResult(
                task_id=task.id,
                status=SubAgentStatus.FAILED,
                model_id=task.model_id,
                errors=[
                    "No model runner is connected to SubAgentRuntime."
                ],
            )

        if time.monotonic() - started >= task.timeout_seconds:
            return SubAgentResult(
                task_id=task.id,
                status=SubAgentStatus.CANCELLED,
                model_id=task.model_id,
                errors=["Sub-agent timeout reached before execution."],
            )

        prompt = self._build_prompt(task, spec)

        try:
            response = self.model_runner(
                prompt,
                {
                    "task_id": task.id,
                    "role": task.role,
                    "model_id": task.model_id,
                    "tools": list(task.tools),
                    "context": dict(task.context),
                    "metadata": dict(task.metadata),
                },
            )

            elapsed = time.monotonic() - started

            if elapsed > task.timeout_seconds:
                return SubAgentResult(
                    task_id=task.id,
                    status=SubAgentStatus.CANCELLED,
                    model_id=task.model_id,
                    errors=["Sub-agent timeout exceeded."],
                    metadata={"elapsed_seconds": elapsed},
                )

            output = self._extract_output(response)

            return SubAgentResult(
                task_id=task.id,
                status=SubAgentStatus.COMPLETED,
                output=output,
                model_id=task.model_id,
                steps=1,
                metadata={
                    "role": task.role,
                    "elapsed_seconds": elapsed,
                },
            )

        except Exception as exc:
            return SubAgentResult(
                task_id=task.id,
                status=SubAgentStatus.FAILED,
                model_id=task.model_id,
                errors=[f"{type(exc).__name__}: {exc}"],
            )

    @staticmethod
    def _build_prompt(
        task: SubAgentTask,
        spec: SubAgentSpec,
    ) -> str:
        context = task.context

        return (
            f"You are Gene's delegated sub-agent.\n\n"
            f"ROLE:\n{spec.role}\n\n"
            f"ROLE DESCRIPTION:\n{spec.description}\n\n"
            f"SYSTEM INSTRUCTIONS:\n{spec.system_instructions}\n\n"
            f"OBJECTIVE:\n{task.objective}\n\n"
            f"ADDITIONAL INSTRUCTIONS:\n{task.instructions}\n\n"
            f"AVAILABLE TOOLS:\n{', '.join(task.tools) or 'none'}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"Complete only the delegated objective. "
            f"Return a concise result suitable for the parent Gene "
            f"agent to synthesize."
        )

    @staticmethod
    def _extract_output(response: Any) -> str:
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        text = getattr(response, "text", None)

        if text is not None:
            return str(text)

        if isinstance(response, dict):
            return str(
                response.get("text")
                or response.get("output")
                or response
            )

        return str(response)
