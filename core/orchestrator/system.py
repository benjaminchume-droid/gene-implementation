import uuid

from .task import Task
from .planner import GenePlanner
from .executor import GeneExecutor
from .verifier import GeneVerifier
from .memory_hook import MemoryHook

from gene.tools.runtime import ToolRuntime


class GeneOrchestrator:

    def __init__(self):

        self.runtime = ToolRuntime()

        self.planner = GenePlanner()
        self.executor = GeneExecutor(self.runtime)
        self.verifier = GeneVerifier()
        self.memory = MemoryHook()

    def capability_status(self):
        return self.runtime.capability_status()

    def inspect(self):
        return {
            "tools": self.runtime.describe_tools(),
            "planner": type(self.planner).__name__,
            "executor": type(self.executor).__name__,
            "verifier": type(self.verifier).__name__,
            "memory": type(self.memory).__name__,
        }

    def run(self, user_input: str):

        task = Task(
            id=str(uuid.uuid4()),
            user_input=user_input,
        )

        task.set_status("planning")

        task.plan = self.planner.create_plan(
            user_input,
            self.runtime.describe_tools(),
        )

        task.set_status("executing")

        for step in task.plan:

            result = self.executor.execute_step(
                step,
                user_input,
            )

            task.observations.append(result)

            verification = self.verifier.verify(result)

            task.observations.append({
                "verification": verification,
            })

            if verification.get("requires_confirmation"):
                task.set_status("awaiting_confirmation")
                task.result = result
                break

            if not verification.get("verified"):
                task.set_status("failed")
                task.error = verification.get("reason")
                break

        else:
            task.set_status("completed")

            if task.observations:
                task.result = task.observations[-2]

        self.memory.record(
            user_input=user_input,
            plan=task.plan,
            observations=task.observations,
            result=task.result,
        )

        return {
            "task_id": task.id,
            "status": task.status,
            "plan": task.plan,
            "observations": task.observations,
            "result": task.result,
            "error": task.error,
        }
