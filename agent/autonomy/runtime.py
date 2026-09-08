from __future__ import annotations

from typing import Any, Callable

from .authorization import (
    MissionAuthorization,
)
from .events import (
    MissionEventLog,
)
from .models import (
    Mission,
    MissionState,
)
from .recovery import (
    RecoveryEngine,
)
from .store import (
    MissionStore,
)


class AgentAutonomyRuntime:

    def __init__(
        self,
        *,
        discovery=None,
        composer=None,
        executor=None,
        learning=None,
        promotion=None,
        authorization=None,
        planner: Callable[
            [str, dict[str, Any]],
            list[str],
        ] | None = None,
        store: MissionStore | None = None,
    ) -> None:

        self.discovery = discovery
        self.composer = composer
        self.executor = executor
        self.learning = learning
        self.promotion = promotion

        self.authorization = (
            authorization
            or MissionAuthorization()
        )

        self.recovery = RecoveryEngine(
            planner=planner
        )

        self.events = MissionEventLog()

        self.store = (
            store
            or MissionStore()
        )

    def run(
        self,
        mission: Mission,
        *,
        max_attempts: int = 3,
    ) -> Mission:

        if max_attempts <= 0:
            raise ValueError(
                "max_attempts must be positive."
            )

        mission.attempts += 1

        try:

            return self._run(
                mission,
                max_attempts=max_attempts,
            )

        except Exception as exc:

            mission.errors.append(
                {
                    "type":
                        type(exc).__name__,
                    "message":
                        str(exc),
                }
            )

            mission.transition(
                MissionState.FAILED
            )

            self.events.emit(
                mission.id,
                "mission_failed",
                error=str(exc),
            )

            self.store.save(
                mission
            )

            return mission

    def _run(
        self,
        mission: Mission,
        *,
        max_attempts: int,
    ) -> Mission:

        # ------------------------------
        # DISCOVERY
        # ------------------------------

        mission.transition(
            MissionState.DISCOVERING
        )

        self.events.emit(
            mission.id,
            "discovery_started",
            objective=
                mission.objective,
        )

        if self.discovery is not None:

            from gene.discovery import (
                DiscoveryMission,
            )

            mission.discovered = (
                self.discovery.discover(
                    DiscoveryMission(
                        objective=
                            mission.objective,
                        constraints=
                            mission.context,
                        metadata=
                            mission.metadata,
                    )
                )
            )

        self.events.emit(
            mission.id,
            "discovery_completed",
            resources=len(
                mission.discovered.get(
                    "resources",
                    [],
                )
            ),
        )

        # ------------------------------
        # PLANNING
        # ------------------------------

        mission.transition(
            MissionState.PLANNING
        )

        self.events.emit(
            mission.id,
            "planning_started",
        )

        if self.composer is None:
            raise RuntimeError(
                "No capability composer is connected."
            )

        if not hasattr(
            self,
            "_planner",
        ):
            raise RuntimeError(
                "No planner was supplied."
            )

        requested_steps = self._planner(
            mission.objective,
            {
                "mission":
                    mission,
                "discovery":
                    mission.discovered,
            },
        )

        composition = (
            self.composer.compose(
                objective=
                    mission.objective,
                requested_steps=
                    requested_steps,
            )
        )

        if not composition.success:

            mission.transition(
                MissionState.BLOCKED
            )

            mission.result = {
                "unresolved":
                    composition.unresolved_steps,
            }

            self.events.emit(
                mission.id,
                "mission_blocked",
                unresolved=
                    composition.unresolved_steps,
            )

            self.store.save(
                mission
            )

            return mission

        mission.plan = (
            composition.plan
        )

        self.events.emit(
            mission.id,
            "planning_completed",
            capabilities=
                composition.capabilities_used,
        )

        # ------------------------------
        # ATTEMPT / RECOVERY LOOP
        # ------------------------------

        for attempt in range(
            max_attempts
        ):

            mission.attempts = (
                attempt + 1
            )

            mission.transition(
                MissionState.EXECUTING
            )

            self.events.emit(
                mission.id,
                "execution_started",
                attempt=
                    mission.attempts,
            )

            if self.executor is None:
                raise RuntimeError(
                    "No plan executor is connected."
                )

            result = self.executor.execute(
                mission.plan,
                context={
                    "mission_id":
                        mission.id,
                    **mission.context,
                },
            )

            mission.result = result

            if result.status == (
                "completed"
            ):

                self.events.emit(
                    mission.id,
                    "execution_completed",
                )

                break

            self.events.emit(
                mission.id,
                "execution_failed",
                result=
                    result.result,
            )

            if attempt + 1 >= max_attempts:

                mission.transition(
                    MissionState.FAILED
                )

                break

            mission.transition(
                MissionState.RECOVERING
            )

            decision = (
                self.recovery.recover(
                    mission.objective,
                    {
                        "mission_id":
                            mission.id,
                        "attempt":
                            attempt + 1,
                        "result":
                            result.result,
                    },
                )
            )

            self.events.emit(
                mission.id,
                "recovery_decision",
                retry=
                    decision.retry,
                steps=
                    decision.replanned_steps,
            )

            if not decision.retry:
                mission.transition(
                    MissionState.FAILED
                )
                break

            mission.transition(
                MissionState.PLANNING
            )

            composition = (
                self.composer.compose(
                    objective=
                        mission.objective,
                    requested_steps=
                        decision.replanned_steps,
                )
            )

            if not composition.success:
                mission.transition(
                    MissionState.FAILED
                )
                break

            mission.plan = (
                composition.plan
            )

        # ------------------------------
        # LEARNING
        # ------------------------------

        mission.transition(
            MissionState.LEARNING
        )

        if self.learning is not None:

            try:

                from gene.learning.loop import (
                    Experience,
                )

                success = (
                    mission.state
                    == MissionState.LEARNING
                    and mission.result is not None
                    and getattr(
                        mission.result,
                        "status",
                        None,
                    )
                    == "completed"
                )

                experience = (
                    Experience.create(
                        objective=
                            mission.objective,
                        outcome=
                            (
                                str(
                                    mission.result.result
                                )
                                if mission.result
                                else "No result."
                            ),
                        success=success,
                        metadata={
                            "mission_id":
                                mission.id,
                            "attempts":
                                mission.attempts,
                        },
                    )
                )

                mission.learning = (
                    self.learning.record(
                        experience
                    )
                )

            except Exception as exc:

                mission.learning = {
                    "success":
                        False,
                    "error":
                        str(exc),
                }

        # ------------------------------
        # FINAL STATE
        # ------------------------------

        completed = (
            mission.result is not None
            and getattr(
                mission.result,
                "status",
                None,
            )
            == "completed"
        )

        mission.transition(
            MissionState.COMPLETED
            if completed
            else MissionState.FAILED
        )

        self.events.emit(
            mission.id,
            (
                "mission_completed"
                if completed
                else "mission_failed"
            ),
        )

        self.store.save(
            mission
        )

        return mission

    def configure_planner(
        self,
        planner: Callable[
            [str, dict[str, Any]],
            list[str],
        ],
    ) -> None:

        self._planner = planner

    def status(self) -> dict:

        return {
            "missions":
                self.store.status(),
            "events":
                self.events.status(),
            "has_discovery":
                self.discovery is not None,
            "has_composer":
                self.composer is not None,
            "has_executor":
                self.executor is not None,
            "has_learning":
                self.learning is not None,
            "has_promotion":
                self.promotion is not None,
        }
