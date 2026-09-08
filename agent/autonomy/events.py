from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MissionEvent:

    mission_id: str
    event: str
    data: dict[str, Any] = field(
        default_factory=dict
    )

    timestamp: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )


class MissionEventLog:

    def __init__(self) -> None:
        self.events: list[
            MissionEvent
        ] = []

    def emit(
        self,
        mission_id: str,
        event: str,
        **data,
    ) -> MissionEvent:

        item = MissionEvent(
            mission_id=mission_id,
            event=event,
            data=data,
        )

        self.events.append(
            item
        )

        return item

    def for_mission(
        self,
        mission_id: str,
    ) -> list[MissionEvent]:

        return [
            item
            for item in self.events
            if item.mission_id
            == mission_id
        ]

    def status(self) -> dict:

        return {
            "events":
                len(self.events)
        }
