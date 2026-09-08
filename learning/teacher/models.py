from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


COMPETENCY_LEVELS = (
    "beginner",
    "foundation",
    "intermediate",
    "advanced",
    "master",
    "professional",
)


@dataclass
class TeacherMessage:

    role: str
    content: str

    timestamp: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )


@dataclass
class LearningSession:

    session_id: str
    mission: str

    competency_level: str = "beginner"
    domain: str = "general"

    teacher_provider: str = "browser"
    teacher_model: str | None = None

    source_url: str | None = None
    source_title: str | None = None

    messages: list[
        TeacherMessage
    ] = field(
        default_factory=list
    )

    screenshots: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    status: str = "raw"

    def add(
        self,
        role: str,
        content: str,
    ) -> TeacherMessage:

        message = TeacherMessage(
            role=role,
            content=content,
        )

        self.messages.append(
            message
        )

        return message

    def to_dict(self) -> dict:

        return {
            "session_id":
                self.session_id,

            "mission":
                self.mission,

            "competency_level":
                self.competency_level,

            "domain":
                self.domain,

            "teacher_provider":
                self.teacher_provider,

            "teacher_model":
                self.teacher_model,

            "source_url":
                self.source_url,

            "source_title":
                self.source_title,

            "messages": [
                asdict(message)
                for message
                in self.messages
            ],

            "screenshots":
                self.screenshots,

            "metadata":
                self.metadata,

            "status":
                self.status,
        }
