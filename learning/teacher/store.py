from __future__ import annotations

import json

from pathlib import Path


class TeacherSessionStore:

    def __init__(
        self,
        root: str = (
            "gene/data/learning/raw/teacher"
        ),
    ) -> None:

        self.root = Path(root)

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        session,
    ) -> str:

        path = (
            self.root
            / f"{session.session_id}.json"
        )

        temporary = path.with_suffix(
            ".tmp"
        )

        temporary.write_text(
            json.dumps(
                session.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temporary.replace(
            path
        )

        return str(path)

    def append_event(
        self,
        event: dict,
    ) -> str:

        path = (
            self.root
            / "events.jsonl"
        )

        with path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )

        return str(path)
