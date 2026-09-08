from __future__ import annotations

import json
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Evidence:

    id: str

    source_id: str

    request: str
    response: str

    observations: list[dict[str, Any]] = field(
        default_factory=list
    )

    actions: list[dict[str, Any]] = field(
        default_factory=list
    )

    verification: dict[str, Any] = field(
        default_factory=dict
    )

    provenance: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )


class EvidenceStore:

    def __init__(
        self,
        path: str = (
            "gene/data/learning/"
            "evidence/evidence.jsonl"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def add(
        self,
        source_id: str,
        request: str,
        response: str,
        *,
        observations=None,
        actions=None,
        verification=None,
        provenance=None,
    ) -> Evidence:

        record = Evidence(
            id=str(
                uuid.uuid4()
            ),
            source_id=source_id,
            request=request,
            response=response,
            observations=
                observations or [],
            actions=
                actions or [],
            verification=
                verification or {},
            provenance=
                provenance or {},
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                )
                + "\n"
            )

        return record
