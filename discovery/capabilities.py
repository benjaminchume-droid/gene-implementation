from __future__ import annotations

import json
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CapabilityCandidate:

    id: str

    objective: str

    resource_id: str
    resource_kind: str

    capability: str

    procedure: list[dict[str, Any]] = field(
        default_factory=list
    )

    evidence_ids: list[str] = field(
        default_factory=list
    )

    confidence: float = 0.0

    verified: bool = False
    reusable: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda:
        datetime.now(
            timezone.utc
        ).isoformat()
    )


class CapabilityStore:

    def __init__(
        self,
        path: str = (
            "gene/data/discovery/"
            "capabilities/registry.jsonl"
        ),
    ) -> None:

        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def add(
        self,
        objective: str,
        resource_id: str,
        resource_kind: str,
        capability: str,
        *,
        procedure=None,
        evidence_ids=None,
        confidence: float = 0.0,
        verified: bool = False,
        reusable: bool = False,
        metadata=None,
    ) -> CapabilityCandidate:

        candidate = CapabilityCandidate(
            id=str(uuid.uuid4()),
            objective=objective,
            resource_id=resource_id,
            resource_kind=resource_kind,
            capability=capability,
            procedure=procedure or [],
            evidence_ids=evidence_ids or [],
            confidence=confidence,
            verified=verified,
            reusable=reusable,
            metadata=metadata or {},
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    asdict(candidate),
                    ensure_ascii=False,
                )
                + "\n"
            )

        return candidate

    def list(self) -> list[dict]:

        if not self.path.exists():
            return []

        records = []

        with self.path.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:

            for line in handle:

                if line.strip():
                    records.append(
                        json.loads(line)
                    )

        return records

    def reusable(self) -> list[dict]:

        return [
            record
            for record in self.list()
            if record.get(
                "reusable"
            ) is True
        ]

    def status(self) -> dict:

        records = self.list()

        return {
            "total":
                len(records),
            "verified":
                sum(
                    1
                    for record
                    in records
                    if record.get(
                        "verified"
                    )
                ),
            "reusable":
                sum(
                    1
                    for record
                    in records
                    if record.get(
                        "reusable"
                    )
                ),
            "storage":
                str(self.path),
        }
