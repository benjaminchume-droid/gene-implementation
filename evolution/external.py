from __future__ import annotations

import copy
import json
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ParameterVersion:
    id: str
    version: int
    parameters: dict[str, Any]
    score: float
    status: str
    created_at: str


@dataclass
class ParameterProposal:
    id: str
    namespace: str
    changes: dict[str, Any]
    reason: str
    evidence: list[str] = field(default_factory=list)
    status: str = "proposed"
    created_at: str = field(
        default_factory=lambda:
        datetime.now(timezone.utc).isoformat()
    )


class ExternalParameterStore:

    def __init__(
        self,
        path: str = (
            "gene/data/evolution/parameters.json"
        ),
    ) -> None:

        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.active: dict[str, dict[str, Any]] = {}
        self.versions: dict[
            str,
            list[ParameterVersion]
        ] = {}

        self._load()

    def _load(self) -> None:

        if not self.path.exists():
            return

        raw = self.path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not raw:
            return

        data = json.loads(raw)

        self.active = data.get(
            "active",
            {},
        )

        self.versions = {}

        for namespace, values in data.get(
            "versions",
            {},
        ).items():

            self.versions[namespace] = [
                ParameterVersion(**item)
                for item in values
            ]

    def _save(self) -> None:

        payload = {
            "active": self.active,
            "versions": {
                namespace: [
                    asdict(version)
                    for version
                    in versions
                ]
                for namespace, versions
                in self.versions.items()
            },
        }

        tmp = self.path.with_suffix(
            ".tmp"
        )

        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        tmp.replace(self.path)

    def get(
        self,
        namespace: str,
    ) -> dict[str, Any]:

        return copy.deepcopy(
            self.active.get(
                namespace,
                {},
            )
        )

    def snapshot(self) -> dict[str, dict]:

        return copy.deepcopy(
            self.active
        )

    def latest_version(
        self,
        namespace: str,
    ) -> ParameterVersion | None:

        versions = self.versions.get(
            namespace,
            [],
        )

        return (
            versions[-1]
            if versions
            else None
        )

    def history(
        self,
        namespace: str,
    ) -> list[ParameterVersion]:

        return list(
            self.versions.get(
                namespace,
                [],
            )
        )

    def set(
        self,
        namespace: str,
        parameters: dict[str, Any],
        score: float = 0.0,
        source: str = "system",
    ) -> ParameterVersion:

        previous = self.versions.get(
            namespace,
            [],
        )

        version_number = (
            previous[-1].version + 1
            if previous
            else 1
        )

        record = ParameterVersion(
            id=str(uuid.uuid4()),
            version=version_number,
            parameters=copy.deepcopy(
                parameters
            ),
            score=max(
                0.0,
                min(1.0, score),
            ),
            status="active",
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        for old in previous:
            old.status = "inactive"

        self.active[namespace] = copy.deepcopy(
            parameters
        )

        self.versions.setdefault(
            namespace,
            [],
        ).append(record)

        self._save()

        return record

    def activate_version(
        self,
        namespace: str,
        version: int,
    ) -> ParameterVersion:

        versions = self.versions.get(
            namespace,
            [],
        )

        target = next(
            (
                item
                for item in versions
                if item.version == version
            ),
            None,
        )

        if target is None:
            raise KeyError(
                f"Unknown parameter version "
                f"{namespace}:{version}"
            )

        for item in versions:
            item.status = (
                "active"
                if item.version == version
                else "inactive"
            )

        self.active[namespace] = copy.deepcopy(
            target.parameters
        )

        self._save()

        return target

    def rollback(
        self,
        namespace: str,
        version: int,
    ) -> ParameterVersion:

        return self.activate_version(
            namespace,
            version,
        )

    def status(self) -> dict:

        return {
            "namespaces": list(
                self.active.keys()
            ),
            "versions": {
                namespace: len(values)
                for namespace, values
                in self.versions.items()
            },
            "storage": str(
                self.path
            ),
            "neural_weights_mutable": False,
        }


class EvolutionController:

    def __init__(
        self,
        store: ExternalParameterStore | None = None,
    ) -> None:

        self.store = (
            store
            or ExternalParameterStore()
        )

        self.audit_path = Path(
            "gene/data/evolution/audit.jsonl"
        )

        self.audit_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _audit(
        self,
        event: dict[str, Any],
    ) -> None:

        with self.audit_path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    {
                        "timestamp":
                            datetime.now(
                                timezone.utc
                            ).isoformat(),
                        **event,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def propose(
        self,
        namespace: str,
        changes: dict[str, Any],
        reason: str,
        evidence: list[str] | None = None,
    ) -> ParameterProposal:

        proposal = ParameterProposal(
            id=str(uuid.uuid4()),
            namespace=namespace,
            changes=copy.deepcopy(changes),
            reason=reason,
            evidence=evidence or [],
        )

        self._audit({
            "event": "proposal_created",
            "proposal": asdict(proposal),
        })

        return proposal

    def candidate_parameters(
        self,
        proposal: ParameterProposal,
    ) -> dict[str, Any]:

        current = self.store.get(
            proposal.namespace
        )

        return {
            **current,
            **proposal.changes,
        }

    def evaluate_and_activate(
        self,
        proposal: ParameterProposal,
        score: float,
        minimum_score: float = 0.80,
    ) -> dict[str, Any]:
        """
        Backward-compatible API for callers that use the
        original EvolutionController interface.
        """
        return self.activate(
            proposal=proposal,
            score=score,
            minimum_score=minimum_score,
        )

    def activate(
        self,
        proposal: ParameterProposal,
        score: float,
        minimum_score: float = 0.80,
    ) -> dict[str, Any]:

        if not 0 <= score <= 1:
            raise ValueError(
                "score must be between 0 and 1"
            )

        if score < minimum_score:

            proposal.status = "rejected"

            result = {
                "success": False,
                "status": "rejected",
                "proposal_id": proposal.id,
                "score": score,
                "reason":
                    "Candidate did not meet "
                    "the minimum evaluation score.",
                "neural_weights_modified": False,
            }

            self._audit({
                "event": "proposal_rejected",
                **result,
            })

            return result

        candidate = (
            self.candidate_parameters(
                proposal
            )
        )

        version = self.store.set(
            proposal.namespace,
            candidate,
            score=score,
        )

        proposal.status = "activated"

        result = {
            "success": True,
            "status": "activated",
            "proposal_id": proposal.id,
            "namespace":
                proposal.namespace,
            "version": version.version,
            "score": version.score,
            "neural_weights_modified": False,
        }

        self._audit({
            "event": "proposal_activated",
            **result,
        })

        return result

    def rollback(
        self,
        namespace: str,
        version: int,
    ) -> dict[str, Any]:

        activated = self.store.rollback(
            namespace,
            version,
        )

        result = {
            "success": True,
            "status": "rolled_back",
            "namespace": namespace,
            "version": activated.version,
            "neural_weights_modified": False,
        }

        self._audit({
            "event": "rollback",
            **result,
        })

        return result
