from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class VerificationResult:

    passed: bool
    score: float

    checks: list[dict[str, Any]] = field(
        default_factory=list
    )

    reasons: list[str] = field(
        default_factory=list
    )


class VerificationEngine:

    def __init__(self) -> None:
        self.checks: dict[
            str,
            Callable,
        ] = {}

    def register(
        self,
        name: str,
        check: Callable,
        *,
        replace: bool = False,
    ) -> None:

        if (
            name in self.checks
            and not replace
        ):
            raise ValueError(
                f"Verification check already exists: {name}"
            )

        self.checks[name] = check

    def verify(
        self,
        result: Any,
        expected: dict[str, Any] | None = None,
    ) -> VerificationResult:

        expected = expected or {}

        if not self.checks:

            passed = result is not None

            return VerificationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                checks=[
                    {
                        "name":
                            "result_present",
                        "passed":
                            passed,
                    }
                ],
                reasons=(
                    []
                    if passed
                    else ["no_result"]
                ),
            )

        checks = []

        for name, check in self.checks.items():

            try:

                passed = bool(
                    check(
                        result,
                        expected,
                    )
                )

                checks.append(
                    {
                        "name":
                            name,
                        "passed":
                            passed,
                    }
                )

            except Exception as exc:

                checks.append(
                    {
                        "name":
                            name,
                        "passed":
                            False,
                        "error":
                            str(exc),
                    }
                )

        passed_count = sum(
            1
            for check in checks
            if check["passed"]
        )

        score = (
            passed_count / len(checks)
            if checks
            else 0.0
        )

        reasons = [
            check["name"]
            for check in checks
            if not check["passed"]
        ]

        return VerificationResult(
            passed=score >= 1.0,
            score=score,
            checks=checks,
            reasons=reasons,
        )

    def status(self) -> dict:
        return {
            "checks":
                list(self.checks.keys()),
            "count":
                len(self.checks),
        }
