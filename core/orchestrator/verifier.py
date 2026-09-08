class GeneVerifier:

    def verify(self, result: dict) -> dict:

        if not isinstance(result, dict):
            return {
                "verified": False,
                "reason": "Tool returned a non-object result.",
            }

        if result.get("success") is True:
            return {
                "verified": True,
                "reason": "Tool execution reported success.",
            }

        if result.get("requires_confirmation"):
            return {
                "verified": False,
                "requires_confirmation": True,
                "reason": result.get("error", "Confirmation required."),
            }

        return {
            "verified": False,
            "reason": result.get("error", "Unknown execution failure."),
        }
