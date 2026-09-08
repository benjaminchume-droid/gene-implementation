from pathlib import Path
import json
from datetime import datetime, timezone


class MemoryHook:

    def __init__(self, root="gene/memory"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.log_file = self.root / "interaction_log.jsonl"

    def record(
        self,
        user_input: str,
        plan: list,
        observations: list,
        result=None,
    ):

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_input": user_input,
            "plan": plan,
            "observations": observations,
            "result": result,
        }

        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record
