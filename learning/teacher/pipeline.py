from __future__ import annotations

import hashlib
import json
import re

from pathlib import Path


class TeacherDataPipeline:

    def __init__(
        self,
        *,
        raw_dir: str = (
            "gene/data/learning/raw/teacher"
        ),
        normalized_dir: str = (
            "gene/data/learning/"
            "normalized/teacher"
        ),
        verified_dir: str = (
            "gene/data/learning/"
            "verified/teacher"
        ),
        training_dir: str = (
            "gene/data/learning/"
            "training/teacher"
        ),
    ) -> None:

        self.raw_dir = Path(
            raw_dir
        )

        self.normalized_dir = Path(
            normalized_dir
        )

        self.verified_dir = Path(
            verified_dir
        )

        self.training_dir = Path(
            training_dir
        )

        for path in (
            self.normalized_dir,
            self.verified_dir,
            self.training_dir,
        ):

            path.mkdir(
                parents=True,
                exist_ok=True,
            )

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def content_hash(
        text: str,
    ) -> str:

        return hashlib.sha256(
            text.encode(
                "utf-8"
            )
        ).hexdigest()

    def normalize_session(
        self,
        raw_path: str,
    ) -> str:

        source = Path(
            raw_path
        )

        payload = json.loads(
            source.read_text(
                encoding="utf-8-sig"
            )
        )

        messages = []

        for message in payload.get(
            "messages",
            [],
        ):

            content = self.normalize_text(
                message.get(
                    "content",
                    "",
                )
            )

            if not content:
                continue

            messages.append(
                {
                    "role":
                        message.get(
                            "role"
                        ),
                    "content":
                        content,
                    "timestamp":
                        message.get(
                            "timestamp"
                        ),
                }
            )

        canonical_text = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
        )

        output = (
            self.normalized_dir
            / source.name
        )

        normalized = {
            "session_id":
                payload.get(
                    "session_id"
                ),

            "mission":
                payload.get(
                    "mission"
                ),

            "domain":
                payload.get(
                    "domain",
                    "general",
                ),

            "competency_level":
                payload.get(
                    "competency_level",
                    "beginner",
                ),

            "teacher_provider":
                payload.get(
                    "teacher_provider",
                    "browser",
                ),

            "teacher_model":
                payload.get(
                    "teacher_model"
                ),

            "source_url":
                payload.get(
                    "source_url"
                ),

            "source_title":
                payload.get(
                    "source_title"
                ),

            "messages":
                messages,

            "screenshots":
                payload.get(
                    "screenshots",
                    [],
                ),

            "content_hash":
                self.content_hash(
                    canonical_text
                ),

            "quality":
                {
                    "status":
                        "candidate",
                    "duplicate":
                        False,
                },
        }

        output.write_text(
            json.dumps(
                normalized,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return str(output)

    def verify_session(
        self,
        normalized_path: str,
        *,
        verified_by: str = "pipeline",
    ) -> str:

        source = Path(
            normalized_path
        )

        payload = json.loads(
            source.read_text(
                encoding="utf-8-sig"
            )
        )

        messages = payload.get(
            "messages",
            [],
        )

        assistant_messages = [
            item
            for item in messages
            if item.get(
                "role"
            ) == "assistant"
        ]

        user_messages = [
            item
            for item in messages
            if item.get(
                "role"
            ) == "user"
        ]

        # Initial objective gate. This does not claim
        # semantic correctness; it only prevents empty,
        # malformed, or unusable records from entering
        # the verified layer.
        usable = (
            bool(
                user_messages
            )
            and bool(
                assistant_messages
            )
            and all(
                item.get(
                    "content",
                    ""
                ).strip()
                for item in messages
            )
        )

        payload["quality"] = {
            "status":
                "verified_candidate"
                if usable
                else "rejected",

            "verified_by":
                verified_by,

            "has_user_turn":
                bool(user_messages),

            "has_assistant_turn":
                bool(assistant_messages),

            "training_eligible":
                usable,
        }

        output = (
            self.verified_dir
            / source.name
        )

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return str(output)

    def convert_to_sft(
        self,
        verified_path: str,
    ) -> str:

        source = Path(
            verified_path
        )

        payload = json.loads(
            source.read_text(
                encoding="utf-8-sig"
            )
        )

        if not payload.get(
            "quality",
            {}
        ).get(
            "training_eligible",
            False,
        ):

            raise ValueError(
                "Verified session is not "
                "training eligible."
            )

        output = (
            self.training_dir
            / (
                source.stem
                + "_sft.jsonl"
            )
        )

        messages = payload.get(
            "messages",
            [],
        )

        record = {
            "messages":
                messages,

            "metadata": {
                "session_id":
                    payload.get(
                        "session_id"
                    ),

                "mission":
                    payload.get(
                        "mission"
                    ),

                "domain":
                    payload.get(
                        "domain"
                    ),

                "competency_level":
                    payload.get(
                        "competency_level"
                    ),

                "teacher_provider":
                    payload.get(
                        "teacher_provider"
                    ),

                "teacher_model":
                    payload.get(
                        "teacher_model"
                    ),

                "source_url":
                    payload.get(
                        "source_url"
                    ),

                "content_hash":
                    payload.get(
                        "content_hash"
                    ),
            },
        }

        with output.open(
            "w",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        return str(output)
