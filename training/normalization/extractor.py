from __future__ import annotations

from typing import Any


class GenericTextExtractor:

    DEFAULT_TEXT_FIELDS = (
        "text",
        "content",
        "body",
        "prompt",
        "response",
        "answer",
        "instruction",
        "description",
    )

    def __init__(
        self,
        text_fields=None,
    ) -> None:

        self.text_fields = tuple(
            text_fields
            or self.DEFAULT_TEXT_FIELDS
        )

    def extract(
        self,
        record: Any,
    ) -> str:

        fragments: list[str] = []

        self._walk(
            record,
            fragments,
        )

        cleaned = []

        seen = set()

        for fragment in fragments:

            if not isinstance(
                fragment,
                str,
            ):
                continue

            value = fragment.strip()

            if not value:
                continue

            if value in seen:
                continue

            seen.add(value)
            cleaned.append(value)

        return "\n\n".join(
            cleaned
        )

    def _walk(
        self,
        value: Any,
        fragments: list[str],
        *,
        field_name: str | None = None,
    ) -> None:

        if isinstance(
            value,
            str,
        ):

            if (
                field_name is None
                or field_name
                in self.text_fields
            ):
                fragments.append(value)

            return

        if isinstance(
            value,
            list,
        ):

            for item in value:

                self._walk(
                    item,
                    fragments,
                    field_name=field_name,
                )

            return

        if not isinstance(
            value,
            dict,
        ):
            return

        for key, child in value.items():

            key_string = str(key).lower()

            if key_string in self.text_fields:

                self._walk(
                    child,
                    fragments,
                    field_name=key_string,
                )

            elif isinstance(
                child,
                (
                    dict,
                    list,
                ),
            ):

                self._walk(
                    child,
                    fragments,
                    field_name=key_string,
                )
