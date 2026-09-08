from __future__ import annotations

import re
from typing import Iterable

from .models import ContextItem


class ContextSelector:

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0

        return max(
            1,
            int(len(text.split()) * 1.35)
        )

    @staticmethod
    def relevance(
        query: str,
        content: str,
    ) -> float:

        query_terms = set(
            re.findall(
                r"[a-zA-Z0-9_]+",
                query.lower()
            )
        )

        content_terms = set(
            re.findall(
                r"[a-zA-Z0-9_]+",
                content.lower()
            )
        )

        if not query_terms:
            return 0.0

        overlap = len(
            query_terms & content_terms
        )

        return overlap / len(query_terms)

    def rank(
        self,
        query: str,
        items: Iterable[ContextItem],
    ) -> list[ContextItem]:

        ranked = []

        for item in items:

            relevance = self.relevance(
                query,
                item.content,
            )

            score = (
                relevance * 0.70
                + item.priority * 0.30
            )

            ranked.append(
                (score, item)
            )

        ranked.sort(
            key=lambda value: value[0],
            reverse=True,
        )

        return [
            item
            for _, item in ranked
        ]

    def select(
        self,
        query: str,
        items: Iterable[ContextItem],
        token_budget: int,
    ) -> list[ContextItem]:

        selected = []
        used = 0

        for item in self.rank(query, items):

            tokens = (
                item.token_estimate
                or self.estimate_tokens(
                    item.content
                )
            )

            if used + tokens > token_budget:
                continue

            item.token_estimate = tokens
            selected.append(item)
            used += tokens

        return selected
