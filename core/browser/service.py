from __future__ import annotations


class BrowserService:
    """
    Browser capability interface.

    The first implementation intentionally contains no browser dependency.
    A Playwright/CDP backend can be attached later without changing Gene's
    capability contract.
    """

    def navigate(self, url: str) -> dict:
        return {
            "action": "navigate",
            "url": url,
            "status": "backend_not_configured",
        }

    def click(self, selector: str) -> dict:
        return {
            "action": "click",
            "selector": selector,
            "status": "backend_not_configured",
        }

    def type_text(self, selector: str, text: str) -> dict:
        return {
            "action": "type",
            "selector": selector,
            "text": text,
            "status": "backend_not_configured",
        }
