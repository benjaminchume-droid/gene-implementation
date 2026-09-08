from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BrowserSession:
    url: str | None = None
    title: str | None = None
    active: bool = False


class BrowserService:

    def __init__(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Browser control requires playwright."
            ) from exc

        self._sync_playwright = sync_playwright
        self._playwright = None
        self.browser = None
        self.context = None
        self.page = None

        self.session = BrowserSession()

    def start(
        self,
        headless: bool = False,
    ) -> dict:

        if self.page is not None:
            return {
                "success": True,
                "status": "already_running",
                "url": self.page.url,
                "title": self.page.title(),
            }

        self._playwright = (
            self._sync_playwright().start()
        )

        self.browser = (
            self._playwright.chromium.launch(
                headless=headless
            )
        )

        self.context = (
            self.browser.new_context(
                accept_downloads=True
            )
        )

        self.page = (
            self.context.new_page()
        )

        self.session.active = True

        return {
            "success": True,
            "status": "started",
        }

    def ensure_started(
        self,
        headless: bool = False,
    ) -> None:
        if self.page is None:
            result = self.start(
                headless=headless
            )

            if not result["success"]:
                raise RuntimeError(
                    result.get(
                        "error",
                        "Unable to start browser.",
                    )
                )

    def navigate(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout: int = 30000,
    ) -> dict:

        self.ensure_started()

        response = self.page.goto(
            url,
            wait_until=wait_until,
            timeout=timeout,
        )

        self.session.url = self.page.url
        self.session.title = self.page.title()

        return {
            "success": True,
            "url": self.page.url,
            "title": self.page.title(),
            "status_code": (
                response.status
                if response is not None
                else None
            ),
        }

    def current_page(self) -> dict:
        self.ensure_started()

        return {
            "success": True,
            "url": self.page.url,
            "title": self.page.title(),
        }

    def visible_text(self) -> dict:
        self.ensure_started()

        text = self.page.locator(
            "body"
        ).inner_text()

        return {
            "success": True,
            "url": self.page.url,
            "title": self.page.title(),
            "text": text,
        }

    def page_content(self) -> dict:
        self.ensure_started()

        return {
            "success": True,
            "html": self.page.content(),
            "url": self.page.url,
        }

    def find(
        self,
        selector: str,
    ) -> dict:

        self.ensure_started()

        locator = self.page.locator(
            selector
        )

        count = locator.count()
        results = []

        for index in range(
            min(count, 100)
        ):
            item = locator.nth(index)

            try:
                results.append({
                    "index": index,
                    "text": item.inner_text(
                        timeout=2000
                    )[:1000],
                    "visible":
                        item.is_visible(),
                    "enabled":
                        item.is_enabled(),
                })
            except Exception:
                results.append({
                    "index": index,
                    "text": None,
                    "visible": False,
                    "enabled": False,
                })

        return {
            "success": True,
            "selector": selector,
            "count": count,
            "results": results,
        }

    def click(
        self,
        selector: str,
        timeout: int = 10000,
    ) -> dict:

        self.ensure_started()

        self.page.locator(
            selector
        ).click(
            timeout=timeout
        )

        return {
            "success": True,
            "action": "click",
            "selector": selector,
            "url": self.page.url,
        }

    def type(
        self,
        selector: str,
        text: str,
        clear: bool = True,
        timeout: int = 10000,
    ) -> dict:

        self.ensure_started()

        locator = self.page.locator(
            selector
        )

        if clear:
            locator.fill(
                text,
                timeout=timeout,
            )
        else:
            locator.press_sequentially(
                text,
                timeout=timeout,
            )

        return {
            "success": True,
            "action": "type",
            "selector": selector,
        }

    def press(
        self,
        key: str,
        selector: str | None = None,
    ) -> dict:

        self.ensure_started()

        if selector:
            self.page.locator(
                selector
            ).press(key)
        else:
            self.page.keyboard.press(key)

        return {
            "success": True,
            "action": "press",
            "key": key,
        }

    def scroll(
        self,
        amount: int,
    ) -> dict:

        self.ensure_started()

        self.page.mouse.wheel(
            0,
            amount,
        )

        return {
            "success": True,
            "action": "scroll",
            "amount": amount,
        }

    def screenshot(
        self,
        output: str = (
            "gene/data/browser/latest.png"
        ),
        full_page: bool = False,
    ) -> dict:

        self.ensure_started()

        target = Path(output)
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.page.screenshot(
            path=str(target),
            full_page=full_page,
        )

        return {
            "success": True,
            "path": str(target.resolve()),
            "url": self.page.url,
        }

    def download(
        self,
        selector: str,
        output_dir: str = (
            "gene/data/browser/downloads"
        ),
    ) -> dict:

        self.ensure_started()

        directory = Path(output_dir)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.page.expect_download() as info:
            self.page.locator(
                selector
            ).click()

        download = info.value
        destination = (
            directory
            / download.suggested_filename
        )

        download.save_as(
            str(destination)
        )

        return {
            "success": True,
            "path":
                str(destination.resolve()),
            "filename":
                download.suggested_filename,
        }

    def close(self) -> dict:

        if self.browser is None:
            return {
                "success": True,
                "status": "already_closed",
            }

        try:
            self.browser.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()

            self.browser = None
            self.context = None
            self.page = None
            self._playwright = None
            self.session = BrowserSession()

        return {
            "success": True,
            "status": "closed",
        }


    def status(self) -> dict:
        return {
            "running": self.page is not None,
            "page": self.page is not None,
            "browser": self.browser is not None,
            "playwright": self._playwright is not None,
        }

    def stop(self) -> dict:
        errors = []

        for attribute in (
            "page",
            "context",
            "browser",
            "_playwright",
        ):
            value = getattr(
                self,
                attribute,
                None,
            )

            if value is None:
                continue

            try:
                if attribute == "page":
                    value.close()
                elif attribute == "context":
                    value.close()
                elif attribute == "browser":
                    value.close()
                elif attribute == "_playwright":
                    value.stop()
            except Exception as exc:
                errors.append(str(exc))

            setattr(
                self,
                attribute,
                None,
            )

        return {
            "success": not errors,
            "errors": errors,
        }
