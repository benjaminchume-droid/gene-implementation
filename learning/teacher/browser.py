from __future__ import annotations

import asyncio

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)


class BrowserTeacher:

    def __init__(
        self,
        *,
        user_data_dir: str = (
            "gene/data/learning/"
            "browser_profile"
        ),
        headless: bool = False,
    ) -> None:

        self.user_data_dir = (
            Path(user_data_dir)
        )

        self.headless = headless

        self.playwright: Playwright | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    async def start(
        self,
    ) -> None:

        self.playwright = (
            await async_playwright()
            .start()
        )

        self.context = (
            await self.playwright.chromium
            .launch_persistent_context(
                str(
                    self.user_data_dir
                ),
                headless=self.headless,
            )
        )

        if self.context.pages:

            self.page = (
                self.context.pages[-1]
            )

        else:

            self.page = (
                await self.context.new_page()
            )

    async def stop(
        self,
    ) -> None:

        if self.context:

            await self.context.close()

        if self.playwright:

            await self.playwright.stop()

        self.page = None
        self.context = None
        self.playwright = None

    async def navigate(
        self,
        url: str,
    ) -> None:

        if self.page is None:
            raise RuntimeError(
                "Browser is not started."
            )

        await self.page.goto(
            url,
            wait_until="domcontentloaded",
        )

    async def inspect(
        self,
    ) -> dict[str, Any]:

        if self.page is None:
            raise RuntimeError(
                "Browser is not started."
            )

        title = await self.page.title()

        url = self.page.url

        return {
            "url": url,
            "title": title,
            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

    async def screenshot(
        self,
        output_path: str,
    ) -> str:

        if self.page is None:
            raise RuntimeError(
                "Browser is not started."
            )

        path = Path(
            output_path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        await self.page.screenshot(
            path=str(path),
            full_page=False,
        )

        return str(path)

    async def _find_composer(self):

        if self.page is None:
            raise RuntimeError(
                "Browser is not started."
            )

        candidates = await self.page.locator(
            "textarea, "
            "[contenteditable='true'], "
            "[role='textbox']"
        ).all()

        visible = []

        for candidate in candidates:

            try:

                if await candidate.is_visible():

                    visible.append(
                        candidate
                    )

            except Exception:
                continue

        if not visible:

            raise RuntimeError(
                "No visible chat composer "
                "was found."
            )

        # Prefer larger visible editable areas.
        best = visible[0]
        best_area = -1

        for candidate in visible:

            try:

                box = await candidate.bounding_box()

                if not box:
                    continue

                area = (
                    box["width"]
                    * box["height"]
                )

                if area > best_area:

                    best = candidate
                    best_area = area

            except Exception:
                continue

        return best

    async def _find_send_button(self):

        if self.page is None:
            raise RuntimeError(
                "Browser is not started."
            )

        selectors = [
            "button[aria-label*='Send' i]",
            "button[title*='Send' i]",
            "[role='button'][aria-label*='Send' i]",
            "[role='button'][title*='Send' i]",
            "button",
        ]

        candidates = []

        for selector in selectors:

            try:

                found = await self.page.locator(
                    selector
                ).all()

                candidates.extend(
                    found
                )

            except Exception:
                continue

        for candidate in candidates:

            try:

                if not await candidate.is_visible():
                    continue

                label = (
                    (
                        await candidate.get_attribute(
                            "aria-label"
                        )
                    )
                    or
                    (
                        await candidate.get_attribute(
                            "title"
                        )
                    )
                    or
                    (
                        await candidate.inner_text()
                    )
                    or ""
                ).strip().lower()

                if "send" in label:

                    return candidate

            except Exception:
                continue

        return None

    async def send_message(
        self,
        text: str,
    ) -> None:

        if self.page is None:
            raise RuntimeError(
                "Browser is not started."
            )

        composer = (
            await self._find_composer()
        )

        await composer.click()

        try:

            await composer.fill(
                text
            )

        except Exception:

            await self.page.keyboard.insert_text(
                text
            )

        send_button = (
            await self._find_send_button()
        )

        if send_button:

            await send_button.click()

        else:

            await self.page.keyboard.press(
                "Enter"
            )

    async def read_visible_text(
        self,
    ) -> str:

        if self.page is None:
            raise RuntimeError(
                "Browser is not started."
            )

        text = await self.page.locator(
            "body"
        ).inner_text()

        return text

    async def wait_for_response(
        self,
        *,
        before_text: str,
        timeout_ms: int = 120000,
        settle_ms: int = 1500,
    ) -> str:

        if self.page is None:
            raise RuntimeError(
                "Browser is not started."
            )

        deadline = (
            asyncio.get_event_loop()
            .time()
            + timeout_ms / 1000
        )

        last_text = before_text

        while (
            asyncio.get_event_loop()
            .time()
            < deadline
        ):

            await asyncio.sleep(
                1.0
            )

            current = (
                await self.read_visible_text()
            )

            if (
                current != before_text
                and current.strip()
            ):

                last_text = current

                await asyncio.sleep(
                    settle_ms / 1000
                )

                settled = (
                    await self.read_visible_text()
                )

                if settled == last_text:

                    return settled

                last_text = settled

        raise TimeoutError(
            "Teacher response did not "
            "settle before timeout."
        )

    async def ask(
        self,
        text: str,
        *,
        timeout_ms: int = 120000,
    ) -> dict:

        before = (
            await self.read_visible_text()
        )

        await self.send_message(
            text
        )

        after = (
            await self.wait_for_response(
                before_text=before,
                timeout_ms=timeout_ms,
            )
        )

        return {
            "prompt": text,
            "page_text_before": before,
            "page_text_after": after,
            "url": self.page.url,
        }
