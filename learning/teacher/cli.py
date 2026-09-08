from __future__ import annotations

import argparse
import asyncio
import uuid

from .browser import (
    BrowserTeacher,
)

from .models import (
    COMPETENCY_LEVELS,
    LearningSession,
)

from .pipeline import (
    TeacherDataPipeline,
)

from .store import (
    TeacherSessionStore,
)


async def run():

    parser = argparse.ArgumentParser(
        description=(
            "Gene browser-based teacher learning."
        )
    )

    parser.add_argument(
        "--url",
        required=True,
        help=(
            "URL of the browser teacher."
        ),
    )

    parser.add_argument(
        "--mission",
        required=True,
        help=(
            "What Gene is supposed to learn."
        ),
    )

    parser.add_argument(
        "--domain",
        default="general",
    )

    parser.add_argument(
        "--level",
        choices=COMPETENCY_LEVELS,
        default="beginner",
    )

    parser.add_argument(
        "--teacher",
        default="browser",
    )

    parser.add_argument(
        "--teacher-model",
        default=None,
    )

    parser.add_argument(
        "--turns",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        default=(
            "gene/data/learning/"
            "raw/teacher"
        ),
    )

    args = parser.parse_args()

    session = LearningSession(
        session_id=(
            "learn-"
            + uuid.uuid4().hex
        ),
        mission=args.mission,
        competency_level=args.level,
        domain=args.domain,
        teacher_provider=args.teacher,
        teacher_model=args.teacher_model,
    )

    browser = BrowserTeacher()

    store = TeacherSessionStore(
        args.output
    )

    pipeline = TeacherDataPipeline()

    await browser.start()

    try:

        await browser.navigate(
            args.url
        )

        page_info = (
            await browser.inspect()
        )

        session.source_url = (
            page_info["url"]
        )

        session.source_title = (
            page_info["title"]
        )

        print()
        print(
            "=" * 72
        )
        print(
            "GENE TEACHER SESSION"
        )
        print(
            "=" * 72
        )
        print(
            f"Mission: {session.mission}"
        )
        print(
            f"Domain: {session.domain}"
        )
        print(
            f"Level: {session.competency_level}"
        )
        print(
            f"Teacher: {session.teacher_provider}"
        )
        print(
            f"URL: {session.source_url}"
        )
        print()
        print(
            "Make sure the browser is logged in "
            "and the teacher page is ready."
        )
        print(
            "Press ENTER to begin."
        )

        input()

        opening_prompt = (
            "You are teaching Gene. "
            f"The learning mission is: "
            f"{session.mission}. "
            f"The current internal competency "
            f"level is {session.competency_level}. "
            "Teach through dialogue. "
            "Do not assume Gene already understands "
            "the subject. Ask questions, explain, "
            "challenge it, correct mistakes, and "
            "give practice tasks. "
            "Keep the lesson grounded in evidence."
        )

        for turn in range(
            args.turns
        ):

            exchange = await browser.ask(
                opening_prompt
                if turn == 0
                else (
                    "Continue teaching Gene. "
                    "Review the previous attempt, "
                    "identify weaknesses, and give "
                    "the next appropriate lesson or "
                    "practice task."
                )
            )

            teacher_text = (
                exchange[
                    "page_text_after"
                ]
            )

            session.add(
                "user",
                opening_prompt
                if turn == 0
                else (
                    "Continue teaching Gene."
                ),
            )

            session.add(
                "assistant",
                teacher_text,
            )

            screenshot = (
                await browser.screenshot(
                    "gene/data/learning/"
                    "screenshots/"
                    f"{session.session_id}-"
                    f"{turn:04d}.png"
                )
            )

            session.screenshots.append(
                screenshot
            )

            session.metadata[
                "turns_completed"
            ] = turn + 1

            store.save(
                session
            )

            print(
                f"TURN {turn + 1}/"
                f"{args.turns} captured."
            )

        raw = store.save(
            session
        )

        normalized = (
            pipeline.normalize_session(
                raw
            )
        )

        verified = (
            pipeline.verify_session(
                normalized
            )
        )

        print()
        print(
            "RAW:",
            raw,
        )

        print(
            "NORMALIZED:",
            normalized,
        )

        print(
            "VERIFIED CANDIDATE:",
            verified,
        )

    finally:

        await browser.stop()


def main():

    asyncio.run(
        run()
    )


if __name__ == "__main__":
    main()
