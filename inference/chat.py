from __future__ import annotations

import argparse

from .engine import (
    GeneInferenceEngine,
    SessionStore,
    create_session,
)


def main():

    parser = argparse.ArgumentParser(
        description="Gene interactive chat."
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a Gene checkpoint.",
    )

    parser.add_argument(
        "--session",
        default=None,
        help="Existing session ID.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    engine = GeneInferenceEngine(
        checkpoint_path=
            args.checkpoint
    )

    store = SessionStore()

    if args.session:

        session = store.load(
            args.session
        )

    else:

        session = create_session()

    print()
    print(
        "=" * 72
    )
    print(
        "GENE CHAT"
    )
    print(
        "=" * 72
    )
    print(
        f"Session: {session.session_id}"
    )

    if args.checkpoint:

        print(
            f"Checkpoint: "
            f"{args.checkpoint}"
        )

    else:

        print(
            "Checkpoint: none "
            "(untrained/random model)"
        )

    print(
        "Commands: /exit /clear /save"
    )
    print()

    while True:

        try:

            user_input = input(
                "You: "
            )

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print()
            break

        text = user_input.strip()

        if not text:
            continue

        if text.lower() in {
            "/exit",
            "exit",
            "quit",
            "/quit",
        }:
            break

        if text == "/clear":

            session.clear()

            print(
                "Conversation cleared."
            )

            continue

        if text == "/save":

            path = store.save(
                session
            )

            print(
                f"Saved: {path}"
            )

            continue

        session.add(
            "user",
            text,
        )

        try:

            response = (
                engine.generate_reply(
                    session,
                    max_new_tokens=
                        args.max_new_tokens,
                    temperature=
                        args.temperature,
                    top_k=
                        args.top_k,
                )
            )

        except Exception as exc:

            # Remove the unpaired user message if
            # generation itself failed.
            session.messages.pop()

            print(
                "Gene error:",
                type(exc).__name__,
                exc,
            )

            continue

        session.add(
            "assistant",
            response,
        )

        print(
            f"Gene: {response}"
        )

    store.save(
        session
    )

    print(
        f"\nSession saved: "
        f"{session.session_id}"
    )


if __name__ == "__main__":
    main()
