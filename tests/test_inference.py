from gene.inference import (
    ChatSession,
    GeneInferenceEngine,
    SessionStore,
)


def test_chat_session():

    session = ChatSession(
        session_id="test-session"
    )

    session.add(
        "user",
        "Hello Gene.",
    )

    session.add(
        "assistant",
        "Hello.",
    )

    assert len(
        session.messages
    ) == 2


def test_session_persistence(
    tmp_path,
):

    store = SessionStore(
        str(tmp_path)
    )

    session = ChatSession(
        session_id="persistent-test"
    )

    session.add(
        "user",
        "Remember this.",
    )

    path = store.save(
        session
    )

    assert path

    restored = store.load(
        "persistent-test"
    )

    assert (
        len(
            restored.messages
        )
        == 1
    )

    assert (
        restored.messages[0]
        .content
        == "Remember this."
    )


def test_model_tokenizer_compatibility():

    engine = GeneInferenceEngine()

    engine.load()

    assert engine.loaded is True

    assert (
        engine.model.config.vocab_size
        == engine.tokenizer.vocab_size_actual()
    )
