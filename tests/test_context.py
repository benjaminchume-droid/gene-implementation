from gene.context import (
    ContextBudget,
    ContextManager,
    ContextSelector,
)


def test_context_budget() -> None:
    budget = ContextBudget()

    assert budget.native_model_context == 32768
    assert budget.effective_context_target == 500000
    assert budget.validate() == []


def test_context_selection() -> None:
    selector = ContextSelector()

    items = [
        selector_item
        for selector_item in [
            __import__(
                "gene.context.models",
                fromlist=["ContextItem"],
            ).ContextItem(
                content="Python debugging and software development",
                source="skill",
                priority=0.9,
            ),
            __import__(
                "gene.context.models",
                fromlist=["ContextItem"],
            ).ContextItem(
                content="Ancient Roman history",
                source="knowledge",
                priority=0.2,
            ),
        ]
    ]

    selected = selector.select(
        "Python debugging",
        items,
        100,
    )

    assert selected
    assert selected[0].source == "skill"


def test_context_manager() -> None:
    manager = ContextManager()

    rendered = manager.build(
        "Python debugging",
        system=["You are Gene."],
        memories=["User is building Gene AI."],
        knowledge=["Python exceptions provide error information."],
        skills=["Python debugging skill."],
        conversation=["Help me debug this code."],
    )

    assert "Gene" in rendered
    assert "Python" in rendered
    assert manager.configuration()[
        "effective_context_target"
    ] == 500000
