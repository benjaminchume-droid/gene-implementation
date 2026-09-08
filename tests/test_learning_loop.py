from gene.learning.loop import (
    Experience,
    ExperienceAnalyzer,
    ExperienceStore,
    LearningLoop,
)


def test_experience_record(tmp_path):

    store = ExperienceStore(
        str(
            tmp_path
            / "experiences.jsonl"
        )
    )

    experience = Experience.create(
        objective="Complete an arbitrary task.",
        outcome="Task completed successfully.",
        success=True,
    )

    store.save(experience)

    assert len(store.successful()) == 1


def test_experience_analysis():

    analyzer = ExperienceAnalyzer()

    result = analyzer.analyze(
        [
            {
                "success": True,
                "capability_ids": ["cap-a"],
                "errors": [],
            },
            {
                "success": False,
                "capability_ids": [],
                "errors": [
                    {
                        "type": "execution_error"
                    }
                ],
            },
        ]
    )

    assert result["total"] == 2
    assert result["success_rate"] == 0.5
    assert result["patterns"]
    assert result["failure_patterns"]


def test_learning_loop(tmp_path):

    store = ExperienceStore(
        str(
            tmp_path
            / "experiences.jsonl"
        )
    )

    loop = LearningLoop(
        store=store
    )

    result = loop.record(
        Experience.create(
            objective="Learn a successful procedure.",
            outcome="Procedure completed and verified.",
            success=True,
        )
    )

    assert result["candidate"] is not None

    status = loop.status()

    assert status["storage"]["successful"] == 1
