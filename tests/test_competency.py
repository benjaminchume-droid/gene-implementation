from gene.learning.competency import (
    CompetencyEvaluator,
    CompetencyStage,
)


def test_initiate():

    profile = CompetencyEvaluator().evaluate(
        knowledge_depth=0.10,
        procedural_ability=0.05,
        independence=0.05,
        transfer_ability=0.05,
        verification_strength=0.10,
        professional_applicability=0.05,
        confidence=0.80,
    )

    assert profile.stage == (
        CompetencyStage.INITIATE
    )


def test_intermediate():

    profile = CompetencyEvaluator().evaluate(
        knowledge_depth=0.55,
        procedural_ability=0.55,
        independence=0.55,
        transfer_ability=0.50,
        verification_strength=0.55,
        professional_applicability=0.40,
        confidence=0.80,
    )

    assert profile.stage == (
        CompetencyStage.INTERMEDIATE
    )


def test_professional():

    profile = CompetencyEvaluator().evaluate(
        knowledge_depth=0.90,
        procedural_ability=0.92,
        independence=0.90,
        transfer_ability=0.88,
        verification_strength=0.94,
        professional_applicability=0.95,
        confidence=0.90,
    )

    assert profile.stage == (
        CompetencyStage.PROFESSIONAL
    )
