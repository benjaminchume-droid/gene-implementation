from gene.evolution.promotion import (
    GeneUnit,
    Genome,
    PromotionEngine,
    PromotionStore,
    SkillCandidate,
)


def test_skill_candidate(
    tmp_path,
):

    store = PromotionStore(
        str(tmp_path)
    )

    engine = PromotionEngine(
        store=store
    )

    class Capability:
        id = "capability-1"
        resource_id = "resource-1"
        resource_kind = "unknown"
        evidence_ids = ["evidence-1"]
        confidence = 0.9
        verified = True
        metadata = {}

    candidate = (
        engine.candidate_from_capability(
            Capability(),
            name="discovered-skill",
            domain="discovered",
            description=
                "A dynamically discovered capability.",
            instructions=
                "Execute the verified procedure.",
        )
    )

    validated = (
        engine.validate_candidate(
            candidate,
            score=0.95,
            tests_passed=10,
            tests_failed=0,
        )
    )

    assert validated.validated

    result = engine.promote_skill(
        validated
    )

    assert result.name == (
        "discovered-skill"
    )


def test_gene_creation(
    tmp_path,
):

    store = PromotionStore(
        str(tmp_path)
    )

    engine = PromotionEngine(
        store=store
    )

    gene = engine.create_gene(
        name="dynamic-capability-unit",
        skill_ids=[
            "skill-a",
            "skill-b",
        ],
    )

    assert gene.skill_ids == [
        "skill-a",
        "skill-b",
    ]


def test_genome_composition(
    tmp_path,
):

    store = PromotionStore(
        str(tmp_path)
    )

    engine = PromotionEngine(
        store=store
    )

    genome = engine.compose_genome(
        name="dynamic-genome",
        gene_ids=[
            "gene-a",
            "gene-b",
        ],
    )

    assert isinstance(
        genome,
        Genome,
    )

    assert genome.version == 1


def test_unvalidated_candidate_rejected(
    tmp_path,
):

    store = PromotionStore(
        str(tmp_path)
    )

    engine = PromotionEngine(
        store=store
    )

    candidate = SkillCandidate(
        id="x",
        name="unverified",
        domain="unknown",
        description="test",
        instructions="test",
        validated=False,
    )

    try:
        engine.promote_skill(
            candidate
        )
        assert False
    except ValueError:
        pass
