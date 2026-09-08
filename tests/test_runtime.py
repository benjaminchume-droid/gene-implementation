from gene.core.runtime import GeneRuntime


def test_gene_runtime_boots() -> None:
    runtime = GeneRuntime()

    health = runtime.health()

    assert health["status"] == "healthy"
    assert health["identity"]["name"] == "Gene"
    assert health["genome"]["modules"] > 0
    assert health["tools"] > 0


def test_gene_runtime_state() -> None:
    runtime = GeneRuntime()

    state = runtime.state()

    snapshot = state.snapshot()

    assert snapshot["identity"]["name"] == "Gene"
    assert "active_sparks" in snapshot
    assert "memory" in snapshot
    assert "knowledge" in snapshot
    assert "skills" in snapshot
    assert "context" in snapshot
