from gene.agent import GeneAgent


def test_agent_boots() -> None:
    agent = GeneAgent()

    status = agent.status()

    assert status["name"] == "Gene"
    assert status["genome"] == "gene-1"
    assert status["tools"] > 0


def test_agent_routes_filesystem() -> None:
    agent = GeneAgent()

    result = agent.run(
        "list the files in gene"
    )

    assert result["route"] == "filesystem"
    assert result["status"] == "completed"


def test_agent_routes_screen() -> None:
    agent = GeneAgent()

    result = agent.run(
        "analyze what is on my screen"
    )

    assert result["route"] == "screen"
