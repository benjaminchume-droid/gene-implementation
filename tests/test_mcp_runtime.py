from gene.mcp.registry import MCPRegistry, MCPServerConfig
from gene.mcp.runtime import MCPRuntime


def test_mcp_registry(tmp_path):

    registry = MCPRegistry(
        str(tmp_path / "servers.json")
    )

    config = MCPServerConfig(
        id="test",
        name="Test",
        transport="stdio",
        command="python",
        args=["-c", "print('mcp')"],
    )

    registry.register(config)

    assert registry.get("test").name == "Test"

    second = MCPRegistry(
        str(tmp_path / "servers.json")
    )

    assert len(second.list()) == 1


def test_mcp_runtime(tmp_path):

    from gene.mcp.registry import MCPRegistry
    from gene.mcp.service import MCPService

    registry = MCPRegistry(
        str(tmp_path / "servers.json")
    )

    # Use an isolated registry for this test so repeated
    # test runs cannot collide with an existing server ID.
    runtime = MCPRuntime()
    runtime.registry = registry
    runtime.service = MCPService(registry)

    result = runtime.register_stdio(
        server_id="test-runtime",
        name="Test Runtime",
        command="python",
        args=["-c", "print('mcp')"],
    )

    assert result["success"] is True
    assert runtime.status()["registered"] >= 1


def test_mcp_service_status():

    runtime = MCPRuntime()

    status = runtime.status()

    assert "registered" in status
    assert "connected" in status
    assert "servers" in status
