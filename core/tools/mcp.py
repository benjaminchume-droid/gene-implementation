from __future__ import annotations


class MCPProvider:
    """
    MCP is one tool-provider protocol inside Gene's capability system.

    It is intentionally not the capability system itself.
    """

    def __init__(self) -> None:
        self.connected_servers: dict[str, object] = {}

    def register_server(self, name: str, server: object) -> None:
        self.connected_servers[name] = server

    def list_servers(self) -> list[str]:
        return list(self.connected_servers)
