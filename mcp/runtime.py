from __future__ import annotations

from gene.mcp.registry import (
    MCPRegistry,
)
from gene.mcp.service import (
    MCPService,
)


class MCPRuntime:

    def __init__(self):
        self.registry = MCPRegistry()

        self.service = MCPService(
            self.registry
        )

    def register_stdio(
        self,
        server_id: str,
        name: str,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict:

        from gene.mcp.registry import (
            MCPServerConfig,
        )

        config = MCPServerConfig(
            id=server_id,
            name=name,
            transport="stdio",
            command=command,
            args=args or [],
            cwd=cwd,
            env=env or {},
        )

        self.registry.register(
            config
        )

        return {
            "success": True,
            "server": config.name,
            "id": config.id,
        }

    def status(self) -> dict:
        return self.service.status()
