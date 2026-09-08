from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MCPServerConfig:
    id: str
    name: str
    transport: str = "stdio"

    command: str | None = None
    args: list[str] = field(default_factory=list)

    url: str | None = None

    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    enabled: bool = True


class MCPRegistry:

    def __init__(
        self,
        path: str = "gene/data/mcp/servers.json",
    ) -> None:

        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.servers: dict[str, MCPServerConfig] = {}

        self._load()

    def _load(self) -> None:

        if not self.path.exists():
            return

        raw = self.path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not raw:
            return

        data = json.loads(raw)

        for item in data:
            config = MCPServerConfig(
                **item
            )

            self.servers[
                config.id
            ] = config

    def _save(self) -> None:

        self.path.write_text(
            json.dumps(
                [
                    asdict(server)
                    for server
                    in self.servers.values()
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def register(
        self,
        config: MCPServerConfig,
    ) -> MCPServerConfig:

        if config.id in self.servers:
            raise ValueError(
                f"MCP server already exists: {config.id}"
            )

        if config.transport == "stdio":
            if not config.command:
                raise ValueError(
                    "stdio MCP server requires command."
                )

        elif config.transport in {
            "streamable_http",
            "sse",
        }:
            if not config.url:
                raise ValueError(
                    f"{config.transport} requires url."
                )

        else:
            raise ValueError(
                f"Unsupported MCP transport: "
                f"{config.transport}"
            )

        self.servers[
            config.id
        ] = config

        self._save()

        return config

    def remove(
        self,
        server_id: str,
    ) -> bool:

        if server_id not in self.servers:
            return False

        del self.servers[
            server_id
        ]

        self._save()

        return True

    def get(
        self,
        server_id: str,
    ) -> MCPServerConfig:

        try:
            return self.servers[
                server_id
            ]
        except KeyError:
            raise KeyError(
                f"Unknown MCP server: {server_id}"
            ) from None

    def list(self) -> list[MCPServerConfig]:
        return list(
            self.servers.values()
        )
