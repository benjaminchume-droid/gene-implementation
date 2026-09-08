from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPConnection:

    def __init__(self, config):
        self.config = config

        self._stdio_context = None
        self._read_stream = None
        self._write_stream = None
        self._session: ClientSession | None = None

        self.initialized = False

    async def connect(self) -> dict:

        if self.initialized:
            return {
                "success": True,
                "status": "already_connected",
            }

        if self.config.transport != "stdio":
            raise NotImplementedError(
                "This runtime currently implements "
                "stdio connections. Streamable HTTP/SSE "
                "remain transport adapters."
            )

        if not self.config.command:
            raise ValueError(
                "stdio server requires command."
            )

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env or None,
            cwd=self.config.cwd,
        )

        self._stdio_context = stdio_client(
            params
        )

        (
            self._read_stream,
            self._write_stream,
        ) = await self._stdio_context.__aenter__()

        self._session = ClientSession(
            self._read_stream,
            self._write_stream,
        )

        await self._session.__aenter__()

        await self._session.initialize()

        self.initialized = True

        return {
            "success": True,
            "status": "connected",
            "server": self.config.name,
        }

    async def disconnect(self) -> None:

        if self._session is not None:
            try:
                await self._session.__aexit__(
                    None,
                    None,
                    None,
                )
            except Exception:
                pass

        if self._stdio_context is not None:
            try:
                await self._stdio_context.__aexit__(
                    None,
                    None,
                    None,
                )
            except Exception:
                pass

        self._session = None
        self._stdio_context = None
        self._read_stream = None
        self._write_stream = None
        self.initialized = False

    def _require_session(self) -> ClientSession:

        if (
            self._session is None
            or not self.initialized
        ):
            raise RuntimeError(
                "MCP server is not connected."
            )

        return self._session

    async def capabilities(self) -> dict:

        session = self._require_session()

        capabilities = (
            session.get_server_capabilities()
        )

        if capabilities is None:
            return {}

        result = {}

        for field in (
            "prompts",
            "resources",
            "tools",
            "logging",
            "completions",
        ):

            value = getattr(
                capabilities,
                field,
                None,
            )

            if value is not None:
                result[field] = (
                    value.model_dump()
                    if hasattr(
                        value,
                        "model_dump",
                    )
                    else str(value)
                )

        return result

    async def tools(self) -> list[dict]:

        session = self._require_session()

        result = await session.list_tools()

        return [
            tool.model_dump()
            if hasattr(
                tool,
                "model_dump",
            )
            else {
                "name": tool.name,
                "description":
                    tool.description,
                "inputSchema":
                    tool.inputSchema,
            }
            for tool in result.tools
        ]

    async def resources(self) -> list[dict]:

        session = self._require_session()

        result = await session.list_resources()

        return [
            item.model_dump()
            if hasattr(
                item,
                "model_dump",
            )
            else str(item)
            for item in result.resources
        ]

    async def prompts(self) -> list[dict]:

        session = self._require_session()

        result = await session.list_prompts()

        return [
            item.model_dump()
            if hasattr(
                item,
                "model_dump",
            )
            else str(item)
            for item in result.prompts
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict:

        session = self._require_session()

        result = await session.call_tool(
            name,
            arguments or {},
        )

        payload = {}

        if hasattr(
            result,
            "model_dump",
        ):
            payload = result.model_dump()

        else:
            payload = {
                "content": [
                    item.model_dump()
                    if hasattr(
                        item,
                        "model_dump",
                    )
                    else str(item)
                    for item in result.content
                ],
            }

            if hasattr(
                result,
                "structured_content",
            ):
                payload[
                    "structured_content"
                ] = result.structured_content

        return {
            "success": not bool(
                getattr(
                    result,
                    "isError",
                    False,
                )
            ),
            **payload,
        }

    async def read_resource(
        self,
        uri: str,
    ) -> dict:

        session = self._require_session()

        result = await session.read_resource(
            uri
        )

        if hasattr(
            result,
            "model_dump",
        ):
            return {
                "success": True,
                **result.model_dump(),
            }

        return {
            "success": True,
            "result": str(result),
        }


class MCPService:

    def __init__(
        self,
        registry,
        permission_check=None,
    ) -> None:

        self.registry = registry
        self.permission_check = (
            permission_check
        )

        self.connections: dict[
            str,
            MCPConnection,
        ] = {}

    async def connect(
        self,
        server_id: str,
    ) -> dict:

        config = self.registry.get(
            server_id
        )

        if not config.enabled:
            return {
                "success": False,
                "error":
                    "MCP server is disabled.",
            }

        connection = MCPConnection(
            config
        )

        result = await connection.connect()

        self.connections[
            server_id
        ] = connection

        return result

    async def disconnect(
        self,
        server_id: str,
    ) -> dict:

        connection = self.connections.get(
            server_id
        )

        if connection is None:
            return {
                "success": True,
                "status": "already_disconnected",
            }

        await connection.disconnect()

        self.connections.pop(
            server_id,
            None,
        )

        return {
            "success": True,
            "status": "disconnected",
        }

    async def inspect(
        self,
        server_id: str,
    ) -> dict:

        connection = self.connections.get(
            server_id
        )

        if connection is None:
            await self.connect(
                server_id
            )

            connection = self.connections[
                server_id
            ]

        return {
            "success": True,
            "capabilities":
                await connection.capabilities(),
            "tools":
                await connection.tools(),
            "resources":
                await connection.resources(),
            "prompts":
                await connection.prompts(),
        }

    async def call(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        confirmed: bool = False,
    ) -> dict:

        if (
            self.permission_check is not None
            and not confirmed
        ):

            allowed = self.permission_check(
                "mcp.execute"
            )

            if not allowed:
                return {
                    "success": False,
                    "requires_confirmation": True,
                    "error":
                        "MCP execution requires "
                        "confirmation.",
                }

        connection = self.connections.get(
            server_id
        )

        if connection is None:
            await self.connect(
                server_id
            )

            connection = self.connections[
                server_id
            ]

        return await connection.call_tool(
            tool_name,
            arguments,
        )

    async def resources(
        self,
        server_id: str,
    ) -> list[dict]:

        connection = self.connections.get(
            server_id
        )

        if connection is None:
            await self.connect(
                server_id
            )

            connection = self.connections[
                server_id
            ]

        return await connection.resources()

    async def prompts(
        self,
        server_id: str,
    ) -> list[dict]:

        connection = self.connections.get(
            server_id
        )

        if connection is None:
            await self.connect(
                server_id
            )

            connection = self.connections[
                server_id
            ]

        return await connection.prompts()

    def status(self) -> dict:

        return {
            "registered":
                len(
                    self.registry.list()
                ),
            "connected":
                len(
                    self.connections
                ),
            "servers": [
                {
                    "id": server.id,
                    "name": server.name,
                    "transport":
                        server.transport,
                    "enabled":
                        server.enabled,
                    "connected":
                        server.id
                        in self.connections,
                }
                for server
                in self.registry.list()
            ],
        }


def run_async(coro):
    return asyncio.run(coro)
