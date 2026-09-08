from pathlib import Path
from datetime import datetime, timezone

from .registry import ToolRegistry, ToolDefinition
from .permissions import PermissionPolicy, PermissionDecision
from .filesystem import FilesystemTools
from .process import ProcessTools
from .apps import AppTools
from .browser import BrowserTools
from .mcp import MCPTools


class ToolRuntime:

    def __init__(self, workspace="gene"):
        self.workspace = Path(workspace).resolve()

        self.registry = ToolRegistry()
        self.permissions = PermissionPolicy()

        self.filesystem = FilesystemTools(self.workspace)
        self.process = ProcessTools()
        self.apps = AppTools()
        self.browser = BrowserTools()
        self.mcp = MCPTools()

        self._register_tools()

    def _register_tools(self):

        self.registry.register(ToolDefinition(
            "filesystem.read",
            "Read text from a file.",
            self.filesystem.read,
        ))

        self.registry.register(ToolDefinition(
            "filesystem.write",
            "Create or replace a text file.",
            self.filesystem.write,
        ))

        self.registry.register(ToolDefinition(
            "filesystem.list",
            "List files and directories.",
            self.filesystem.list,
        ))

        self.registry.register(ToolDefinition(
            "filesystem.search",
            "Search a directory tree by filename.",
            self.filesystem.search,
        ))

        self.registry.register(ToolDefinition(
            "process.execute",
            "Execute a local operating-system command.",
            self.process.execute,
            requires_confirmation=True,
        ))

        self.registry.register(ToolDefinition(
            "apps.find",
            "Find installed applications and executable commands.",
            self.apps.find,
        ))

        self.registry.register(ToolDefinition(
            "browser.navigate",
            "Navigate a browser adapter to a URL.",
            self.browser.navigate,
        ))

        self.registry.register(ToolDefinition(
            "browser.execute",
            "Execute a browser interaction.",
            self.browser.execute,
            requires_confirmation=True,
        ))

        self.registry.register(ToolDefinition(
            "mcp.register",
            "Register an MCP server configuration.",
            self.mcp.register_server,
        ))

        self.registry.register(ToolDefinition(
            "mcp.list",
            "List registered MCP servers.",
            self.mcp.list_servers,
        ))

        self.registry.register(ToolDefinition(
            "mcp.execute",
            "Execute a tool exposed by an MCP server.",
            self.mcp.execute,
            requires_confirmation=True,
        ))

    def capability_status(self):
        return self.registry.names()

    def execute(self, tool_name: str, **kwargs):

        timestamp = datetime.now(timezone.utc).isoformat()

        if not self.registry.has(tool_name):
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
            }

        permission = self.permissions.check(tool_name)

        if permission == PermissionDecision.DENIED:
            return {
                "success": False,
                "error": f"Permission denied: {tool_name}",
                "permission": permission.value,
            }

        if permission == PermissionDecision.CONFIRMATION_REQUIRED:
            return {
                "success": False,
                "error": f"Confirmation required: {tool_name}",
                "permission": permission.value,
                "requires_confirmation": True,
            }

        definition = self.registry.get(tool_name)

        try:

            result = definition.handler(**kwargs)

            return {
                **result,
                "tool": tool_name,
                "timestamp": timestamp,
                "permission": permission.value,
            }

        except Exception as exc:

            return {
                "success": False,
                "tool": tool_name,
                "timestamp": timestamp,
                "permission": permission.value,
                "error": str(exc),
            }

    def describe_tools(self):
        return self.registry.describe()
