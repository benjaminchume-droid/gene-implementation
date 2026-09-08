from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MCPServer:
    id: str
    name: str
    command: list[str]
    cwd: str | None = None
    env: dict[str, str] | None = None
    status: str = "registered"


class MCPLifecycle:

    def __init__(
        self,
        path: str = "gene/data/mcp/servers.json",
    ) -> None:

        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.servers: dict[str, MCPServer] = {}
        self.processes: dict[str, subprocess.Popen] = {}

        self._load()

    def _load(self):

        if not self.path.exists():
            return

        data = json.loads(
            self.path.read_text(
                encoding="utf-8-sig"
            )
        )

        for item in data:
            server = MCPServer(**item)
            self.servers[server.id] = server

    def _save(self):

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
        name: str,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> MCPServer:

        server = MCPServer(
            id=str(uuid.uuid4()),
            name=name,
            command=command,
            cwd=cwd,
            env=env,
        )

        self.servers[server.id] = server
        self._save()

        return server

    def start(
        self,
        server_id: str,
    ) -> dict:

        server = self.servers[server_id]

        process = subprocess.Popen(
            server.command,
            cwd=server.cwd,
            env=server.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.processes[server_id] = process
        server.status = "running"

        self._save()

        return {
            "success": True,
            "server_id": server_id,
            "pid": process.pid,
        }

    def stop(
        self,
        server_id: str,
    ) -> dict:

        process = self.processes.get(
            server_id
        )

        if process is None:
            self.servers[
                server_id
            ].status = "stopped"

            self._save()

            return {
                "success": True,
                "already_stopped": True,
            }

        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

        self.processes.pop(
            server_id,
            None,
        )

        self.servers[
            server_id
        ].status = "stopped"

        self._save()

        return {
            "success": True,
            "server_id": server_id,
        }

    def list(self) -> list[dict]:
        return [
            asdict(server)
            for server
            in self.servers.values()
        ]