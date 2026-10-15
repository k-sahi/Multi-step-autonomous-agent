"""Load tools from any MCP server into a ToolRegistry the agent can use.

The graph nodes are synchronous (simpler checkpointing and tracing), while the MCP client
is async. MCPToolbox keeps one MCP session open on a private event loop thread and
exposes a blocking `execute(name, args)` that the tools node calls.

    box = MCPToolbox.connect(server)                    # in-process MCPServer (tests, demo)
    box = MCPToolbox.connect("http://tools:8001/mcp")   # streamable HTTP (Cloud Run sidecar/service)
    deps = Deps(llm, box.registry, policy, executor=box.execute)
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from mcp import Client

from .registry import ToolRegistry, ToolSpec, classify


class MCPToolbox:
    def __init__(self, target: Any):
        self.target = target
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="mcp-client")
        self.thread.start()
        self.registry = ToolRegistry()
        self._client: Client | None = None

    @classmethod
    def connect(cls, target: Any) -> "MCPToolbox":
        box = cls(target)
        box._run(box._open())
        return box

    def _run(self, coro, timeout: float = 60.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _open(self):
        self._client = Client(self.target)
        await self._client.__aenter__()
        listed = await self._client.list_tools()
        for t in listed.tools:
            self.registry.add(spec_from_mcp(t))

    def execute(self, name: str, arguments: dict) -> tuple[str, bool]:
        if name not in self.registry:
            return f"Error: unknown tool '{name}'", True
        try:
            res = self._run(self._client.call_tool(name, arguments))
        except Exception as exc:  # transport failure: surface to the model as a tool error
            return f"Error: tool server unavailable ({type(exc).__name__})", True
        if res.structured_content is not None:
            sc = res.structured_content
            content = json.dumps(sc["result"] if set(sc) == {"result"} else sc, default=str)
            if isinstance(sc, dict) and set(sc) == {"result"} and isinstance(sc["result"], str):
                content = sc["result"]
        else:
            content = "\n".join(getattr(b, "text", "") for b in res.content)
        if res.is_error:
            content = content.replace(f"Error executing tool {name}: ", "")
            return f"Error: {content}", True
        return content, False

    def close(self) -> None:
        if self._client is not None:
            try:
                self._run(self._client.__aexit__(None, None, None), timeout=10)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)


def spec_from_mcp(tool) -> ToolSpec:
    """Map an MCP Tool to a ToolSpec. Our own `meta` wins; then standard annotations; then
    the conservative name-based fallback."""
    meta = (getattr(tool, "meta", None) or {}).get("supportagent")
    if meta:
        risk = {k: meta[k] for k in ("kind", "irreversible", "confirm", "approve") if k in meta}
        note = meta.get("risk_note", "")
    else:
        ann = getattr(tool, "annotations", None)
        base = classify(tool.name)
        if ann is not None and ann.read_only_hint:
            risk = {"kind": "read" if base["kind"] != "generic" else "generic"}
        elif ann is not None and ann.destructive_hint:
            risk = {"kind": "write", "irreversible": True, "confirm": True, "approve": True}
        else:
            risk = base
        note = ""
    schema = dict(tool.input_schema or {"type": "object", "properties": {}})
    schema.pop("title", None)
    return ToolSpec(name=tool.name, description=tool.description or "", input_schema=schema, fn=None,
                    risk_note=note, **risk)
