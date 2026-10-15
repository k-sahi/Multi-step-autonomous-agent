"""Expose a domain's tools as an MCP server (MCP Python SDK v2, `MCPServer`).

Risk metadata travels with each tool so any MCP client — this agent, Claude Desktop, an
IDE — sees which tools are read-only and which are destructive:

  ToolAnnotations.read_only_hint    kind in (read, generic)
  ToolAnnotations.destructive_hint  irreversible writes
  ToolAnnotations.idempotent_hint   reads
  meta["supportagent"]              {kind, irreversible, confirm, approve}  (our richer policy)

Annotations are *hints*: the agent never trusts a remote server's own claim that a tool is
safe. MCPToolbox maps them to ToolSpecs, and unknown tools fall back to the conservative
name-based classification (tools/registry.classify).

Run:
  python -m supportagent.tools.mcp_server                       # stdio
  python -m supportagent.tools.mcp_server --http --port 8001    # streamable HTTP at /mcp
"""
from __future__ import annotations

import argparse
import functools

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .registry import ToolExecutionError, ToolRegistry


def _wrap(fn):
    @functools.wraps(fn)          # keeps the signature, which MCPServer uses to build the input schema
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolExecutionError as exc:
            raise ToolError(str(exc)) from exc   # expected failure: message is shown to the model
    return inner


def build_server(registry: ToolRegistry, name: str = "support-tools") -> MCPServer:
    server = MCPServer(name)
    for spec in registry:
        if spec.fn is None:
            continue
        ann = ToolAnnotations(
            title=spec.name.replace("_", " "),
            read_only_hint=spec.kind in ("read", "generic"),
            destructive_hint=spec.irreversible,
            idempotent_hint=spec.kind == "read",
            open_world_hint=False,
        )
        meta = {"supportagent": {"kind": spec.kind, "irreversible": spec.irreversible,
                                 "confirm": spec.confirm, "approve": spec.approve, "risk_note": spec.risk_note}}
        server.tool(name=spec.name, description=spec.description, annotations=ann, meta=meta)(_wrap(spec.fn))
    return server


def main():
    from ..domains.retail import RetailDB, build_tools

    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    server = build_server(build_tools(RetailDB()), "retail-support-tools")
    if a.http:
        # Stateless HTTP: any replica can serve any request (Cloud Run scales instances freely).
        server.run(transport="streamable-http", host=a.host, port=a.port, stateless_http=True)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
