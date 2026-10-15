"""Graph topology (defined once) and the LangGraph builder.

TOPOLOGY is consumed by both `build_langgraph` (production) and `runtime.MiniGraph`
(tests / offline demo), so the two can't drift apart.
"""
from __future__ import annotations

from typing import Any

from .nodes import Deps, make_nodes

END = "__end__"

# node -> fixed next node, or (router name, {route label: next node})
TOPOLOGY: dict[str, Any] = {
    "agent": ("route_agent", {"guard": "guard", "output_guard": "output_guard", "escalate": "escalate"}),
    "guard": ("route_guard", {"approval": "approval", "tools": "tools", "agent": "agent", "escalate": "escalate"}),
    "approval": ("route_approval", {"tools": "tools", "agent": "agent"}),
    "tools": "agent",
    "output_guard": ("route_output", {"agent": "agent", "end": END}),
    "escalate": END,
}
ENTRY = "agent"


def build_langgraph(deps: Deps, checkpointer=None):
    """Compile the agent as a LangGraph StateGraph with a checkpointer (required for interrupts)."""
    from langgraph.graph import END as LG_END
    from langgraph.graph import START, StateGraph
    from langgraph.types import interrupt

    from .state import AgentState

    deps.interrupt_fn = interrupt
    nodes = make_nodes(deps)
    b = StateGraph(AgentState)
    for name in TOPOLOGY:
        b.add_node(name, nodes[name])
    b.add_edge(START, ENTRY)
    for name, nxt in TOPOLOGY.items():
        if isinstance(nxt, str):
            b.add_edge(name, LG_END if nxt == END else nxt)
        else:
            router, mapping = nxt
            b.add_conditional_edges(name, nodes[router],
                                    {k: (LG_END if v == END else v) for k, v in mapping.items()})
    return b.compile(checkpointer=checkpointer or make_checkpointer())


def make_checkpointer(url: str | None = None):
    """sqlite:///path for dev, postgresql://... for Cloud Run (a container's disk is ephemeral,
    so paused approvals must live in a database to survive restarts and scale-out)."""
    from ..config import settings

    url = url or settings.checkpoint_url
    if url.startswith("memory"):
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    if url.startswith("sqlite"):
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver(sqlite3.connect(url.removeprefix("sqlite:///"), check_same_thread=False))
    if url.startswith("postgres"):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
        from psycopg.rows import dict_row

        conn = Connection.connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row)
        saver = PostgresSaver(conn)
        saver.setup()
        return saver
    raise ValueError(f"unsupported CHECKPOINT_URL {url}")
