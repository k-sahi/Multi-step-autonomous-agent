"""Graph state.

A TypedDict (what LangGraph expects) holding only JSON-serialisable values, so it can be
checkpointed to SQLite/Postgres, sent through Temporal, and written to traces unchanged.
`messages` uses an append reducer: nodes return only the *new* messages.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    thread_id: str
    domain: str
    messages: Annotated[list[dict], operator.add]     # Message.model_dump() rows
    pending_calls: list[dict]                          # tool calls proposed this step, post-guard
    decisions: list[dict]                              # guard decisions for pending_calls
    approvals: Annotated[list[dict], operator.add]     # audit log of human sign-offs
    guard_events: Annotated[list[dict], operator.add]  # every block / flag, for evals and traces
    steps_this_turn: int
    blocked_retries: int
    status: str                                        # running | awaiting_user | transferred | escalated
    escalation_reason: str
    draft: dict | None                                 # text reply awaiting the output guard
    feedback: str | None                               # output-guard feedback for one rewrite
    output_retry: bool


def initial_state(thread_id: str, domain: str, messages: list[dict] | None = None) -> AgentState:
    return AgentState(thread_id=thread_id, domain=domain, messages=messages or [], pending_calls=[],
                      decisions=[], approvals=[], guard_events=[], steps_this_turn=0, blocked_retries=0,
                      status="running", escalation_reason="")


def merge(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Apply a node's update with the same reducer semantics LangGraph uses."""
    out = dict(state)
    for k, v in update.items():
        if k in ("messages", "approvals", "guard_events"):
            out[k] = list(out.get(k, [])) + list(v)
        else:
            out[k] = v
    return out
