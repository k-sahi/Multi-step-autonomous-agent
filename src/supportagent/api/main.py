"""FastAPI service: customer chat + operator approval queue.

Customer-facing
  POST /v1/sessions                          -> {thread_id}
  POST /v1/sessions/{thread_id}/messages     {"text"} -> TurnResult
  GET  /v1/sessions/{thread_id}              -> status, messages (customer-visible), approvals
Operator-facing (requires X-Operator-Token; reviewer identity from X-Reviewer)
  GET  /v1/approvals                         -> pending approval requests across sessions
  POST /v1/approvals/{thread_id}             {"decisions": {call_id: {"approved", "reason"}}} -> TurnResult
  GET  /v1/sessions/{thread_id}/trace        -> trace records (debugging)
Ops
  GET  /healthz

Each session here gets its own copy of the demo retail DB so the service is self-contained.
In production the tools are an MCP server in front of real systems (TOOLS_MCP_URL), and
the LangGraph checkpointer is Postgres, so any replica can resume any paused session.
"""
from __future__ import annotations

import hmac
import os
import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..agent import Deps, SupportAgent
from ..config import settings
from ..domains.retail import POLICY, RetailDB, build_tools
from ..tracing import Tracer, load_trace

app = FastAPI(title="Guarded Support Agent", version="0.1.0")
SESSIONS: dict[str, dict] = {}


def _make_agent(thread_id: str) -> SupportAgent:
    from ..llm import AnthropicChat

    mcp_url = os.environ.get("TOOLS_MCP_URL")
    if mcp_url:
        from ..tools.mcp_client import MCPToolbox

        box = MCPToolbox.connect(mcp_url)
        deps = Deps(AnthropicChat(), box.registry, POLICY, executor=box.execute, approval_mode="human",
                    tracer=Tracer(thread_id))
    else:
        db = RetailDB()
        deps = Deps(AnthropicChat(), build_tools(db), POLICY, approval_mode="human", tracer=Tracer(thread_id))
    return SupportAgent(deps)


def _session(thread_id: str) -> dict:
    s = SESSIONS.get(thread_id)
    if s is None and os.environ.get("TOOLS_MCP_URL"):
        # Stateless mode (Cloud Run): the conversation lives in the shared Postgres checkpointer, so a
        # request that lands on a different instance rebuilds the agent and resumes from the checkpoint.
        agent = _make_agent(thread_id)
        if agent.backend == "langgraph" and agent._state(thread_id):
            s = SESSIONS[thread_id] = {"agent": agent, "created": time.time(), "domain": "retail"}
    if s is None:
        raise HTTPException(404, "session not found")
    return s


def operator(x_operator_token: str = Header(""), x_reviewer: str = Header("")) -> str:
    expected = os.environ.get("OPERATOR_TOKEN", "")
    if not expected or not hmac.compare_digest(x_operator_token, expected):
        raise HTTPException(401, "operator token required")
    if not x_reviewer:
        raise HTTPException(400, "X-Reviewer header required (who is approving)")
    return x_reviewer


class NewSession(BaseModel):
    domain: Literal["retail"] = "retail"


class UserText(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class ApprovalDecision(BaseModel):
    approved: bool
    reason: str = ""


class ApprovalIn(BaseModel):
    decisions: dict[str, ApprovalDecision]


@app.get("/healthz")
def healthz():
    return {"ok": True, "anthropic": bool(settings.anthropic_api_key), "sessions": len(SESSIONS)}


@app.post("/v1/sessions", status_code=201)
def create_session(body: NewSession):
    tid = uuid.uuid4().hex[:16]
    SESSIONS[tid] = {"agent": _make_agent(tid), "created": time.time(), "domain": body.domain}
    return {"thread_id": tid}


@app.post("/v1/sessions/{thread_id}/messages")
def post_message(thread_id: str, body: UserText):
    s = _session(thread_id)
    agent: SupportAgent = s["agent"]
    if agent.pending_approval(thread_id):
        raise HTTPException(409, "an action is awaiting supervisor approval; please wait")
    return agent.send(thread_id, body.text).to_dict()


@app.get("/v1/sessions/{thread_id}")
def get_session(thread_id: str):
    agent: SupportAgent = _session(thread_id)["agent"]
    state = agent._state(thread_id)
    visible = [{"role": m["role"], "content": m["content"]} for m in state.get("messages", [])
               if m["role"] in ("user", "assistant") and m["content"]]
    return {"thread_id": thread_id, "status": state.get("status"), "messages": visible,
            "approvals": state.get("approvals", []), "pending_approval": agent.pending_approval(thread_id)}


@app.get("/v1/approvals")
def list_approvals(reviewer: str = Depends(operator)):
    # Lists approvals held by this instance. With several instances, operators work from a queue
    # fed by the approval request (or Temporal's workflow list) rather than from this endpoint.
    out = []
    for tid, s in SESSIONS.items():
        req = s["agent"].pending_approval(tid)
        if req:
            out.append({"thread_id": tid, **req})
    return {"pending": out}


@app.post("/v1/approvals/{thread_id}")
def decide(thread_id: str, body: ApprovalIn, reviewer: str = Depends(operator)):
    agent: SupportAgent = _session(thread_id)["agent"]
    req = agent.pending_approval(thread_id)
    if not req:
        raise HTTPException(409, "nothing to approve")
    wanted = {a["call_id"] for a in req["actions"]}
    if set(body.decisions) != wanted:
        raise HTTPException(422, f"decide every pending action: {sorted(wanted)}")
    decisions = {cid: {"approved": d.approved, "reason": d.reason, "reviewer": reviewer}
                 for cid, d in body.decisions.items()}
    return agent.resolve(thread_id, decisions).to_dict()


@app.get("/v1/sessions/{thread_id}/trace")
def get_trace(thread_id: str, reviewer: str = Depends(operator)):
    _session(thread_id)
    path = settings.trace_dir / f"{thread_id}.jsonl"
    return {"records": load_trace(path) if path.exists() else []}
