"""Optional durable layer: one Temporal workflow per support conversation.

Why add Temporal on top of LangGraph checkpoints?
  * Approvals can take hours. A Temporal workflow waits on a signal for days at no cost,
    survives deploys and worker crashes, and enforces an approval *timeout* (auto-deny and
    escalate) — something a checkpoint alone cannot do, because nothing wakes it up.
  * Tool side effects run as activities with retries and timeouts, and Temporal's history
    is an audit log of who approved what, when.
LangGraph still owns the reasoning loop; each agent turn is one activity whose state is in
the Postgres checkpointer, so the activity is safe to retry (the graph resumes from the
last checkpoint instead of re-running completed steps).

Signals:  user_message(text), approval(decisions, reviewer)
Queries:  status() -> {"status", "pending_approval", "last_reply"}
Run:      python -m supportagent.temporal.worker
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from ..config import settings


@dataclass
class TurnInput:
    thread_id: str
    text: str | None = None
    decisions: dict | None = None


@dataclass
class TurnOutput:
    status: str
    reply: str | None = None
    approval_request: dict | None = None


@dataclass
class ApprovalSignal:
    decisions: dict = field(default_factory=dict)
    reviewer: str = ""


_agent = None


def _get_agent():
    """One SupportAgent per worker process, using the Postgres checkpointer (shared by all workers)."""
    global _agent
    if _agent is None:
        from ..agent import Deps, SupportAgent
        from ..domains.retail import POLICY, RetailDB, build_tools
        from ..llm import AnthropicChat

        deps = Deps(AnthropicChat(), build_tools(RetailDB()), POLICY, approval_mode="human")
        _agent = SupportAgent(deps, backend="langgraph")
    return _agent


@activity.defn
async def agent_turn(inp: TurnInput) -> TurnOutput:
    agent = _get_agent()
    res = agent.resolve(inp.thread_id, inp.decisions) if inp.decisions is not None else agent.send(inp.thread_id, inp.text)
    return TurnOutput(res.status, res.reply, res.approval_request)


@workflow.defn
class SupportConversation:
    def __init__(self) -> None:
        self.inbox: list[str] = []
        self.approval: ApprovalSignal | None = None
        self.state = TurnOutput("awaiting_user")

    @workflow.signal
    def user_message(self, text: str) -> None:
        self.inbox.append(text)

    @workflow.signal
    def approve(self, sig: ApprovalSignal) -> None:
        self.approval = sig

    @workflow.query
    def status(self) -> dict:
        return {"status": self.state.status, "last_reply": self.state.reply,
                "pending_approval": self.state.approval_request}

    @workflow.run
    async def run(self, thread_id: str) -> str:
        opts = dict(start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=2)))
        while True:
            try:
                await workflow.wait_condition(lambda: bool(self.inbox), timeout=timedelta(hours=24))
            except TimeoutError:
                return "idle_timeout"
            self.state = await workflow.execute_activity(agent_turn, TurnInput(thread_id, text=self.inbox.pop(0)),
                                                         **opts)
            while self.state.status == "awaiting_approval":
                try:
                    await workflow.wait_condition(lambda: self.approval is not None,
                                                  timeout=timedelta(seconds=settings.approval_timeout_s))
                    decisions = {k: {**v, "reviewer": self.approval.reviewer} for k, v in self.approval.decisions.items()}
                except TimeoutError:
                    # Nobody signed off in time: fail safe (deny), the agent tells the customer and offers a transfer.
                    decisions = {a["call_id"]: {"approved": False, "reviewer": "timeout",
                                                "reason": "no supervisor decision in time"}
                                 for a in self.state.approval_request["actions"]}
                self.approval = None
                self.state = await workflow.execute_activity(agent_turn, TurnInput(thread_id, decisions=decisions),
                                                             **opts)
            if self.state.status in ("transferred", "escalated"):
                return self.state.status
