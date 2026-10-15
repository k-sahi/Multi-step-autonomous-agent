"""SupportAgent: the conversation-level API used by FastAPI, Temporal and the demo.

  send(thread_id, text)         -> TurnResult   (reply, or a pending approval request)
  pending_approval(thread_id)   -> request | None
  resolve(thread_id, decisions) -> TurnResult   (resumes the paused graph)

Backed by LangGraph when installed (durable checkpointer), else by MiniGraph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..messages import Message, from_dicts
from ..tracing import Tracer
from .nodes import Deps
from .state import initial_state


@dataclass
class TurnResult:
    thread_id: str
    status: str                        # awaiting_user | awaiting_approval | transferred | escalated
    reply: str | None = None
    approval_request: dict | None = None
    new_messages: list[dict] = field(default_factory=list)
    guard_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class SupportAgent:
    def __init__(self, deps: Deps, backend: str = "auto", checkpointer=None):
        self.deps = deps
        self.backend = backend
        if backend in ("auto", "langgraph"):
            try:
                from .graph import build_langgraph

                self.graph = build_langgraph(deps, checkpointer)
                self.backend = "langgraph"
                from langgraph.types import Command as LGCommand

                self._Command = LGCommand
            except ImportError:
                if backend == "langgraph":
                    raise
                backend = "mini"
        if backend == "mini":
            from .runtime import Command, MiniGraph

            self.graph = MiniGraph(deps)
            self.backend = "mini"
            self._Command = Command
        self._seen: dict[str, int] = {}
        self._seen_events: dict[str, int] = {}

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    def _state(self, thread_id: str) -> dict:
        s = self.graph.get_state(self._config(thread_id))
        return s if isinstance(s, dict) else (s.values or {})

    def send(self, thread_id: str, text: str) -> TurnResult:
        self.deps.tracer = self.deps.tracer if self.deps.tracer.thread_id == thread_id else Tracer(thread_id)
        state = self._state(thread_id)
        user = Message.user(text).model_dump(mode="json")
        if not state:
            inp: dict[str, Any] = initial_state(thread_id, self.deps.domain, [user])
        else:
            if state.get("status") in ("transferred", "escalated"):
                return TurnResult(thread_id, state["status"], reply="This conversation was handed to a human agent.")
            inp = {"messages": [user], "status": "running", "blocked_retries": 0}
        return self._result(thread_id, self.graph.invoke(inp, self._config(thread_id)))

    def pending_approval(self, thread_id: str) -> dict | None:
        if self.backend == "mini":
            cp = self.graph.checkpoints.get(thread_id) or {}
            return cp.get("interrupt")
        snap = self.graph.get_state(self._config(thread_id))
        for task in getattr(snap, "tasks", []) or []:
            for intr in getattr(task, "interrupts", []) or []:
                return intr.value
        return None

    def resolve(self, thread_id: str, decisions: dict[str, dict]) -> TurnResult:
        """decisions: {call_id: {"approved": bool, "reviewer": str, "reason": str}}"""
        if self.pending_approval(thread_id) is None:
            raise ValueError(f"no pending approval for thread {thread_id}")
        out = self.graph.invoke(self._Command(resume=decisions), self._config(thread_id))
        return self._result(thread_id, out)

    def messages(self, thread_id: str) -> list[Message]:
        return from_dicts(self._state(thread_id).get("messages", []))

    def _result(self, thread_id: str, out: dict) -> TurnResult:
        msgs = out.get("messages", [])
        start = self._seen.get(thread_id, 0)
        new = msgs[start:]
        self._seen[thread_id] = len(msgs)
        all_events = out.get("guard_events", [])
        events = all_events[self._seen_events.get(thread_id, 0):]
        self._seen_events[thread_id] = len(all_events)
        if out.get("__interrupt__"):
            intr = out["__interrupt__"][0]
            return TurnResult(thread_id, "awaiting_approval", approval_request=getattr(intr, "value", intr),
                              new_messages=new, guard_events=events)
        reply = next((m["content"] for m in reversed(new) if m["role"] == "assistant" and m["content"]
                      and not m.get("tool_calls")), None)
        return TurnResult(thread_id, out.get("status", "awaiting_user"), reply=reply, new_messages=new,
                          guard_events=events)
