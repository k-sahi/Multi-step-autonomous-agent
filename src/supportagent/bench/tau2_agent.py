"""Run the guarded agent inside tau2-bench.

In tau2 the *environment* executes tools and the *user simulator* plays the customer; the
agent only decides the next message. So inside the benchmark our agent is:

    model proposes ──► guard ──(blocked)──► feedback to model, retry (env never sees it)
                          ├─(irreversible)─► approval policy (auto in benchmarks; logged)
                          ├─(stuck)────────► transfer_to_human_agents
                          └─(allowed)──────► emit tool call to tau2's environment
    model replies  ──► output guard ──(unbacked claim)──► one rewrite

Blocked attempts live only in the agent's internal history and in a sidecar file
(<results>.guard.jsonl) so evals can count them without changing tau2's trajectory format.

Written against tau2's HalfDuplexAgent interface (tau3 release): get_init_state /
generate_next_message, factory registered via registry.register_agent_factory.
If an import fails, check src/tau2/agent/README.md of the installed version.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import settings
from ..guardrails import ToolGuard, build_context, check_output, screen
from ..llm import AnthropicChat, ChatModel
from ..messages import Message, ToolCall
from ..tools.registry import ToolRegistry, ToolSpec, classify

try:
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, SystemMessage, ToolMessage, UserMessage
    from tau2.data_model.message import ToolCall as TauToolCall
except ImportError as exc:  # pragma: no cover
    raise ImportError("tau2 is not installed: clone sierra-research/tau2-bench and `uv sync` / pip install -e .") from exc

SIDECAR_LOCK = threading.Lock()
_trial_counter: dict[str, int] = defaultdict(int)


class GuardedState(BaseModel):
    system_messages: list[Any] = Field(default_factory=list)
    messages: list[Any] = Field(default_factory=list)          # what tau2 sees
    internal: list[dict] = Field(default_factory=list)         # our full history incl. blocked attempts
    guard_events: list[dict] = Field(default_factory=list)
    approvals: list[dict] = Field(default_factory=list)
    blocked_calls: list[dict] = Field(default_factory=list)
    name_by_call_id: dict[str, str] = Field(default_factory=dict)


def registry_from_tau2(tools) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        fn = t.openai_schema["function"]
        reg.add(ToolSpec(name=fn["name"], description=fn.get("description", ""),
                         input_schema=fn.get("parameters") or {"type": "object", "properties": {}},
                         fn=None, **classify(fn["name"])))
    return reg


class GuardedReActAgent(HalfDuplexAgent[GuardedState]):
    def __init__(self, tools, domain_policy: str, llm: str | None = None, llm_args: dict | None = None,
                 domain: str = "retail", approval_mode: str = "auto_approve", sidecar: str | None = None,
                 task_id: str | None = None, chat: ChatModel | None = None):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.registry = registry_from_tau2(tools)
        self.guard = ToolGuard(self.registry, domain)
        model = (llm or settings.agent_model).split("/")[-1]
        self.chat = chat or AnthropicChat(model=model, **{k: v for k, v in (llm_args or {}).items()
                                                          if k in ("temperature", "max_tokens")})
        from ..agent.nodes import SYSTEM_TEMPLATE

        self.system = SYSTEM_TEMPLATE.format(policy=domain_policy)
        self.write_tools = {t.name for t in self.registry if t.kind == "write"}
        self.approval_mode = approval_mode
        self.sidecar = sidecar
        self.task_id = task_id

    # ---------------------------------------------------------------- tau2 interface
    def get_init_state(self, message_history: list | None = None) -> GuardedState:
        st = GuardedState(system_messages=[SystemMessage(role="system", content=self.system)],
                          messages=list(message_history or []))
        for m in message_history or []:
            self._ingest(st, m)
        return st

    def generate_next_message(self, message, state: GuardedState):
        if message is not None:
            state.messages.append(message)
            self._ingest(state, message)
        out = self._decide(state)
        state.messages.append(out)
        if out.tool_calls is None and self.sidecar:
            self._flush(state)
        return out, state

    # ---------------------------------------------------------------- core loop
    def _decide(self, st: GuardedState):
        feedback = None
        for _ in range(settings.max_blocked_retries + 2):
            msgs = [Message.model_validate(m) for m in st.internal]
            if feedback:
                msgs.append(Message.user(f"[guardrail feedback, not from the customer] {feedback}"))
            reply = self.chat(self.system, msgs, self.registry.schemas())
            if not reply.tool_calls:
                ctx = build_context([Message.model_validate(m) for m in st.internal], self.write_tools)
                v = check_output(reply.content, ctx)
                if v and feedback is None:
                    feedback = " ".join(x.reason for x in v)
                    st.guard_events += [{"type": "output_blocked", **x.to_dict()} for x in v]
                    continue
                st.internal.append(reply.model_dump(mode="json"))
                return AssistantMessage(role="assistant", content=reply.content or "...")
            feedback = None
            ctx = build_context([Message.model_validate(m) for m in st.internal], self.write_tools)
            decisions = self.guard.check(reply.tool_calls, ctx)
            if any(d.action == "escalate" for d in decisions):
                st.guard_events.append({"type": "escalation", **next(d for d in decisions
                                                                     if d.action == "escalate").to_dict()})
                return self._emit(st, [ToolCall(name="transfer_to_human_agents",
                                                arguments={"summary": "Agent could not complete the request."})])
            allowed: list[ToolCall] = []
            blocked_msgs: list[Message] = []
            for d in decisions:
                if d.action == "approve":
                    ok = self.approval_mode == "auto_approve"
                    st.approvals.append({"tool": d.call.name, "arguments": d.call.arguments, "approved": ok,
                                         "reviewer": f"policy:{self.approval_mode}", "reason": d.reason})
                    if ok:
                        allowed.append(d.call)
                    else:
                        blocked_msgs.append(Message.tool(d.call, "Not executed: a supervisor declined this action.",
                                                         is_error=True, blocked=True))
                elif d.action == "allow":
                    allowed.append(d.call)
                else:
                    st.guard_events.append({"type": "tool_blocked", **d.to_dict()})
                    st.blocked_calls.append({"name": d.call.name, "arguments": d.call.arguments, "rule": d.rule})
                    blocked_msgs.append(Message.tool(d.call, f"Blocked by guardrail ({d.rule}): {d.reason}",
                                                     is_error=True, blocked=True))
            if allowed:
                # Blocked siblings are dropped from the emitted message; the model re-proposes them later.
                return self._emit(st, allowed)
            st.internal.append(Message.assistant(reply.content, reply.tool_calls).model_dump(mode="json"))
            st.internal += [m.model_dump(mode="json") for m in blocked_msgs]
        st.guard_events.append({"type": "escalation", "rule": "blocked_retries", "reason": "too many blocked calls"})
        return self._emit(st, [ToolCall(name="transfer_to_human_agents",
                                        arguments={"summary": "Agent could not complete the request within policy."})])

    def _emit(self, st: GuardedState, calls: list[ToolCall]):
        st.internal.append(Message.assistant("", calls).model_dump(mode="json"))
        for c in calls:
            st.name_by_call_id[c.id] = c.name
        return AssistantMessage(role="assistant", content=None,
                                tool_calls=[TauToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in calls])

    # ---------------------------------------------------------------- conversions
    def _ingest(self, st: GuardedState, m) -> None:
        if isinstance(m, MultiToolMessage):
            for tm in m.tool_messages:
                self._ingest(st, tm)
        elif isinstance(m, ToolMessage):
            name = st.name_by_call_id.get(m.id, "tool")
            content = m.content or ""
            flags = screen(content, f"tool:{name}")
            st.guard_events += [{"type": "injection_flag", **f.to_dict()} for f in flags]
            st.internal.append(Message(role="tool", content=content, tool_call_id=m.id, name=name,
                                       is_error=bool(m.error),
                                       meta={"injection_flags": [f.to_dict() for f in flags]}).model_dump(mode="json"))
        elif isinstance(m, UserMessage):
            st.internal.append(Message.user(m.content or "").model_dump(mode="json"))
        elif isinstance(m, AssistantMessage):
            calls = [ToolCall(id=c.id or ToolCall(name=c.name).id, name=c.name, arguments=c.arguments)
                     for c in (m.tool_calls or [])]
            for c in calls:
                st.name_by_call_id[c.id] = c.name
            st.internal.append(Message.assistant(m.content or "", calls).model_dump(mode="json"))

    def _flush(self, st: GuardedState) -> None:
        """Rewrite this simulation's sidecar row (last write wins; keyed by task + trial index)."""
        if not self.task_id:
            return
        row = {"task_id": self.task_id, "trial": self._trial, "guard_events": st.guard_events,
               "approvals": st.approvals, "blocked_calls": st.blocked_calls}
        with SIDECAR_LOCK:
            p = Path(self.sidecar)
            p.parent.mkdir(parents=True, exist_ok=True)
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
            rows = [r for r in rows if not (r["task_id"] == row["task_id"] and r["trial"] == row["trial"])]
            p.write_text("\n".join(json.dumps(r) for r in rows + [row]) + "\n")

    @property
    def _trial(self) -> int:
        if not hasattr(self, "_trial_idx"):
            with SIDECAR_LOCK:
                self._trial_idx = _trial_counter[self.task_id]
                _trial_counter[self.task_id] += 1
        return self._trial_idx


def create_guarded_agent(tools, domain_policy, **kwargs):
    """Factory for tau2's registry: registry.register_agent_factory(create_guarded_agent, "guarded_react")."""
    import os

    task = kwargs.get("task")
    return GuardedReActAgent(
        tools, domain_policy, llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"),
        domain=os.environ.get("TAU2_DOMAIN", "retail"),
        approval_mode=os.environ.get("BENCH_APPROVAL_MODE", "auto_approve"),
        sidecar=os.environ.get("GUARD_SIDECAR"), task_id=str(getattr(task, "id", "")) or None)


def register() -> None:
    from tau2.registry import registry

    if "guarded_react" not in registry.get_agents():
        registry.register_agent_factory(create_guarded_agent, "guarded_react")
