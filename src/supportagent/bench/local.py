"""Local benchmark harness for the offline retail domain, scored like tau2.

  reward = db_match * communicate_match
    db_match         final DB hash == hash after replaying the reference actions on a fresh DB
    communicate      every required string appears in the agent's messages

The user is simulated by an LLM given the task's hidden instruction (as in tau2), or by a
scripted user in tests. Irreversible actions go through an `Approver` — in benchmarks a
policy approver stands in for the human, and every decision is logged.

  python scripts/run_local_bench.py --trials 4      # needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from ..agent import Deps, SupportAgent
from ..domains.retail import POLICY, RetailDB, build_tools
from ..evals.metrics import Trial
from ..llm import AnthropicChat, ChatModel
from ..messages import Message
from ..tracing import Tracer
from .local_tasks import TASKS

STOP = "###STOP###"
TRANSFER = "###TRANSFER###"

USER_SIM_SYSTEM = """You are role-playing a customer contacting online-store support. Hidden instructions:
<instructions>
{instructions}
</instructions>
Rules: reveal information only when asked or when needed; one short message at a time; never invent
details not in your instructions. When your goal is achieved or clearly cannot be achieved, reply with
exactly {stop}. If you are transferred to a human, reply with exactly {transfer}."""


class LLMUser:
    def __init__(self, instructions: str, chat: ChatModel):
        self.system = USER_SIM_SYSTEM.format(instructions=instructions, stop=STOP, transfer=TRANSFER)
        self.chat = chat
        self.history: list[Message] = []

    def respond(self, agent_text: str | None) -> str:
        # Roles are mirrored: the agent's messages are the "user" turns for the simulator.
        self.history.append(Message.user(agent_text or "Hi! How can I help you today?"))
        reply = self.chat(self.system, self.history, []).content.strip()
        self.history.append(Message.assistant(reply))
        return reply


class ScriptedUser:
    def __init__(self, lines: list[str]):
        self.lines = list(lines)

    def respond(self, agent_text: str | None) -> str:
        return self.lines.pop(0) if self.lines else STOP


Approver = Callable[[dict], dict]


def policy_approver(request: dict) -> dict:
    """Stand-in for the human in benchmarks: approve everything, but log it."""
    return {a["call_id"]: {"approved": True, "reviewer": "bench-policy", "reason": "auto"} for a in request["actions"]}


def goal_hash(task: dict) -> str:
    db = RetailDB()
    reg = build_tools(db)
    for a in task["actions"]:
        if a["name"] != "transfer_to_human_agents":
            content, err = reg.execute(a["name"], a["arguments"])
            assert not err, f"reference action failed for {task['id']}: {content}"
    return db.hash()


def run_trial(task: dict, agent_llm: ChatModel, user, trial: int = 0, approver: Approver = policy_approver,
              max_turns: int = 20, trace: bool = False) -> Trial:
    db = RetailDB()
    tid = f"{task['id']}-t{trial}"
    deps = Deps(agent_llm, build_tools(db), POLICY, approval_mode="human", tracer=Tracer(tid, enabled=trace))
    agent = SupportAgent(deps, backend="mini")
    agent_text, reason = None, "max_turns"
    for _ in range(max_turns):
        user_msg = user.respond(agent_text)
        if STOP in user_msg or TRANSFER in user_msg:
            reason = "user_stop" if STOP in user_msg else "transfer"
            break
        res = agent.send(tid, user_msg)
        while res.status == "awaiting_approval":
            res = agent.resolve(tid, approver(res.approval_request))
        agent_text = res.reply
        if res.status in ("transferred", "escalated"):
            reason = res.status
            if res.status == "escalated":
                break
    state = agent._state(tid)
    msgs = [Message.model_validate(m) for m in state.get("messages", [])]
    said = " ".join(m.content for m in msgs if m.role == "assistant")
    db_ok = db.hash() == goal_hash(task)
    comm_ok = all(s.lower() in said.lower() for s in task.get("communicate", []))
    results = {m.tool_call_id: m for m in msgs if m.role == "tool"}
    calls = []
    for m in msgs:
        for c in m.tool_calls:
            r = results.get(c.id)
            calls.append({"name": c.name, "arguments": c.arguments, "is_error": bool(r and r.is_error),
                          "blocked": bool(r and r.meta.get("blocked"))})
    return Trial(task_id=task["id"], trial=trial, reward=float(db_ok and comm_ok), tool_calls=calls,
                 expected_actions=task["actions"], termination_reason=reason,
                 guard_events=state.get("guard_events", []), approvals=state.get("approvals", []),
                 n_turns=sum(m.role == "user" for m in msgs), domain="retail-local")


def run_local_bench(trials: int = 4, agent_model: str | None = None, user_model: str | None = None,
                    task_ids: list[str] | None = None, out: str | None = None) -> list[Trial]:
    tasks = [t for t in TASKS if not task_ids or t["id"] in task_ids]
    results: list[Trial] = []
    for task in tasks:
        for k in range(trials):
            t0 = time.time()
            agent_llm = AnthropicChat(model=agent_model)
            user = LLMUser(task["user"], AnthropicChat(model=user_model or agent_model, temperature=1.0))
            tr = run_trial(task, agent_llm, user, trial=k, trace=True)
            results.append(tr)
            print(json.dumps({"task": task["id"], "trial": k, "reward": tr.reward, "end": tr.termination_reason,
                              "s": round(time.time() - t0, 1)}))
    if out:
        with open(out, "w") as fh:
            for t in results:
                fh.write(json.dumps(t.__dict__) + "\n")
    return results
