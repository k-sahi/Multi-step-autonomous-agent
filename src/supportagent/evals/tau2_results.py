"""Load tau2-bench result files into `Trial` objects.

tau2 writes one JSON per run under data/simulations/ with (among others):
  tasks[]:        id, evaluation_criteria.actions[] {name, arguments, compare_args?}
  simulations[]:  task_id, trial, reward_info.reward, messages[] {role, content, tool_calls[]},
                  termination_reason, agent_cost

The parser is deliberately tolerant (uses .get everywhere): tau2's schema evolves between
releases (tau2 -> tau3), and an eval script that crashes on a renamed field is worse than
one that reports a field as missing. Check `src/tau2/data_model/simulation.py` of the
version you run if a metric comes back empty.

Our guarded agent also writes a sidecar file (<run>.guard.jsonl, one row per
simulation) with guard events and approvals, joined here on (task_id, trial).
"""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import Trial


def _calls(messages: list[dict]) -> list[dict]:
    out, by_id = [], {}
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            for c in m.get("tool_calls") or []:
                rec = {"name": c.get("name"), "arguments": c.get("arguments") or {}, "id": c.get("id"),
                       "is_error": False}
                out.append(rec)
                if c.get("id"):
                    by_id[c["id"]] = rec
        elif role == "tool":
            rec = by_id.get(m.get("id") or m.get("tool_call_id"))
            if rec is not None:
                rec["is_error"] = bool(m.get("error") or m.get("is_error"))
    return out


def load_tau2_run(path: str | Path, sidecar: str | Path | None = None) -> list[Trial]:
    path = Path(path)
    data = json.loads(path.read_text())
    domain = (data.get("info") or {}).get("environment_info", {}).get("domain_name", "")
    tasks = {str(t.get("id")): t for t in data.get("tasks", [])}
    guard: dict[tuple[str, int], dict] = {}
    side = Path(sidecar) if sidecar else path.with_suffix(".guard.jsonl")
    if side.exists():
        for line in side.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                guard[(str(row["task_id"]), int(row.get("trial", 0)))] = row
    trials = []
    for sim in data.get("simulations", []):
        tid = str(sim.get("task_id"))
        task = tasks.get(tid, {})
        crit = task.get("evaluation_criteria") or {}
        msgs = sim.get("messages") or []
        g = guard.get((tid, int(sim.get("trial", 0))), {})
        calls = _calls(msgs)
        for c in g.get("blocked_calls", []):
            calls.append({**c, "blocked": True})
        trials.append(Trial(
            task_id=tid, trial=int(sim.get("trial", 0)),
            reward=float((sim.get("reward_info") or {}).get("reward", 0.0)),
            tool_calls=calls,
            expected_actions=[{"name": a.get("name"), "arguments": a.get("arguments") or {},
                               "compare_args": a.get("compare_args")} for a in crit.get("actions") or []],
            termination_reason=str(sim.get("termination_reason", "")),
            guard_events=g.get("guard_events", []), approvals=g.get("approvals", []),
            n_turns=sum(1 for m in msgs if m.get("role") == "user"),
            cost=sim.get("agent_cost"), domain=domain,
        ))
    return trials
