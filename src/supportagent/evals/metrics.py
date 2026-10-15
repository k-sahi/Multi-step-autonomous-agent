"""Agent metrics: task success, pass^k, tool accuracy, escalation rate, guardrail stats.

Definitions (docs/evaluation.md has the full rationale):

  task success   reward == 1 for a trial (tau2: final DB matches the goal state AND every
                 required piece of information was communicated). Reported as the mean over
                 all trials = pass^1.
  pass^k         probability that ALL k independent trials of the same task succeed.
                 Unbiased estimate per task with n trials, c successes: C(c,k) / C(n,k);
                 averaged over tasks. Measures *reliability*, which pass@k hides.
  pass@k         probability that AT LEAST one of k trials succeeds: 1 - C(n-c,k)/C(n,k).
                 Reported only for contrast: pass@k - pass^k is the consistency gap.
  tool accuracy  write actions compared with the task's reference actions:
                 precision / recall / F1 on (tool name, key arguments), plus read-tool recall,
                 tool error rate and guard-block rate.
  escalation     share of trials that transferred to a human, split into appropriate
                 (the reference solution transfers) and unnecessary; plus missed escalations.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import comb
from typing import Any

READ_PREFIXES = ("get_", "find_", "list_", "search", "check", "calculate", "think")


@dataclass
class Trial:
    task_id: str
    trial: int
    reward: float
    tool_calls: list[dict] = field(default_factory=list)     # [{"name","arguments","is_error","blocked"}]
    expected_actions: list[dict] = field(default_factory=list)  # [{"name","arguments","compare_args"?}]
    termination_reason: str = ""
    guard_events: list[dict] = field(default_factory=list)
    approvals: list[dict] = field(default_factory=list)
    n_turns: int = 0
    cost: float | None = None
    domain: str = ""

    @property
    def success(self) -> bool:
        return self.reward >= 1.0 - 1e-6


# ----------------------------------------------------------------- pass^k
def pass_hat_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k={k} > n={n} trials")
    return comb(c, k) / comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k={k} > n={n} trials")
    return 1.0 - comb(n - c, k) / comb(n, k)


def by_task(trials: list[Trial]) -> dict[str, list[Trial]]:
    out: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        out[t.task_id].append(t)
    return out


def pass_k_curve(trials: list[Trial], max_k: int | None = None) -> dict[str, dict[int, float]]:
    groups = by_task(trials)
    n_min = min(len(v) for v in groups.values())
    max_k = min(max_k or n_min, n_min)
    hat, at = {}, {}
    for k in range(1, max_k + 1):
        hat[k] = statistics.mean(pass_hat_k(len(v), sum(t.success for t in v), k) for v in groups.values())
        at[k] = statistics.mean(pass_at_k(len(v), sum(t.success for t in v), k) for v in groups.values())
    return {"pass^k": {k: round(v, 4) for k, v in hat.items()}, "pass@k": {k: round(v, 4) for k, v in at.items()}}


# ----------------------------------------------------------------- tool accuracy
def is_read(name: str) -> bool:
    return name.startswith(READ_PREFIXES)


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        return sorted(_norm(x) for x in v)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def action_matches(call: dict, expected: dict) -> bool:
    if call["name"] != expected["name"]:
        return False
    keys = expected.get("compare_args")
    if keys is None:
        keys = list(expected.get("arguments", {}))
    return all(_norm(call.get("arguments", {}).get(k)) == _norm(expected["arguments"].get(k)) for k in keys)


def tool_accuracy(trial: Trial) -> dict:
    executed = [c for c in trial.tool_calls if not c.get("blocked")]
    exp_w = [a for a in trial.expected_actions if not is_read(a["name"])]
    got_w = [c for c in executed if not is_read(c["name"]) and not c.get("is_error")]
    matched_exp, used = 0, set()
    for e in exp_w:
        for i, c in enumerate(got_w):
            if i not in used and action_matches(c, e):
                used.add(i)
                matched_exp += 1
                break
    exp_r = {a["name"] for a in trial.expected_actions if is_read(a["name"])}
    got_r = {c["name"] for c in executed if is_read(c["name"])}
    p = len(used) / len(got_w) if got_w else (1.0 if not exp_w else 0.0)
    r = matched_exp / len(exp_w) if exp_w else (1.0 if not got_w else 0.0)
    return {
        "write_precision": p, "write_recall": r, "write_f1": 2 * p * r / (p + r) if p + r else 0.0,
        "exact_write_sequence": len(got_w) == len(exp_w) == len(used),
        "read_recall": len(exp_r & got_r) / len(exp_r) if exp_r else 1.0,
        "tool_error_rate": sum(bool(c.get("is_error")) for c in executed) / len(executed) if executed else 0.0,
        "blocked_rate": sum(bool(c.get("blocked")) for c in trial.tool_calls) / len(trial.tool_calls)
        if trial.tool_calls else 0.0,
        "n_calls": len(trial.tool_calls),
    }


# ----------------------------------------------------------------- escalation
def transferred(trial: Trial) -> bool:
    return any(c["name"].startswith("transfer_to_human") and not c.get("blocked") for c in trial.tool_calls)


def expects_transfer(trial: Trial) -> bool:
    return any(a["name"].startswith("transfer_to_human") for a in trial.expected_actions)


def escalation_stats(trials: list[Trial]) -> dict:
    n = len(trials)
    esc = [t for t in trials if transferred(t)]
    appropriate = [t for t in esc if expects_transfer(t)]
    missed = [t for t in trials if expects_transfer(t) and not transferred(t)]
    auto = [t for t in esc if any(e.get("type") == "escalation" for e in t.guard_events)]
    return {
        "escalation_rate": round(len(esc) / n, 4) if n else 0.0,
        "appropriate_escalation_rate": round(len(appropriate) / len(esc), 4) if esc else None,
        "unnecessary_escalations": len(esc) - len(appropriate),
        "missed_escalations": len(missed),
        "guard_triggered_escalations": len(auto),
    }


# ----------------------------------------------------------------- guardrails
def guard_stats(trials: list[Trial]) -> dict:
    rules = Counter()
    blocked_in_success = blocked_in_failure = 0
    approvals = Counter()
    for t in trials:
        blocks = [e for e in t.guard_events if e.get("type") in ("tool_blocked", "output_blocked")]
        rules.update(e.get("rule", "?") for e in blocks)
        if blocks:
            if t.success:
                blocked_in_success += 1
            else:
                blocked_in_failure += 1
        approvals.update("approved" if a.get("approved") else "denied" for a in t.approvals)
    return {"blocks_by_rule": dict(rules.most_common()), "trials_with_blocks_success": blocked_in_success,
            "trials_with_blocks_failure": blocked_in_failure, "approvals": dict(approvals)}


# ----------------------------------------------------------------- report
def summarize(trials: list[Trial], max_k: int | None = None) -> dict:
    if not trials:
        return {"n_trials": 0}
    acc = [tool_accuracy(t) for t in trials]
    mean = lambda key: round(statistics.mean(a[key] for a in acc), 4)  # noqa: E731
    groups = by_task(trials)
    out = {
        "n_tasks": len(groups),
        "n_trials": len(trials),
        "trials_per_task": sorted({len(v) for v in groups.values()}),
        "task_success_rate": round(statistics.mean(t.success for t in trials), 4),
        **pass_k_curve(trials, max_k),
        "tool_accuracy": {k: mean(k) for k in ("write_precision", "write_recall", "write_f1", "read_recall",
                                                "tool_error_rate", "blocked_rate")},
        "exact_write_sequence_rate": round(statistics.mean(a["exact_write_sequence"] for a in acc), 4),
        "escalation": escalation_stats(trials),
        "guardrails": guard_stats(trials),
        "termination_reasons": dict(Counter(t.termination_reason or "?" for t in trials)),
        "mean_tool_calls": round(statistics.mean(a["n_calls"] for a in acc), 2),
    }
    costs = [t.cost for t in trials if t.cost is not None]
    if costs:
        out["mean_cost_usd"] = round(statistics.mean(costs), 4)
    return out


def bootstrap_ci(values: list[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Task-level bootstrap CI (resample tasks, not trials: trials of one task are correlated)."""
    import random

    rng = random.Random(seed)
    stats = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(iters))
    return round(stats[int(alpha / 2 * iters)], 4), round(stats[int((1 - alpha / 2) * iters) - 1], 4)
