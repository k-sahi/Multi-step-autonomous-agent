"""Failure taxonomy: label every failed trial with the most likely cause.

This is the bridge between metrics and debugging. A success rate tells you *that* the
agent fails; the taxonomy tells you *which* fix to try first, and the trace inspector
(scripts/inspect_trace.py) shows the exact step where it went wrong.

Labels are assigned by ordered heuristics on the trajectory vs the reference actions.
They are triage hints, not ground truth: the protocol in docs/debugging.md has a human
confirm labels on a sample, and the agreement rate is reported.
"""
from __future__ import annotations

from collections import Counter

from .metrics import Trial, action_matches, expects_transfer, is_read, transferred

LABELS = {
    "guard_escalation": "a guardrail escalated (loop / step budget / repeated blocks)",
    "approval_denied": "the operator declined an irreversible action",
    "missed_escalation": "should have transferred to a human but did not",
    "unnecessary_escalation": "transferred although the task was solvable",
    "no_action": "expected an account change, made none",
    "wrong_arguments": "right tool, wrong arguments (ids, items, payment method, reason)",
    "wrong_tool": "made a change with a tool the reference does not use",
    "extra_action": "all expected changes made, plus an extra one",
    "incomplete": "made some but not all expected changes",
    "communication": "DB state right but required information not communicated (or user-sim issue)",
    "max_steps_or_error": "simulation ended by step limit or error",
}


def classify_failure(t: Trial) -> str:
    if t.success:
        return "success"
    if any(e.get("type") == "escalation" for e in t.guard_events):
        return "guard_escalation"
    if any(not a.get("approved") for a in t.approvals):
        return "approval_denied"
    if t.termination_reason and any(s in t.termination_reason.lower() for s in ("max", "error", "too_many")):
        return "max_steps_or_error"
    if expects_transfer(t) and not transferred(t):
        return "missed_escalation"
    if transferred(t) and not expects_transfer(t):
        return "unnecessary_escalation"
    exp_w = [a for a in t.expected_actions if not is_read(a["name"]) and not a["name"].startswith("transfer")]
    got_w = [c for c in t.tool_calls if not is_read(c["name"]) and not c.get("blocked") and not c.get("is_error")
             and not c["name"].startswith("transfer")]
    if exp_w and not got_w:
        return "no_action"
    matched = [e for e in exp_w if any(action_matches(c, e) for c in got_w)]
    exp_names = {e["name"] for e in exp_w}
    if got_w and not matched:
        return "wrong_arguments" if {c["name"] for c in got_w} & exp_names else "wrong_tool"
    if len(matched) < len(exp_w):
        return "incomplete"
    if len(got_w) > len(exp_w):
        return "extra_action"
    return "communication"


def failure_breakdown(trials: list[Trial]) -> dict:
    labels = Counter(classify_failure(t) for t in trials if not t.success)
    total = sum(labels.values())
    return {"n_failures": total,
            "by_label": {k: {"count": v, "share": round(v / total, 3), "meaning": LABELS.get(k, "")}
                         for k, v in labels.most_common()}}
