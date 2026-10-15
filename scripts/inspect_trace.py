"""Render a trace file as a step-by-step timeline — the first tool for debugging a failure.

  python scripts/inspect_trace.py traces/demo/demo-cancel.jsonl
  python scripts/inspect_trace.py traces/<thread>.jsonl --only guard,tool --full

Each row: time offset, span, duration, and the attributes that matter for that span type:
  llm    -> which tools the model proposed, or the first part of its text
  guard  -> decision per call (tool:action:rule)
  tool.* -> arguments, error flag, output preview
  event  -> approvals (who, what, approved?) and escalations (why)
A summary at the end counts blocks by rule and flags patterns worth a closer look
(repeated identical calls, consecutive blocks, slow LLM steps, denied approvals).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportagent.tracing import load_trace  # noqa: E402

C = {"llm": "\033[36m", "guard": "\033[33m", "tool": "\033[32m", "event": "\033[35m", "err": "\033[31m", "end": "\033[0m"}


def color(kind: str, s: str, on: bool) -> str:
    return f"{C.get(kind, '')}{s}{C['end']}" if on else s


def describe(r: dict, full: bool) -> tuple[str, str]:
    a = r.get("attrs", {})
    cut = (lambda s: s) if full else (lambda s: (s[:110] + "…") if len(s) > 110 else s)
    if r["type"] == "event":
        if r["name"] == "approval":
            ds = a.get("decisions", [])
            return "event", "APPROVAL " + "; ".join(
                f"{d['tool']} -> {'approved' if d['approved'] else 'DENIED'} by {d['reviewer']}" for d in ds)
        return "event", f"{r['name'].upper()} {cut(json.dumps(a))}"
    name = r["name"]
    if name == "llm":
        calls = a.get("tool_calls") or []
        body = f"proposes {', '.join(calls)}" if calls else f"says: {cut(a.get('output', ''))!r}"
        return "llm", f"llm[{a.get('model', '')}] {body}"
    if name == "guard":
        return "guard", "guard " + ", ".join(a.get("decisions", []))
    if name.startswith("tool."):
        err = " ERROR" if a.get("is_error") else ""
        return ("err" if err else "tool"), f"{name} {json.dumps(a.get('arguments', {}))}{err} -> {cut(' '.join(str(a.get('output', '')).split()))}"
    return "event", f"{name} {cut(json.dumps(a))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--only", help="comma list: llm,guard,tool,event")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()
    recs = load_trace(a.trace)
    if not recs:
        print("empty trace")
        return
    t0 = min(r.get("start", r.get("ts", 0)) for r in recs)
    only = set(a.only.split(",")) if a.only else None
    use_color = not a.no_color and sys.stdout.isatty()
    blocks, sigs, slow, denied = Counter(), Counter(), [], 0
    print(f"trace {recs[0].get('thread_id')}  ({len(recs)} records)\n")
    for i, r in enumerate(sorted(recs, key=lambda r: r.get("start", r.get("ts", 0)))):
        kind, text = describe(r, a.full)
        base = "tool" if kind in ("tool", "err") else kind
        t = r.get("start", r.get("ts", 0)) - t0
        dur = f"{r.get('duration_s', 0):6.2f}s" if r["type"] == "span" else "       "
        if not only or base in only:
            print(f"{i:3d} +{t:7.2f}s {dur}  {color(kind, text, use_color)}")
        at = r.get("attrs", {})
        for d in at.get("decisions", []) if r.get("name") == "guard" else []:
            tool, action, rule = d.split(":")
            if action == "block":
                blocks[rule] += 1
        if r.get("name", "").startswith("tool."):
            sigs[(r["name"], json.dumps(at.get("arguments", {}), sort_keys=True))] += 1
        if r.get("name") == "llm" and r.get("duration_s", 0) > 10:
            slow.append(i)
        if r.get("name") == "approval":
            denied += sum(not d["approved"] for d in at.get("decisions", []))
    print("\nsummary")
    print(f"  blocks by rule: {dict(blocks) or 'none'}")
    repeats = {f"{k[0]} {k[1]}": v for k, v in sigs.items() if v > 1}
    print(f"  repeated identical tool calls: {repeats or 'none'}")
    print(f"  slow llm steps (>10s): {slow or 'none'}")
    print(f"  denied approvals: {denied}")
    if blocks.get("needs_user_confirmation", 0) > 1:
        print("  hint: model keeps acting before confirming -> strengthen the confirmation instruction or add an example")
    if blocks.get("ungrounded_id"):
        print("  hint: model guessed ids -> check it calls the lookup tool first; consider a tool-description fix")
    if repeats:
        print("  hint: identical retries -> the model is not reading tool errors; check the error text is actionable")


if __name__ == "__main__":
    main()
