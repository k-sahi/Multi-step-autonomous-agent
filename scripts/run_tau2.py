"""Run the guarded agent on tau2-bench, then compute our metrics.

Prerequisites
  git clone https://github.com/sierra-research/tau2-bench && cd tau2-bench && uv sync
  (then from this repo, in the same environment) pip install -e /path/to/support-agent
  export ANTHROPIC_API_KEY=...

Examples
  python scripts/run_tau2.py --domain retail  --trials 4 --tasks 20
  python scripts/run_tau2.py --domain airline --trials 4 --agent-llm anthropic/claude-sonnet-5-5
  python scripts/run_tau2.py --domain retail  --baseline        # tau2's stock llm_agent, same settings

What it does
  1. registers our agent factory as "guarded_react" in tau2's registry (in-process)
  2. invokes tau2's CLI `run` with --agent guarded_react (or llm_agent for the baseline)
  3. loads the newest results file, joins the guard sidecar, prints/saves metrics
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail", choices=["retail", "airline", "telecom"])
    ap.add_argument("--trials", type=int, default=4, help="trials per task; pass^k needs k <= trials")
    ap.add_argument("--tasks", type=int, default=None, help="limit number of tasks (cost control)")
    ap.add_argument("--agent-llm", default="anthropic/claude-sonnet-5-5")
    ap.add_argument("--user-llm", default="anthropic/claude-sonnet-5-5")
    ap.add_argument("--approval", default="auto_approve", choices=["auto_approve", "auto_deny"])
    ap.add_argument("--baseline", action="store_true", help="run tau2's stock llm_agent instead")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to `tau2 run`")
    a = ap.parse_args()

    from tau2.cli import main as tau2_main

    from supportagent.bench.tau2_agent import register

    stamp = time.strftime("%Y%m%d-%H%M%S")
    agent = "llm_agent" if a.baseline else "guarded_react"
    sidecar = Path("outputs") / "tau2" / f"{a.domain}-{agent}-{stamp}.guard.jsonl"
    os.environ.update(TAU2_DOMAIN=a.domain, BENCH_APPROVAL_MODE=a.approval, GUARD_SIDECAR=str(sidecar))
    if not a.baseline:
        register()

    argv = ["tau2", "run", "--domain", a.domain, "--agent", agent, "--agent-llm", a.agent_llm,
            "--user-llm", a.user_llm, "--num-trials", str(a.trials)]
    if a.tasks:
        argv += ["--num-tasks", str(a.tasks)]
    sys.argv = argv + a.extra
    t0 = time.time()
    try:
        tau2_main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    sims = sorted(Path("data/simulations").glob("*.json"), key=lambda p: p.stat().st_mtime)
    sims = [p for p in sims if p.stat().st_mtime >= t0 - 1]
    if not sims:
        print("no results file found in data/simulations/")
        return
    from supportagent.evals import failure_breakdown, summarize
    from supportagent.evals.tau2_results import load_tau2_run

    trials = load_tau2_run(sims[-1], sidecar if sidecar.exists() else None)
    report = {"run": str(sims[-1]), "agent": agent, "domain": a.domain, "summary": summarize(trials),
              "failures": failure_breakdown(trials)}
    out = Path("outputs") / "tau2" / f"{a.domain}-{agent}-{stamp}.metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"\nmetrics -> {out}")


if __name__ == "__main__":
    main()
