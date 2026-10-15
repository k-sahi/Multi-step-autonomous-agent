"""Run the local retail benchmark (7 tasks) with an LLM agent and LLM user simulator.

  python scripts/run_local_bench.py --trials 4 --out outputs/local_bench.jsonl
  python scripts/compute_metrics.py outputs/local_bench.jsonl

Cheap and fast compared with tau2 (seconds per trial); use it for iteration on prompts and
guardrails, and tau2 for reported numbers.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supportagent.bench.local import run_local_bench  # noqa: E402
from supportagent.evals import failure_breakdown, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--agent-model")
    ap.add_argument("--user-model")
    ap.add_argument("--tasks", help="comma-separated task ids")
    ap.add_argument("--out", default="outputs/local_bench.jsonl")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    trials = run_local_bench(a.trials, a.agent_model, a.user_model,
                             a.tasks.split(",") if a.tasks else None, a.out)
    print(json.dumps({"summary": summarize(trials), "failures": failure_breakdown(trials)}, indent=1))


if __name__ == "__main__":
    main()
