"""Compute metrics from an existing tau2 results file or a local-bench JSONL, and optionally
compare two runs (e.g. guarded agent vs tau2's stock agent) with task-level bootstrap CIs.

  python scripts/compute_metrics.py data/simulations/<run>.json
  python scripts/compute_metrics.py outputs/local_bench.jsonl
  python scripts/compute_metrics.py A.json --compare B.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supportagent.evals import failure_breakdown, summarize  # noqa: E402
from supportagent.evals.metrics import Trial, bootstrap_ci, by_task  # noqa: E402
from supportagent.evals.tau2_results import load_tau2_run  # noqa: E402


def load(path: str) -> list[Trial]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [Trial(**json.loads(l)) for l in p.read_text().splitlines() if l.strip()]
    return load_tau2_run(p)


def task_success(trials: list[Trial]) -> dict[str, float]:
    return {k: statistics.mean(t.success for t in v) for k, v in by_task(trials).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--compare")
    ap.add_argument("--max-k", type=int)
    a = ap.parse_args()
    A = load(a.run)
    rep = {"summary": summarize(A, a.max_k), "failures": failure_breakdown(A)}
    sa = task_success(A)
    rep["summary"]["task_success_95ci"] = bootstrap_ci(list(sa.values()))
    if a.compare:
        B = load(a.compare)
        sb = task_success(B)
        common = sorted(set(sa) & set(sb))
        diffs = [sa[t] - sb[t] for t in common]
        rep["comparison"] = {"tasks_in_common": len(common),
                             "success_A": round(statistics.mean(sa[t] for t in common), 4),
                             "success_B": round(statistics.mean(sb[t] for t in common), 4),
                             "diff_A_minus_B": round(statistics.mean(diffs), 4),
                             "diff_95ci": bootstrap_ci(diffs),
                             "B_summary": summarize(B, a.max_k)}
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
