# Evaluation

## Benchmark

**tau2-bench** (sierra-research/tau2-bench; current release is published as τ³-bench). Each domain
(retail, airline, telecom) has a policy, tools, a database, and tasks. An LLM user simulator plays the
customer from hidden instructions; the agent talks to it and calls tools. A trial's **reward** is the
product of the components in the task's `reward_basis` (by default DB and COMMUNICATE):

* **DB** — final database hash equals the hash after replaying the reference actions on a fresh DB, so any
  trajectory reaching the same end state passes;
* **COMMUNICATE** — every required string appears in the agent's messages.

The local harness (`bench/local.py`) scores the offline retail domain the same way.

## Metrics

| Metric | Definition | Why |
|---|---|---|
| **Task success** (pass^1) | mean over trials of 1[reward = 1] | headline |
| **pass^k** | per task with n trials and c successes: C(c,k)/C(n,k); mean over tasks | reliability: all k attempts must succeed |
| pass@k | 1 − C(n−c,k)/C(n,k) | contrast only; pass@k − pass^k = inconsistency |
| **Write precision / recall / F1** | executed successful writes vs reference writes, matching name + `compare_args` (normalised) | did it do the right thing |
| Read recall | reference read tools that were called | did it gather what it needed |
| Tool error rate | executed calls that returned an error | arguments / sequencing quality |
| Blocked rate | proposed calls blocked by guards | how often guards intervened |
| Exact write sequence | same number of writes, all matched | strict trajectory check |
| **Escalation rate** | trials that transferred to a human | cost |
| ↳ appropriate / unnecessary / missed | vs whether the reference transfers | quality of the decision to escalate |
| Guard stats | blocks by rule; trials with blocks that succeeded vs failed; approvals approved/denied | guard usefulness and operator load |
| Failure taxonomy | label per failed trial (see `evals/failures.py`) | what to fix first |
| Cost, tool calls per trial | from tau2 `agent_cost`, trajectory | efficiency |

### Why pass^k

With 4 trials per task, an agent with 75 % success could be "always right on 75 % of tasks" (pass^4 = 0.75)
or "right 75 % of the time on every task" (pass^4 ≈ 0.32). Customers experience the second as unreliable.
pass^k separates them; the demo's scripted example shows pass^1 = 0.75 → pass^4 = 0.

### Statistics

* Run **k ≥ 4 trials** per task at the agent's production temperature; pass^k needs k ≤ trials.
* **Bootstrap over tasks**, not trials (trials of the same task are correlated). `compute_metrics.py`
  reports a 95 % CI for success and for paired differences between two runs on the same tasks.
* Compare against tau2's stock `llm_agent` with the **same agent model, user model, tasks and k**
  (`run_tau2.py --baseline`). User simulators are themselves LLMs and occasionally fail; report the same
  user model for both runs and spot-check failures labelled `communication`.

## Running

```bash
python scripts/run_tau2.py --domain retail --trials 4 --tasks 20            # guarded agent
python scripts/run_tau2.py --domain retail --trials 4 --tasks 20 --baseline # stock agent
python scripts/compute_metrics.py data/simulations/<guarded>.json --compare data/simulations/<baseline>.json
python scripts/run_local_bench.py --trials 4                                # fast iteration loop
```

Outputs: `outputs/tau2/<domain>-<agent>-<time>.metrics.json` with `summary` and `failures`.

## Guardrail ablation (recommended experiment)

Same tasks, same k: (a) stock agent, (b) guarded with auto-approve, (c) guarded with auto-deny. (b) vs (a)
shows what the guards cost or gain in success and pass^k; (c) shows graceful behaviour when sign-off is
refused (should tell the customer, not claim success); unsafe-action counts (writes without a preceding
customer yes, actions on non-owned objects) are computed from the trajectories of (a).
