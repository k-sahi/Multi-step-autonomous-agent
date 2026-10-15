# Design decisions

The four things this project is meant to show, with reasoning and alternatives.

---

## 1. Where the human sits

There are **two humans** in a support conversation, and they authorise different things.

| Human | Authorises | Mechanism | Why |
|---|---|---|---|
| **Customer** | *intent*: "yes, cancel order #W2378156" | `r_confirmation` guard: the last customer message is an explicit yes, the agent's message before it named the same object, and no other change happened since | The policy requires consent; the guard enforces it in code, so the model cannot skip it however it is prompted |
| **Operator** (support lead) | *risk*: irreversible money or booking changes | `approval` node → LangGraph `interrupt` → approval queue in the API → `Command(resume=…)` | Catches what the customer can't: fraud, policy edge cases, model errors that are consistent with a customer's yes |

**Which actions need the operator?** Only tools marked `approve` — irreversible writes (cancel, return,
exchange, item modification; in airline: cancel/book reservation, certificates). Reversible writes
(address change) need the customer's yes only. Reads need neither. Putting every write behind an
operator would make the operator a rubber stamp; putting none would leave refunds to a probabilistic model.

**What the operator sees.** The interrupt payload: tool, exact arguments, the risk note, and a digest of
the last few turns. The operator approves or denies each action with a reason; the decision, reviewer
and time go into `approvals` (state + trace + Temporal history).

**Failure behaviour.**
* Denied → the call becomes a tool error ("a supervisor declined… offer alternatives"), the agent tells the
  customer, nothing executes.
* Nobody decides → with Temporal, the workflow times out and resumes with a denial (fail safe). Without
  Temporal, the conversation stays paused in the checkpointer and the API refuses new customer messages
  with 409 until a decision is made.
* The agent is stuck → automatic escalation to a human agent with a summary.

**In benchmarks** nobody is there to click, so a logged policy approves (or denies) in place of the
operator. The approvals count is reported, which is the operator workload the agent would create.

**Alternatives considered.** Approve-every-tool-call (unusable, approvers stop reading); confidence-based
approval (an LLM's self-reported confidence is poorly calibrated); approval after execution with rollback
(not possible for refunds and bookings). Risk-tiered static metadata is simple, auditable, and testable.

---

## 2. Guardrails

Principles:
1. **Deterministic first.** Policy is enforced by code that inspects the conversation log, not by asking
   the model to behave. Code is fast, free, testable and can't be talked out of a rule.
2. **Guard the action, not the words.** A user can persuade the model to *propose* anything; only guarded
   actions *execute*. That makes prompt injection a nuisance rather than a breach.
3. **Block with an actionable reason.** Blocked calls return to the model as tool errors that say what to
   do ("authenticate first", "ask for explicit confirmation"), which lets the agent recover in-turn.
4. **Fail closed, but escalate rather than loop.** Unknown tools default to "write needing confirmation";
   repeated blocks, identical-call loops and step budgets escalate to a human.
5. **LLM checks may only tighten.** The optional LLM injection classifier can add a flag; it can never
   clear one.

Full rule list in [guardrails.md](guardrails.md).

---

## 3. Evals

tau2-bench scores a trial as success only if the final database equals the goal state **and** required
information was communicated. That is the headline (`task_success`, = pass^1). Around it:

* **pass^k** — all of k trials succeed. A support agent that solves a task 3 times out of 4 fails one
  customer in four; pass^k exposes that, pass@k hides it. We report the curve k = 1…n and the
  pass@k − pass^k gap.
* **Tool accuracy** — write precision/recall/F1 against reference actions (with tau2's `compare_args`),
  read-tool recall, error rate, guard-block rate. It separates "did the wrong thing" from "said the wrong
  thing".
* **Escalation rate** — split into appropriate, unnecessary and missed. Unnecessary escalations cost
  money and look like success to a naive guard; missed escalations strand customers.
* **Guardrail stats** — blocks by rule, and how often blocked trials still succeed (a guard that blocks
  and the agent recovers is working; blocks concentrated in failures point at a prompt or tool fix).
* **Failure taxonomy** — every failed trial gets a label (wrong arguments, no action, missed escalation,
  communication…), the bridge to trace debugging.
* **Uncertainty** — task-level bootstrap CIs (trials of one task are correlated, so resample tasks), and
  paired comparisons against tau2's stock agent on the same tasks.

---

## 4. Debugging with traces

Every LLM call, guard decision, tool call, approval and escalation is a span (local JSONL + Arize via
OpenInference). The loop: metrics → failure taxonomy → open the trace for a failed trial →
`inspect_trace.py` timeline → find the first wrong step → classify the root cause (prompt, tool
description, guard rule, user simulator) → fix → re-run the same tasks with the same k and compare.
Worked examples in [debugging.md](debugging.md).

---

## 5. Smaller decisions

| Decision | Why |
|---|---|
| LangGraph for orchestration | Explicit graph (auditable control flow), first-class `interrupt` + checkpointers for human approval, streaming, and the state can be inspected at every step |
| MCP for tools | Tools become a separately deployable, reusable service with standard risk annotations; the agent can use any MCP server and the tool server can serve any MCP client |
| Temporal optional | Checkpoints make pauses *resumable*; Temporal makes them *timed* and *supervised* (timeouts, retries, history). Not needed for a demo; valuable when approvals take hours |
| Sync graph nodes | Simpler checkpoint/trace semantics; the async MCP client lives on its own loop thread |
| `MiniGraph` | Tests and offline demo run the real nodes with LangGraph's interrupt semantics and no dependencies; production uses LangGraph |
| Stateless MCP HTTP | Cloud Run can route any request to any instance |
| One write per confirmation | Mirrors tau2 policy and stops a single "yes" from authorising a chain of changes |
