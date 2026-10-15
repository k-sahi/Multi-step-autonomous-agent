# Debugging with traces

## What gets recorded

| Span / event | Attributes |
|---|---|
| `llm` (LLM) | model, number of messages, latency, proposed tool names or first 500 chars of text |
| `guard` | one `tool:action:rule` string per proposed call |
| `tool.<name>` (TOOL) | arguments, is_error, first 500 chars of output |
| `approval` (event) | tool, approved, reviewer, reason per action |
| `escalation` (event) | reason |

Locally: `traces/<thread_id>.jsonl`. In Arize: the same spans with OpenInference attributes
(`openinference.span.kind`, `llm.model_name`, `tool.name`, `session.id`) so a conversation is one session.
All PII is redacted before export.

## The loop

1. **Metrics** say *how often* (success, pass^k, tool F1).
2. **Failure taxonomy** says *what kind*: `python scripts/compute_metrics.py run.json` → `failures.by_label`.
3. **Pick the biggest label**, open 3–5 traces of it:
   `python scripts/inspect_trace.py traces/<thread>.jsonl` (or filter by `--only guard,tool`).
4. **Find the first wrong step** — not the last. Most failures are decided early (wrong order looked up,
   wrong assumption) and only become visible at the final write.
5. **Classify the root cause**: prompt/policy reading, tool description, guard rule (false block), tool
   bug, or user-simulator error (then it isn't the agent's failure — note it, don't "fix" the agent for it).
6. **Fix one thing**, re-run the same tasks with the same k, compare with paired bootstrap.

## Patterns and what they usually mean

| Trace pattern | Likely cause | Fix tried first |
|---|---|---|
| `guard … block:needs_user_confirmation` repeated in one turn | model acts before asking | system prompt: "state the action, then ask"; the guard's block reason already says this, so check the model is reading tool errors |
| `block:ungrounded_id` | model guessed an id (often a plausible order or item id) | tool description: "ids come from get_user_details / get_order_details"; the guard turns this into a recoverable error |
| identical `tool.X` calls with errors | model ignores error text | make the tool error actionable; the loop rule escalates after the limit |
| write executed with the right tool, wrong item ids | read the wrong variant from `get_product_details` | check that the product output is clear; add an example to the policy |
| `output_blocked: unbacked_claim` | model narrates success before the tool runs | usually harmless after the rewrite; if frequent, the model is skipping the tool call |
| `escalation: step budget` | the agent wanders through reads | check for a missing tool (task needs a capability it doesn't have) or vague policy |
| success but many `approval` events | operator load | consider whether some tools are really reversible and could drop to customer-confirm only |

## Worked example (from the offline demo)

```
$ python scripts/inspect_trace.py traces/demo/demo-cancel.jsonl
  9  llm[scripted] proposes cancel_pending_order
 10  guard cancel_pending_order:block:needs_user_confirmation      ← acted before asking
 11  llm[scripted] says: "Order #W2378156 … Shall I proceed?"       ← recovered from the block reason
 12  llm[scripted] proposes cancel_pending_order
 13  guard cancel_pending_order:approve:irreversible               ← customer said yes
 14  APPROVAL cancel_pending_order -> approved by alice
 15  tool.cancel_pending_order {...} -> {"status": "cancelled", ...}
summary: blocks by rule: {'needs_user_confirmation': 1}
```
Reading it: one block, recovered in the same turn, approved by a named reviewer, executed once. In a real
run, a high count of this block across trials would point at the system prompt.

## In Arize

Filter spans with `name = "guard"` and attribute containing `block`, group by rule; open the session to see
the surrounding steps. Attach eval results (success, failure label) to the session to slice failures by
label, model version, or prompt version.
