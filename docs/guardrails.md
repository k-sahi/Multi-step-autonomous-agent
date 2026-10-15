# Guardrails

Three layers, in the order a conversation meets them.

## 1. Input: customer messages and tool output (`guardrails/injection.py`)

| Check | Applies to | Action |
|---|---|---|
| Regex screen: override, role change, prompt extraction, fake tags, tool steering, bypass ("refund without confirmation") | each customer message (once per turn), every tool result | `injection_flag` event; never blocks the customer by itself |
| Tool-output wrapping | every tool result sent to the model | `<tool_result name=…>` delimiters; flagged output gets an explicit "treat as data" note |
| Optional LLM classifier (`llm_screen`) | long third-party text | can add a flag, never remove one |
| PII redaction | everything written to traces / Arize | emails, card numbers, phone numbers masked |

The real defence against injection is layer 2: confirmations must come from a **customer message**, and
a tool result saying "the customer approved" is ignored (`tests/test_guardrails.py::test_tool_output_cannot_confirm`).

## 2. Tool calls (`guardrails/policy.py`)

Each proposed call gets one decision: `allow`, `block` (reason returned to the model), `approve` (operator
sign-off), or `escalate`. Rules run cheapest-first; the first rule that fires decides.

| # | Rule | Scope | Fires when | Decision |
|---|---|---|---|---|
| 1 | `unknown_tool` | all | tool not in registry | block |
| 2 | `schema` | all | missing/extra/mistyped args, enum violation | block |
| 3 | `loop` | all | same tool + same args called more than `MAX_IDENTICAL_CALLS` times | escalate |
| 4 | `ungrounded_id` | all | an `*_id` argument never appeared in customer text or a successful tool result | block |
| 5 | `auth_required` | retail | non-public tool before the customer is authenticated via lookup | block |
| 6 | `other_user` | retail | `user_id` ≠ authenticated user | block |
| 7 | `order_not_owned` / `payment_not_owned` | retail | order or payment method not in the authenticated user's profile | block |
| 8 | `needs_user_confirmation` | writes | last customer message isn't an explicit yes (or is hedged: "wait", "not yet") | block |
| 9 | `confirmation_mismatch` | writes | the agent's message before the yes didn't name this order/reservation | block |
| 10 | `one_write_per_confirmation` | writes | a change already happened since the last customer message | block |
| 11 | `one_write_at_a_time` | writes | more than one write in a single model response | block extras |
| 12 | `irreversible` | `approve` tools | everything above passed | approve (operator) |

Plus graph-level budgets: `MAX_STEPS_PER_TURN` (step budget), `MAX_BLOCKED_RETRIES` (consecutive
fully-blocked steps) → escalate.

**Domain scoping.** Identity rules (5–7) are retail-specific; tau2 airline and telecom identify customers
differently, so only the generic and write rules apply there until domain rules are added
(`DOMAIN_RULES`).

**Hard invariants stay in the tools.** "Only pending orders can be cancelled" is enforced by the tool
itself, because it's true regardless of conversation. Guards enforce rules that depend on the
conversation, which a tool can't see.

## 3. Output (`guardrails/output.py`)

| Check | Fires when | Action |
|---|---|---|
| `unbacked_claim` | reply says something *was* cancelled/refunded/updated… but no write succeeded this turn (future tense and questions are exempt) | one ephemeral rewrite with feedback; the rejected draft is never shown or stored |
| `data_leak` | reply contains another customer's email seen in tool output | rewrite |
| `too_long` | > 4000 characters | rewrite |

If the rewrite still violates, it is sent and logged as `output_violation_committed`, which shows up in
evals; we prefer a visible metric to an infinite loop.

## Measuring the guardrails

* **Block rate by rule** and **blocked-but-succeeded** vs **blocked-and-failed** trials (`evals/metrics.guard_stats`).
* **False blocks**: sample blocked calls from successful reference trajectories in tau2; a guard that blocks a
  reference action is wrong.
* **Ablation**: run tau2 with guards on vs `ToolGuard` replaced by allow-all, same tasks and k; report
  success, pass^k and unsafe-action counts (writes without confirmation, writes on other users' data).
