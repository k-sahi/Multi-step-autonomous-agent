# Architecture

## 1. The loop

A ReAct agent alternates **reason → act → observe** until it can answer. Here each turn of
that loop is a LangGraph node, and two extra nodes sit between "the model wants to act" and
"the action happens": a guard and (for irreversible actions) a human approval.

| Node | Input | Does | Next |
|---|---|---|---|
| `agent` | message log | calls the LLM with policy + tools; counts steps; screens the new customer message | `guard` if tool calls, `output_guard` if text, `escalate` if over budget |
| `guard` | proposed tool calls | rebuilds facts from the log, runs the rules, writes blocked calls back as tool errors | `approval` / `tools` / `agent` (all blocked) / `escalate` |
| `approval` | `approve` decisions | `interrupt(request)` → waits for an operator; denied calls become tool errors | `tools` or `agent` |
| `tools` | allowed calls | executes via the registry or the MCP toolbox; screens output for injection; marks transfers | `agent` |
| `output_guard` | draft reply | checks unbacked claims, data leaks, length; one ephemeral rewrite | END (wait for customer) or `agent` |
| `escalate` | stuck state | calls `transfer_to_human_agents` with a summary; tells the customer | END |

The topology is declared once (`agent/graph.py: TOPOLOGY`) and consumed by both the LangGraph builder
and `MiniGraph`, so tests exercise exactly the production wiring.

## 2. State

`AgentState` (TypedDict, JSON-only so it checkpoints anywhere):

| Field | Reducer | Purpose |
|---|---|---|
| `messages` | append | user / assistant (with tool calls) / tool messages, incl. blocked attempts |
| `pending_calls`, `decisions` | replace | calls in flight and their guard decisions |
| `approvals` | append | audit log: tool, args, approved, reviewer, reason, time |
| `guard_events` | append | every block, flag, escalation: input for evals and traces |
| `steps_this_turn`, `blocked_retries` | replace | loop budgets |
| `draft`, `feedback`, `output_retry` | replace | output-guard rewrite cycle, never shown to the customer |
| `status` | replace | `awaiting_user`, `transferred`, `escalated` |

Guards **recompute** facts (authenticated user, owned orders, did the customer just say yes) from
`messages` every time rather than trusting flags. A resumed checkpoint or replayed trace therefore
produces identical decisions.

## 3. Tools and MCP

`ToolSpec` carries risk metadata — `kind` (read / write / transfer / generic), `irreversible`, `confirm`
(customer must say yes), `approve` (operator must sign off). The retail tools are served by an
MCP server (`MCPServer`, MCP Python SDK v2) that publishes standard `ToolAnnotations`
(`read_only_hint`, `destructive_hint`, `idempotent_hint`) plus our richer metadata in `meta`.
`MCPToolbox` loads any MCP server's tools into a registry. Our own metadata wins, then the
standard annotations, and tools with neither fall back to name-based classification that fails
closed (an unknown verb is treated as a write needing confirmation).

The graph is synchronous and the MCP client is async; `MCPToolbox` runs one client session on a
private event-loop thread and exposes a blocking `execute()`.

## 4. Human-in-the-loop mechanics

`approval` calls `interrupt(payload)`. With LangGraph this persists the state in the checkpointer and
returns `__interrupt__` to the caller; the API stores nothing else. An operator later calls
`POST /v1/approvals/{thread}`, which runs `graph.invoke(Command(resume=decisions))`. LangGraph re-runs
the `approval` node from the top (so it is written to be side-effect-free before `interrupt`), and the
`tools` node — a separate node — executes the approved calls. That separation is what makes resumption
safe: a crash or retry can never execute a refund twice before approval.

## 5. Inside tau2-bench

In tau2 the environment executes tools and an LLM plays the customer. `bench/tau2_agent.py`
implements tau2's `HalfDuplexAgent` interface with the same guard and output checks: blocked calls are
fed back to the model internally and never reach tau2's environment; allowed calls are emitted; stuck
→ `transfer_to_human_agents`. Approvals use a logged policy (auto-approve or auto-deny). Guard events go
to a sidecar file joined by the metrics loader.

## 6. Durable execution (optional Temporal)

`temporal/workflow.py` runs one workflow per conversation. Customer messages and operator decisions
are **signals**; each agent turn is an **activity** with retries. While an approval is pending the
workflow waits on a signal with a timeout (`APPROVAL_TIMEOUT_S`); on timeout it resumes the graph with a
denial, so the customer is told and offered a transfer instead of waiting forever. Workflow history is
an additional audit trail.

## 7. Deployment

One image, three roles: API (FastAPI), tools (MCP streamable HTTP, stateless), worker (Temporal).
On Cloud Run: API with Cloud SQL Postgres as the LangGraph checkpointer (so any instance can resume
any paused conversation), tools service on internal ingress, secrets from Secret Manager, traces to Arize.
