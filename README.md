# Guarded Multi-Step Support Agent

A ReAct-style customer-support agent that reasons, calls a tool, reads the result and decides
again over many steps — with **deterministic guardrails** on every tool call and reply,
**human sign-off before irreversible actions**, full **tracing**, and an evaluation suite on
**tau2-bench** (retail, airline, telecom).

```
customer ─► LangGraph agent ──────────────────────────────────────────────────────────────┐
              │  agent (LLM proposes)                                                      │
              │    ├─ tool calls ─► guard ─┬─ allow ───────────────► tools (MCP) ─► agent  │
              │    │                       ├─ irreversible ─► approval (interrupt) ─► tools│
              │    │                       ├─ block ─► reason back to the model ─► agent   │
              │    │                       └─ stuck ─► escalate ─► transfer to human       │
              │    └─ text ─► output guard ─┬─ ok ─► reply to customer                     │
              │                             └─ unbacked claim / leak ─► one rewrite        │
              │  state checkpointed (SQLite / Postgres)    traces ─► JSONL + Arize         │
operator ◄── approval queue (FastAPI) ◄── interrupt payload ◄─────────────────────────────┘
              optional: Temporal workflow per conversation (approval timeouts, durable waits)
```

## What it demonstrates

| Question | Answer in one line | Where |
|---|---|---|
| Where does the human sit? | Two places: the **customer** must confirm every account change (policy), and an **operator** signs off every irreversible action (graph `interrupt`); timeouts fail safe (deny + escalate). | [docs/design_decisions.md §1](docs/design_decisions.md#1-where-the-human-sits) |
| What evals did you build? | Task success (pass^1), **pass^k** reliability curve, tool accuracy (write P/R/F1, read recall, error and block rates), escalation rate (appropriate / unnecessary / missed), guardrail stats, a failure taxonomy, bootstrap CIs. | [docs/evaluation.md](docs/evaluation.md) |
| What guardrails? | 12 deterministic tool-call rules (schema, auth, ownership, grounded ids, confirmation, one write at a time, loop, budget…), output checks (unbacked claims, data leaks), injection screening, PII-redacted traces. | [docs/guardrails.md](docs/guardrails.md) |
| How did you debug traces? | Every step is a span (llm / guard / tool / approval); `inspect_trace.py` renders a timeline with hints; failure taxonomy points to the trace to open. | [docs/debugging.md](docs/debugging.md) |

## Quick start

```bash
# No keys needed: 52 tests + a demo of the real graph, guards, approvals and MCP tools
python -m unittest discover -s tests
python scripts/demo_offline.py
python scripts/inspect_trace.py traces/demo/demo-cancel.jsonl

# With keys
cp .env.example .env            # ANTHROPIC_API_KEY, OPERATOR_TOKEN, ...
pip install -r requirements.txt && pip install -e . --no-deps
python scripts/run_local_bench.py --trials 4          # 7 local tasks, LLM user simulator
uvicorn supportagent.api.main:app --port 8080          # chat + operator approval API

# tau2-bench (inside a tau2-bench checkout's environment)
python scripts/run_tau2.py --domain retail --trials 4 --tasks 20
python scripts/run_tau2.py --domain retail --trials 4 --tasks 20 --baseline   # tau2's stock agent
python scripts/compute_metrics.py <guarded>.json --compare <baseline>.json
```

## Repository map

```
src/supportagent/
  agent/       nodes.py (all node logic), graph.py (topology + LangGraph builder), runtime.py
               (MiniGraph: same nodes, LangGraph-compatible interrupt semantics, for tests),
               state.py, service.py (send / pending_approval / resolve)
  guardrails/  policy.py (tool-call rules), context.py (facts derived from the log), output.py,
               injection.py (screening, tool-output wrapping, PII redaction)
  tools/       registry.py (ToolSpec risk metadata, schema validation, classification),
               mcp_server.py (MCP SDK v2 MCPServer with annotations), mcp_client.py (MCPToolbox)
  domains/retail/  offline retail domain in tau2's shape: data, tools, policy
  bench/       tau2_agent.py (guarded agent for tau2), local.py + local_tasks.py (local harness)
  evals/       metrics.py (success, pass^k, tool accuracy, escalation, CIs), failures.py
               (taxonomy), tau2_results.py (loader)
  api/         FastAPI chat + operator approval queue
  temporal/    optional durable workflow (signals, approval timeout) + worker
  tracing.py   JSONL + Arize (OpenInference) spans, PII-redacted
scripts/       demo_offline, inspect_trace, run_local_bench, run_tau2, compute_metrics
tests/         52 tests (+1 API test that runs when FastAPI is installed)
docs/          architecture, design decisions, guardrails, evaluation, debugging, tau2, API,
               deployment, Question_Bank.pdf (103 interview Q&As; regenerate with
               python docs/question_bank/build_pdf.py)
Dockerfile, docker-compose.yml (API + MCP tools + Postgres + Temporal), deploy/cloudrun.sh
```

## Results

Left blank on purpose until runs are done; fill from `outputs/tau2/*.metrics.json`.

| tau2 domain | Agent | Tasks × trials | Success (pass^1) | pass^2 | pass^4 | Write F1 | Escalation (unnecessary) |
|---|---|---|---|---|---|---|---|
| retail | tau2 llm_agent (baseline) | | | | | | |
| retail | guarded_react | | | | | | |
| airline | guarded_react | | | | | | |

## Status and honest limits

* **Tested here:** guardrails, graph routing, interrupt/resume, approval allow/deny, escalation, output
  rewrites, MCP server and client (real MCP SDK 2.2, in-process), retail tools, metrics, failure taxonomy,
  tau2 results loader, local benchmark, and the tau2 adapter against a stand-in of tau2's documented
  agent interface.
* **Not run here** (no network or keys in the build environment): LangGraph itself (the same nodes run
  under `MiniGraph`), Postgres/SQLite checkpointers, Temporal, FastAPI, Arize, real LLM calls, and tau2-bench.
  These follow each project's current docs; expect small fixes on first run, especially in
  `bench/tau2_agent.py` because tau2's internal API changes between releases.
* tau2 runs use an **auto-approve** policy in place of the operator (nobody is there to click). Every
  auto-approval is logged, and `--approval auto_deny` measures how the agent behaves when sign-off is refused.
