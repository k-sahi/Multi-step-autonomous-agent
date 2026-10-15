# Deployment

## Components

| Service | Command | Scales on | State |
|---|---|---|---|
| API (FastAPI) | `uvicorn supportagent.api.main:app` | conversations | none in process; LangGraph checkpoints in Postgres |
| Tools (MCP) | `python -m supportagent.tools.mcp_server --http` | tool calls | the backing systems (here: demo DB) |
| Worker (Temporal, optional) | `python -m supportagent.temporal.worker` | turns | Temporal history + Postgres checkpoints |

## Local

```bash
cp .env.example .env
docker compose up --build                       # API :8080, MCP tools :8001, Postgres :5432
docker compose --profile temporal up --build    # + Temporal server and worker
```

## Cloud Run

`deploy/cloudrun.sh` builds one image and deploys:
* `support-tools` — MCP server, **internal ingress** (not reachable from the internet), stateless HTTP so
  any instance can serve any request.
* `support-agent-api` — public API with Cloud SQL Postgres attached as the LangGraph checkpointer, Direct VPC
  egress to reach the tools service, secrets from Secret Manager, `--min-instances 1` to avoid cold starts on
  chat, `--timeout 300` because one turn can include several LLM calls.

```bash
PROJECT=my-proj SQL_INSTANCE=my-proj:us-central1:agent-pg bash deploy/cloudrun.sh
```

Why Postgres: Cloud Run instances are ephemeral and scale out. A conversation paused for approval must survive
the instance that paused it; with the checkpoint in Postgres, the operator's approval can land on any instance
and resume the graph.

Temporal on Cloud Run: run the worker as a Cloud Run **worker pool** or on GKE and point it at Temporal Cloud;
workers make outbound connections only.

## Security

* Operators authenticate with `X-Operator-Token` (replace with IAP/OIDC in production) and must send
  `X-Reviewer`, which is recorded with every decision.
* The tool server is never public; the API is the only caller.
* Traces sent to Arize are PII-redacted; keys come from Secret Manager.
* The agent's tool permissions are the MCP server's tools — keep that server's surface minimal per domain.

## Operations

* Alert on: escalation rate, approval queue age, guard block rate by rule, tool error rate, p95 turn latency.
* Re-run the local benchmark in CI on every prompt or guard change; run tau2 nightly or per release.
