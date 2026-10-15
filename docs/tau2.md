# Running on tau2-bench

## Setup

```bash
git clone https://github.com/sierra-research/tau2-bench && cd tau2-bench
uv sync                                    # Python >= 3.12; installs the tau2 CLI and domains
source .venv/bin/activate
pip install -e /path/to/support-agent      # our package into tau2's environment
cp .env.example .env                       # tau2 uses LiteLLM: set ANTHROPIC_API_KEY etc.
```

## Run

```bash
cd tau2-bench
python /path/to/support-agent/scripts/run_tau2.py --domain retail --trials 4 --tasks 20
python /path/to/support-agent/scripts/run_tau2.py --domain retail --trials 4 --tasks 20 --baseline
tau2 view                                   # browse trajectories
```

`run_tau2.py` registers our factory (`registry.register_agent_factory(create_guarded_agent, "guarded_react")`)
in-process, then calls tau2's CLI with `--agent guarded_react`. The newest file in `data/simulations/` is
parsed together with the guard sidecar.

## Integration contract (verify against your tau2 version)

| tau2 piece | What we use |
|---|---|
| `tau2.agent.base_agent.HalfDuplexAgent` | base class; `get_init_state(message_history)`, `generate_next_message(message, state)` |
| `tau2.data_model.message` | `AssistantMessage`, `UserMessage`, `ToolMessage(id, content, error)`, `MultiToolMessage`, `ToolCall(id, name, arguments)` |
| `Tool.openai_schema["function"]` | name, description, JSON-schema parameters |
| `tau2.registry.registry.register_agent_factory` | factory `(tools, domain_policy, **kwargs)` |
| results JSON | `tasks[].evaluation_criteria.actions`, `simulations[].reward_info.reward`, `messages`, `termination_reason`, `agent_cost` |

tau2's internals change between releases (tau2 → τ³ refactored some APIs). The adapter is tested against a
stand-in implementing the contract above (`tests/test_evals.py::TestTau2Adapter`); if a real run fails on import or
a field name, adjust `bench/tau2_agent.py` and `evals/tau2_results.py` only.

## Notes

* The **sidecar** (`*.guard.jsonl`) holds guard events, blocked calls and approvals per `(task_id, trial)`.
  Trial indices are assigned in creation order per task; with `--max-concurrency > 1` verify the join.
* Approvals in benchmarks: `--approval auto_approve` (default) or `auto_deny`.
* Use the same `--user-llm` for every run you compare. User-simulator errors exist; spot-check failures.
* Cost: trials × tasks × (agent + user tokens). Iterate with `--tasks 20` or the local harness; run full
  domains for reported numbers.
