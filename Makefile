.PHONY: install test demo trace bench-local tau2 tau2-baseline api tools worker docker deploy lint

install:
	pip install -r requirements.txt && pip install -e . --no-deps

test:                ## 50+ unit tests, no keys, no network
	python -m unittest discover -s tests -v

demo:                ## real graph + guards + approval + MCP, scripted model
	python scripts/demo_offline.py

trace:               ## timeline of the demo cancel conversation
	python scripts/inspect_trace.py traces/demo/demo-cancel.jsonl

bench-local:         ## 7 local tasks x 4 trials with an LLM agent + LLM user
	python scripts/run_local_bench.py --trials 4

tau2:                ## guarded agent on tau2 retail (run inside the tau2 environment)
	python scripts/run_tau2.py --domain retail --trials 4

tau2-baseline:
	python scripts/run_tau2.py --domain retail --trials 4 --baseline

api:
	uvicorn supportagent.api.main:app --reload --port 8080

tools:
	python -m supportagent.tools.mcp_server --http --port 8001

worker:
	python -m supportagent.temporal.worker

docker:
	docker compose up --build

deploy:
	bash deploy/cloudrun.sh

lint:
	ruff check src scripts tests
