"""Settings from environment variables (and an optional .env file).

Every key is optional at import time so the offline parts (domain, guardrails,
local graph runtime, metrics) run and test without credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


ApprovalMode = Literal["human", "auto_approve", "auto_deny"]


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    agent_model: str = field(default_factory=lambda: _env("AGENT_MODEL", "claude-sonnet-5-5"))
    guard_model: str = field(default_factory=lambda: _env("GUARD_MODEL", "claude-haiku-5-5"))

    # Loop limits: the cheapest guardrail against runaway agents.
    max_steps_per_turn: int = int(_env("MAX_STEPS_PER_TURN", "12"))
    max_identical_calls: int = int(_env("MAX_IDENTICAL_CALLS", "2"))
    max_blocked_retries: int = int(_env("MAX_BLOCKED_RETRIES", "3"))

    # Who signs off on irreversible actions. "human" pauses the graph (interrupt);
    # the auto modes are for benchmarks, where no operator is present.
    approval_mode: ApprovalMode = field(default_factory=lambda: _env("APPROVAL_MODE", "human"))  # type: ignore
    approval_timeout_s: int = int(_env("APPROVAL_TIMEOUT_S", "900"))

    # Persistence for LangGraph checkpoints. sqlite path for dev; postgres URL in prod.
    checkpoint_url: str = field(default_factory=lambda: _env("CHECKPOINT_URL", "sqlite:///checkpoints.db"))

    # Observability
    arize_space_id: str | None = field(default_factory=lambda: _env("ARIZE_SPACE_ID"))
    arize_api_key: str | None = field(default_factory=lambda: _env("ARIZE_API_KEY"))
    arize_project: str = field(default_factory=lambda: _env("ARIZE_PROJECT", "support-agent"))
    trace_dir: Path = field(default_factory=lambda: Path(_env("TRACE_DIR", "traces")))

    # Temporal (optional durable layer)
    temporal_address: str = field(default_factory=lambda: _env("TEMPORAL_ADDRESS", "localhost:7233"))
    temporal_namespace: str = field(default_factory=lambda: _env("TEMPORAL_NAMESPACE", "default"))
    temporal_task_queue: str = field(default_factory=lambda: _env("TEMPORAL_TASK_QUEUE", "support-agent"))


settings = Settings()
