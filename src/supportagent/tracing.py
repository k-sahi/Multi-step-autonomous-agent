"""Tracing: local JSONL (always) + Arize via OpenTelemetry/OpenInference (when configured).

Every agent turn produces spans: llm, guard, tool.<name>, plus events for approvals and
escalations. Locally they go to traces/<thread_id>.jsonl, which `scripts/inspect_trace.py`
renders as a step-by-step timeline — the main debugging tool (docs/debugging.md).
In Arize the same spans carry OpenInference attributes (span kind, model, tool name,
inputs/outputs), so failures can be filtered by guard rule, tool, or eval outcome.

PII is redacted before anything leaves the process.
"""
from __future__ import annotations

import contextlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from .config import settings
from .guardrails.injection import redact

_otel_tracer = None
_otel_init = False


def _otel():
    global _otel_tracer, _otel_init
    if _otel_init:
        return _otel_tracer
    _otel_init = True
    if settings.arize_space_id and settings.arize_api_key:
        try:
            from arize.otel import register
            from opentelemetry import trace

            provider = register(space_id=settings.arize_space_id, api_key=settings.arize_api_key,
                                project_name=settings.arize_project)
            _otel_tracer = trace.get_tracer("supportagent", tracer_provider=provider)
        except Exception:  # pragma: no cover - optional dependency
            _otel_tracer = None
    return _otel_tracer


def _clean(v: Any) -> Any:
    if isinstance(v, str):
        return redact(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


class _Span:
    def __init__(self, rec: dict, otel_span=None):
        self.rec, self.otel = rec, otel_span

    def set(self, key: str, value: Any) -> None:
        self.rec["attrs"][key] = value
        if self.otel is not None:
            v = _clean(value)
            self.otel.set_attribute(key, v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str))


class Tracer:
    def __init__(self, thread_id: str | None = None, trace_dir: Path | None = None, enabled: bool = True):
        self.thread_id = thread_id or uuid.uuid4().hex[:12]
        self.dir = Path(trace_dir or settings.trace_dir)
        self.enabled = enabled
        self.records: list[dict] = []

    @property
    def path(self) -> Path:
        return self.dir / f"{self.thread_id}.jsonl"

    def _write(self, rec: dict) -> None:
        self.records.append(rec)
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(_clean(rec), default=str) + "\n")

    @contextlib.contextmanager
    def span(self, name: str, kind: str = "CHAIN", **attrs: Any) -> Iterator[_Span]:
        rec = {"type": "span", "name": name, "kind": kind, "thread_id": self.thread_id,
               "start": time.time(), "attrs": dict(attrs)}
        ot = _otel()
        cm = ot.start_as_current_span(name) if ot else contextlib.nullcontext()
        with cm as otel_span:
            s = _Span(rec, otel_span)
            if otel_span is not None:
                otel_span.set_attribute("openinference.span.kind", kind)
                otel_span.set_attribute("session.id", self.thread_id)
                if kind == "LLM" and "model" in attrs:
                    otel_span.set_attribute("llm.model_name", attrs["model"])
                if kind == "TOOL":
                    otel_span.set_attribute("tool.name", name.removeprefix("tool."))
                for k, v in attrs.items():
                    s.set(k, v)
            try:
                yield s
            except Exception as exc:
                rec["error"] = repr(exc)
                raise
            finally:
                rec["end"] = time.time()
                rec["duration_s"] = round(rec["end"] - rec["start"], 4)
                self._write(rec)

    def event(self, name: str, **attrs: Any) -> None:
        self._write({"type": "event", "name": name, "thread_id": self.thread_id, "ts": time.time(),
                     "attrs": attrs})


def load_trace(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
