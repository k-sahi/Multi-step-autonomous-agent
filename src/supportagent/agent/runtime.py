"""MiniGraph: a minimal local executor with LangGraph-compatible semantics.

Why it exists: the test suite and the offline demo must exercise the *real* nodes,
routing and human-approval flow with no network and no extra dependencies. MiniGraph
implements just the LangGraph behaviour the agent relies on:

  * append reducers for `messages` / `approvals` / `guard_events`
  * per-thread checkpoints (in memory)
  * `interrupt(payload)` pauses the run; `invoke(Command(resume=value), config)` re-executes
    the interrupted node from the top, where `interrupt()` now returns `value`
  * the result of an interrupted run contains `__interrupt__`

It is not a replacement for LangGraph in production (no durable checkpointer, streaming or
parallel branches). `SupportAgent` picks LangGraph when it is installed.
"""
from __future__ import annotations

import contextvars
import copy
from dataclasses import dataclass
from typing import Any

from .graph import END, ENTRY, TOPOLOGY
from .nodes import Deps, make_nodes
from .state import merge

_UNSET = object()
_resume_value: contextvars.ContextVar[Any] = contextvars.ContextVar("resume", default=_UNSET)


class GraphInterrupt(Exception):
    def __init__(self, value: Any):
        super().__init__("interrupt")
        self.value = value


@dataclass
class Interrupt:
    value: Any
    id: str = "interrupt-0"


@dataclass
class Command:
    resume: Any = None


def mini_interrupt(value: Any) -> Any:
    v = _resume_value.get()
    if v is _UNSET:
        raise GraphInterrupt(value)
    _resume_value.set(_UNSET)
    return v


class MiniGraph:
    def __init__(self, deps: Deps, recursion_limit: int = 60):
        deps.interrupt_fn = mini_interrupt
        self.nodes = make_nodes(deps)
        self.recursion_limit = recursion_limit
        self.checkpoints: dict[str, dict] = {}
        self.history: dict[str, list[str]] = {}        # node visit order per thread, for tests/debugging

    def get_state(self, config: dict) -> dict:
        return copy.deepcopy(self.checkpoints.get(_tid(config), {}).get("state", {}))

    def invoke(self, inp: Any, config: dict) -> dict:
        tid = _tid(config)
        cp = self.checkpoints.get(tid)
        if isinstance(inp, Command):
            if not cp or not cp.get("next"):
                raise RuntimeError(f"thread {tid} has no pending interrupt")
            state, node, token = cp["state"], cp["next"], _resume_value.set(inp.resume)
        else:
            state = merge(cp["state"] if cp else {}, inp)
            node, token = ENTRY, None
        trail = self.history.setdefault(tid, [])
        try:
            for _ in range(self.recursion_limit):
                if node == END:
                    break
                trail.append(node)
                try:
                    update = self.nodes[node](state)
                except GraphInterrupt as gi:
                    self.checkpoints[tid] = {"state": state, "next": node, "interrupt": gi.value}
                    return {**copy.deepcopy(state), "__interrupt__": [Interrupt(gi.value)]}
                state = merge(state, update)
                node = self._next(node, state)
            else:
                raise RuntimeError(f"recursion limit {self.recursion_limit} reached")
        finally:
            if token is not None:
                _resume_value.reset(token)
        self.checkpoints[tid] = {"state": state, "next": None, "interrupt": None}
        return copy.deepcopy(state)

    def _next(self, node: str, state: dict) -> str:
        nxt = TOPOLOGY[node]
        if isinstance(nxt, str):
            return nxt
        router, mapping = nxt
        return mapping[self.nodes[router](state)]


def _tid(config: dict) -> str:
    return config["configurable"]["thread_id"]
