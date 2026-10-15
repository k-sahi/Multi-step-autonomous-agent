"""Tool specifications with risk metadata.

Every tool the agent can call is described by a ToolSpec. The risk fields are what the
guardrails and the human-approval step key off — so classifying tools correctly is
itself a safety decision, made once here and tested, not left to the model:

  kind         read      looks something up, no side effects
               write     changes state (DB, money, bookings)
               transfer  hands the conversation to a human agent
               generic   pure computation (calculator, think)
  irreversible a write that cannot be undone by another tool call (cancel, refund, charge)
  confirm      the domain policy requires the *customer's* explicit "yes" first
  approve      an *operator* must sign off (human-in-the-loop) before it executes
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

ToolKind = Literal["read", "write", "transfer", "generic"]


class ToolExecutionError(Exception):
    """Raised by tool functions for expected, user-visible failures (e.g. order not pending)."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any] | None = None
    kind: ToolKind = "read"
    irreversible: bool = False
    confirm: bool = False
    approve: bool = False
    risk_note: str = ""

    def to_anthropic(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class ToolRegistry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def add(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __iter__(self):
        return iter(self.tools.values())

    def names(self) -> list[str]:
        return list(self.tools)

    def schemas(self) -> list[dict]:
        return [t.to_anthropic() for t in self.tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool locally. Returns (content, is_error). Never raises for tool failures:
        the error string goes back to the model so it can recover, which is how tau2 works too."""
        spec = self.tools.get(name)
        if spec is None or spec.fn is None:
            return f"Error: unknown tool '{name}'", True
        try:
            out = spec.fn(**arguments)
        except ToolExecutionError as exc:
            return f"Error: {exc}", True
        except TypeError as exc:  # wrong/missing arguments
            return f"Error: invalid arguments for {name}: {exc}", True
        return (out if isinstance(out, str) else json.dumps(out, default=str)), False


# ------------------------------------------------------------------- validation
_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
               "array": list, "object": dict}


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Small JSON-schema subset validator (type, required, enum, items, additionalProperties).
    Enough for tool schemas, and dependency-free so it runs inside the guard node."""
    errors: list[str] = []
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            errors.append(f"missing required argument '{req}'")
    if schema.get("additionalProperties") is False:
        for k in args:
            if k not in props:
                errors.append(f"unexpected argument '{k}'")
    for k, v in args.items():
        spec = props.get(k)
        if not spec:
            continue
        errors += _check(k, spec, v)
    return errors


def _check(path: str, spec: dict, v: Any) -> list[str]:
    t = spec.get("type")
    if t:
        py = _JSON_TYPES.get(t)
        if py and not isinstance(v, py) or (t in ("integer", "number") and isinstance(v, bool)):
            return [f"'{path}' should be {t}, got {type(v).__name__}"]
    if "enum" in spec and v not in spec["enum"]:
        return [f"'{path}' must be one of {spec['enum']}, got {v!r}"]
    if t == "array" and "items" in spec:
        out = []
        for i, item in enumerate(v):
            out += _check(f"{path}[{i}]", spec["items"], item)
        return out
    return []


# ------------------------------------------------------------------- classification
WRITE_PREFIXES = ("cancel", "modify", "update", "return", "exchange", "book", "refund", "send",
                  "toggle", "enable", "disable", "reset", "add_", "remove", "suspend", "resume", "make_payment",
                  "set_", "change")
IRREVERSIBLE_PREFIXES = ("cancel", "return", "exchange", "refund", "book", "make_payment", "send_certificate")


def classify(name: str) -> dict[str, Any]:
    """Default risk classification by tool name, used for tools that arrive without metadata
    (e.g. tau2 airline/telecom tools, or third-party MCP servers without annotations).
    Explicit metadata always wins; this is the conservative fallback."""
    n = name.lower()
    if n.startswith("transfer_to_human"):
        return {"kind": "transfer"}
    if n in ("calculate", "think"):
        return {"kind": "generic"}
    if n.startswith(WRITE_PREFIXES):
        irrev = n.startswith(IRREVERSIBLE_PREFIXES)
        return {"kind": "write", "irreversible": irrev, "confirm": True, "approve": irrev}
    if n.startswith(("get_", "find_", "list_", "search", "check", "lookup", "read")):
        return {"kind": "read"}
    # Unknown verb: treat as a write that needs confirmation. Failing closed costs a question;
    # failing open can cost a refund.
    return {"kind": "write", "irreversible": False, "confirm": True, "approve": False}
