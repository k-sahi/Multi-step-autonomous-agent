"""Provider-neutral message and tool-call types used across the agent.

Kept deliberately small and JSON-serialisable: they live in LangGraph checkpoints,
Temporal payloads and trace files, so every field must round-trip through JSON.
Adapters convert to/from the Anthropic API format (llm.py) and tau2's types (bench/tau2_agent.py).
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id(prefix: str = "call") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ToolCall(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def signature(self) -> str:
        """Stable key for loop detection: same tool + same args."""
        import json

        return f"{self.name}:{json.dumps(self.arguments, sort_keys=True, default=str)}"


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None          # for role == "tool"
    name: str | None = None                  # tool name for role == "tool"
    is_error: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)   # guard notes, latency, token usage

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str = "", tool_calls: list[ToolCall] | None = None, **meta) -> "Message":
        return cls(role="assistant", content=text, tool_calls=tool_calls or [], meta=meta)

    @classmethod
    def tool(cls, call: ToolCall, content: str, is_error: bool = False, **meta) -> "Message":
        return cls(role="tool", content=content, tool_call_id=call.id, name=call.name, is_error=is_error, meta=meta)


def to_dicts(msgs: list[Message]) -> list[dict]:
    return [m.model_dump(mode="json") for m in msgs]


def from_dicts(rows: list[dict]) -> list[Message]:
    return [Message.model_validate(r) for r in rows]
