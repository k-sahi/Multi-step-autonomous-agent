"""LLM clients.

`AnthropicChat` calls the Messages API with native tool use. `ScriptedLLM` replays a
fixed list of responses, which lets tests and the offline demo drive the *real* graph,
guards and approval flow deterministically.
"""
from __future__ import annotations

import time
from typing import Any, Protocol

import requests

from .config import settings
from .guardrails.injection import wrap_tool_output
from .messages import Message, ToolCall

API_URL = "https://api.anthropic.com/v1/messages"


class ChatModel(Protocol):
    model: str

    def __call__(self, system: str, messages: list[Message], tools: list[dict]) -> Message: ...


def to_anthropic(messages: list[Message]) -> list[dict]:
    """Our messages -> Anthropic format. Consecutive tool results are merged into a single
    user turn (the API requires tool_result blocks to directly follow the tool_use turn)."""
    out: list[dict] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "user":
            out.append({"role": "user", "content": m.content or "(empty)"})
        elif m.role == "assistant":
            blocks: list[dict] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for c in m.tool_calls:
                blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no content)"}]})
        elif m.role == "tool":
            from .guardrails.injection import InjectionFlag

            flags = [InjectionFlag(**f) for f in m.meta.get("injection_flags", [])]
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                     "content": wrap_tool_output(m.name or "tool", m.content, flags), "is_error": m.is_error}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


class AnthropicChat:
    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 2048,
                 temperature: float = 0.0, timeout: float = 120.0):
        self.model = model or settings.agent_model
        self.api_key = api_key or settings.anthropic_api_key
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

    def __call__(self, system: str, messages: list[Message], tools: list[dict]) -> Message:
        body: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            # Cache the long, static prefix (policy + tool schemas): most of each request's tokens.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": tools, "messages": to_anthropic(messages),
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        t0, last = time.perf_counter(), ""
        for attempt in range(5):
            r = requests.post(API_URL, headers=headers, json=body, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
                calls = [ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {})
                         for b in data["content"] if b["type"] == "tool_use"]
                return Message.assistant(text.strip(), calls, model=self.model, usage=data.get("usage", {}),
                                         stop_reason=data.get("stop_reason"),
                                         latency_s=round(time.perf_counter() - t0, 3))
            last = f"{r.status_code}: {r.text[:300]}"
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(min(30, 2 ** attempt))
                continue
            break
        raise RuntimeError(f"Anthropic API error {last}")


class ScriptedLLM:
    """Replays responses in order. Each item is a Message, a str (text reply), or a
    (tool_name, args) tuple / list of tuples (tool calls)."""

    def __init__(self, script: list[Any], model: str = "scripted"):
        self.script = list(script)
        self.model = model
        self.calls: list[list[Message]] = []

    def __call__(self, system: str, messages: list[Message], tools: list[dict]) -> Message:
        self.calls.append(list(messages))
        if not self.script:
            return Message.assistant("Is there anything else I can help you with?")
        item = self.script.pop(0)
        if callable(item) and not isinstance(item, Message):
            item = item(messages)
        if isinstance(item, Message):
            return item
        if isinstance(item, str):
            return Message.assistant(item)
        if isinstance(item, tuple):
            item = [item]
        return Message.assistant("", [ToolCall(name=n, arguments=a) for n, a in item])
