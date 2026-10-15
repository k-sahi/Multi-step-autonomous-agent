"""Facts the guardrails derive from the conversation so far.

Guards must not trust the model's own claims ("the user is verified"), so everything
here is recomputed from the message log: tool results that succeeded, and what the user
actually typed. Recomputing (rather than storing flags) also means a resumed checkpoint
or a replayed trace produces exactly the same guard decisions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..messages import Message

AUTH_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}

AFFIRM = re.compile(
    r"^\s*(yes|yeah|yep|yup|y|sure|ok(ay)?|confirm(ed)?|correct|go ahead|please (do|proceed)|proceed|"
    r"do it|that'?s (right|correct)|sounds good|absolutely|i confirm)\b", re.I)
NEGATE = re.compile(r"\b(no|not|don'?t|do not|wait|hold on|cancel that|stop|never mind)\b", re.I)
ID_PATTERN = re.compile(r"#?[A-Za-z]*[_#]?\w*\d{4,}\w*")


@dataclass
class GuardContext:
    authenticated_user: str | None = None
    seen_text: str = ""                                 # user text + successful tool outputs
    user_text: str = ""
    owned_orders: set[str] = field(default_factory=set)
    own_payment_methods: set[str] = field(default_factory=set)
    own_emails: set[str] = field(default_factory=set)
    other_user_emails: set[str] = field(default_factory=set)
    call_counts: dict[str, int] = field(default_factory=dict)
    last_error_for: dict[str, bool] = field(default_factory=dict)
    last_user_message: str = ""
    assistant_before_last_user: str = ""
    writes_since_last_user: int = 0
    successful_writes_this_turn: list[str] = field(default_factory=list)

    @property
    def user_confirmed(self) -> bool:
        m = self.last_user_message
        return bool(AFFIRM.search(m)) and not (NEGATE.search(m) and not AFFIRM.match(m))


def build_context(messages: list[Message], write_tools: set[str]) -> GuardContext:
    ctx = GuardContext()
    pending: dict[str, str] = {}  # tool_call_id -> tool name
    last_assistant_text = ""
    for m in messages:
        if m.role == "user":
            ctx.user_text += "\n" + m.content
            ctx.seen_text += "\n" + m.content
            ctx.last_user_message = m.content
            ctx.assistant_before_last_user = last_assistant_text
            ctx.writes_since_last_user = 0
            ctx.successful_writes_this_turn = []
        elif m.role == "assistant":
            if m.content:
                last_assistant_text = m.content
            for c in m.tool_calls:
                pending[c.id] = c.name
                sig = c.signature()
                ctx.call_counts[sig] = ctx.call_counts.get(sig, 0) + 1
        elif m.role == "tool":
            name = m.name or pending.get(m.tool_call_id or "", "")
            sig_key = name
            ctx.last_error_for[sig_key] = m.is_error
            if m.is_error or m.meta.get("blocked"):
                continue
            ctx.seen_text += "\n" + m.content
            if name in AUTH_TOOLS:
                uid = m.content.strip().strip('"')
                ctx.authenticated_user = uid
            if name == "get_user_details":
                _absorb_user(ctx, m.content)
            if name in write_tools:
                ctx.writes_since_last_user += 1
                ctx.successful_writes_this_turn.append(name)
    return ctx


def _absorb_user(ctx: GuardContext, content: str) -> None:
    try:
        u = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(u, dict):
        return
    if u.get("user_id") == ctx.authenticated_user:
        ctx.owned_orders |= set(u.get("orders", []))
        pms = u.get("payment_methods", {})
        ctx.own_payment_methods |= set(pms if isinstance(pms, dict) else [p.get("id") for p in pms])
        if u.get("email"):
            ctx.own_emails.add(u["email"].lower())
    elif u.get("email"):
        ctx.other_user_emails.add(u["email"].lower())
