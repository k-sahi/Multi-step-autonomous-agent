"""Pre-execution guardrails for tool calls.

Every proposed tool call gets exactly one decision:

  ALLOW     execute it
  BLOCK     don't execute; send the reason back to the model as a tool error so it can
            correct course (ask for confirmation, authenticate first, fix arguments, ...)
  APPROVE   valid and policy-compliant, but irreversible: pause for an operator's sign-off
  ESCALATE  the agent is stuck (too many blocks / loops): hand off to a human

Rules run cheapest-first and are deterministic code, not LLM calls. That makes them fast,
free, testable, and impossible to talk out of — the model can be persuaded by a user, the
guard cannot. An LLM-based check is only used where semantics are unavoidable (see
injection.py), and even then it can only make a decision *stricter*.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from ..config import settings
from ..messages import ToolCall
from ..tools.registry import ToolRegistry, validate_arguments
from .context import AUTH_TOOLS, GuardContext

Action = Literal["allow", "block", "approve", "escalate"]

# Tools that need no authenticated user
PUBLIC_TOOLS = AUTH_TOOLS | {"list_all_product_types", "get_product_details", "calculate", "think",
                             "transfer_to_human_agents"}


@dataclass
class Decision:
    call: ToolCall
    action: Action
    rule: str = "ok"
    reason: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"tool": self.call.name, "call_id": self.call.id, "action": self.action, "rule": self.rule,
                "reason": self.reason, "arguments": self.call.arguments}


Rule = Callable[[ToolCall, GuardContext, ToolRegistry], "Decision | None"]


# ----------------------------------------------------------------- generic rules
def r_known_tool(call, ctx, reg):
    if call.name not in reg:
        return Decision(call, "block", "unknown_tool", f"There is no tool named '{call.name}'. "
                        f"Available tools: {', '.join(reg.names())}.")


def r_schema(call, ctx, reg):
    errs = validate_arguments(reg.get(call.name).input_schema, call.arguments)
    if errs:
        return Decision(call, "block", "schema", "Invalid arguments: " + "; ".join(errs))


def r_loop(call, ctx, reg):
    n = ctx.call_counts.get(call.signature(), 0)
    if n > settings.max_identical_calls:
        return Decision(call, "escalate", "loop",
                        f"'{call.name}' was already called {n} times with identical arguments.")


def r_authenticated(call, ctx, reg):
    if call.name not in PUBLIC_TOOLS and ctx.authenticated_user is None:
        return Decision(call, "block", "auth_required",
                        "The user is not authenticated. Ask for their email, or name and zip code, "
                        "and look them up first.")


def r_same_user(call, ctx, reg):
    uid = call.arguments.get("user_id")
    if uid and ctx.authenticated_user and uid != ctx.authenticated_user:
        return Decision(call, "block", "other_user",
                        "You may only act on the authenticated user's account.")


def r_grounded_ids(call, ctx, reg):
    """Every id-like argument must have appeared in the user's messages or a tool result.
    Catches the most common agent hallucination: a plausible-looking order or item id."""
    for key, val in call.arguments.items():
        vals = val if isinstance(val, list) else [val]
        for v in vals:
            if not isinstance(v, str) or not _looks_like_id(key, v):
                continue
            if v not in ctx.seen_text:
                return Decision(call, "block", "ungrounded_id",
                                f"Argument {key}={v!r} does not appear in the conversation or any tool "
                                f"result. Look it up instead of guessing.")


def r_confirmation(call, ctx, reg):
    spec = reg.get(call.name)
    if not spec.confirm:
        return None
    if not ctx.user_confirmed:
        return Decision(call, "block", "needs_user_confirmation",
                        "This action changes the account. First describe the exact action and its details "
                        "to the user and ask them to explicitly confirm (yes) before calling the tool.")
    # the confirmation must be about *this* action: the agent's message before the "yes"
    # must mention the object being changed
    target = (call.arguments.get("order_id") or call.arguments.get("reservation_id")
              or call.arguments.get("user_id"))
    if target and target not in ctx.assistant_before_last_user and \
            target.lstrip("#") not in ctx.assistant_before_last_user:
        return Decision(call, "block", "confirmation_mismatch",
                        f"The user's confirmation was not for an action on {target}. Describe this exact "
                        "action and ask again.")
    if ctx.writes_since_last_user >= 1:
        return Decision(call, "block", "one_write_per_confirmation",
                        "A change was already made after the user's last confirmation. Ask the user to "
                        "confirm this additional change separately.")


def r_approval(call, ctx, reg):
    spec = reg.get(call.name)
    if spec.approve:
        return Decision(call, "approve", "irreversible", spec.risk_note or "Irreversible action.")


# ----------------------------------------------------------------- domain rules (retail)
def r_retail_ownership(call, ctx, reg):
    oid = call.arguments.get("order_id")
    if oid and ctx.owned_orders and oid not in ctx.owned_orders:
        return Decision(call, "block", "order_not_owned",
                        f"Order {oid} does not belong to the authenticated user.")
    pm = call.arguments.get("payment_method_id")
    if pm and ctx.own_payment_methods and pm not in ctx.own_payment_methods:
        return Decision(call, "block", "payment_not_owned",
                        f"Payment method {pm} does not belong to the authenticated user.")


_ID_KEYS = re.compile(r"(^|_)(id|ids)$")


def _looks_like_id(key: str, v: str) -> bool:
    return bool(_ID_KEYS.search(key)) and any(ch.isdigit() for ch in v)


GENERIC_RULES: list[Rule] = [r_known_tool, r_schema, r_loop, r_grounded_ids]
WRITE_RULES: list[Rule] = [r_confirmation]
# Identity rules are domain-specific: retail authenticates via lookup tools; tau2 airline and
# telecom identify the customer differently, so only the generic rules apply there by default.
DOMAIN_RULES: dict[str, list[Rule]] = {"retail": [r_authenticated, r_same_user, r_retail_ownership],
                                       "airline": [], "telecom": []}


class ToolGuard:
    def __init__(self, registry: ToolRegistry, domain: str = "retail", extra_rules: list[Rule] | None = None):
        self.reg = registry
        self.domain_rules = DOMAIN_RULES.get(domain, []) + (extra_rules or [])

    def check(self, calls: list[ToolCall], ctx: GuardContext) -> list[Decision]:
        decisions: list[Decision] = []
        writes_in_batch = 0
        for call in calls:
            d = self._check_one(call, ctx)
            spec = self.reg.get(call.name)
            if d.action in ("allow", "approve") and spec and spec.kind == "write":
                writes_in_batch += 1
                if writes_in_batch > 1:
                    d = Decision(call, "block", "one_write_at_a_time",
                                 "Only one account-changing action at a time. Do this one after the "
                                 "previous one completes and the user confirms it.")
            decisions.append(d)
        return decisions

    def _check_one(self, call: ToolCall, ctx: GuardContext) -> Decision:
        for rule in GENERIC_RULES:
            d = rule(call, ctx, self.reg)
            if d:
                return d
        for rule in self.domain_rules:
            d = rule(call, ctx, self.reg)
            if d:
                return d
        spec = self.reg.get(call.name)
        if spec.kind == "write":
            for rule in WRITE_RULES:
                d = rule(call, ctx, self.reg)
                if d:
                    return d
        return r_approval(call, ctx, self.reg) or Decision(call, "allow")
