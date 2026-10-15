"""Guardrails on what the agent *says* to the user.

1. Claim consistency: the agent must not tell the user an action happened ("I've cancelled
   your order") unless a write tool actually succeeded in this turn. This is the most
   damaging failure in support: the customer leaves believing something that isn't true.
2. Data leakage: never reveal another customer's email/ids seen in tool output.
3. Length/format sanity for the channel.

Violations are sent back to the model as feedback for one rewrite; persistent violations
escalate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .context import GuardContext

CLAIM = re.compile(
    r"\b(i('ve| have)|we('ve| have)|has been|have been|is now|was|successfully)\b[^.!?]{0,40}?"
    r"\b(?P<verb>cancel+ed|refunded|returned|exchanged|modified|updated|changed|processed|booked|transferred)\b", re.I)
FUTURE_OR_QUESTION = re.compile(r"\b(will|would|can|could|shall|should|want|like me to|if you)\b[^.!?]{0,30}$", re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


@dataclass
class OutputViolation:
    rule: str
    reason: str

    def to_dict(self) -> dict:
        return {"rule": self.rule, "reason": self.reason}


def check_output(text: str, ctx: GuardContext, transferred: bool = False) -> list[OutputViolation]:
    out: list[OutputViolation] = []
    for m in CLAIM.finditer(text):
        sentence_start = max(text.rfind(".", 0, m.start()), text.rfind("?", 0, m.start()), 0)
        prefix = text[sentence_start:m.start()]
        if FUTURE_OR_QUESTION.search(prefix):
            continue
        verb = m.group("verb").lower()
        if verb.startswith("transferred") and transferred:
            continue
        if not ctx.successful_writes_this_turn:
            out.append(OutputViolation(
                "unbacked_claim",
                f"You told the user something was {verb}, but no change succeeded in this turn. "
                "Either perform the action (after confirmation) or correct the message."))
            break
    for email in EMAIL.findall(text):
        if email.lower() in ctx.other_user_emails and email.lower() not in ctx.own_emails:
            out.append(OutputViolation("data_leak", "The message contains another customer's email address."))
    if len(text) > 4000:
        out.append(OutputViolation("too_long", "The reply is too long; keep it under 4000 characters."))
    return out
