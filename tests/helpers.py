import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from supportagent.agent import Deps, SupportAgent  # noqa: E402
from supportagent.domains.retail import POLICY, RetailDB, build_tools  # noqa: E402
from supportagent.guardrails import ToolGuard, build_context  # noqa: E402
from supportagent.llm import ScriptedLLM  # noqa: E402
from supportagent.messages import Message, ToolCall  # noqa: E402
from supportagent.tracing import Tracer  # noqa: E402

EMAIL = "yusuf.rossi7301@example.com"
UID = "yusuf_rossi_9620"


def make_agent(script, db=None, approval_mode="human", **kw):
    db = db or RetailDB()
    kw.setdefault("tracer", Tracer("t", enabled=False))
    deps = Deps(ScriptedLLM(script), build_tools(db), POLICY, approval_mode=approval_mode, **kw)
    return SupportAgent(deps, backend="mini"), db


def authed_history(db=None, extra=None):
    """Conversation in which the user is authenticated and their details were fetched."""
    db = db or RetailDB()
    reg = build_tools(db)
    msgs = [Message.user(f"hi I'm {EMAIL}")]
    for name, args in [("find_user_id_by_email", {"email": EMAIL}), ("get_user_details", {"user_id": UID})]:
        c = ToolCall(name=name, arguments=args)
        content, err = reg.execute(name, args)
        msgs += [Message.assistant("", [c]), Message.tool(c, content, err)]
    return msgs + (extra or []), reg


def check(call_name, args, msgs, reg, domain="retail"):
    guard = ToolGuard(reg, domain)
    ctx = build_context(msgs, {t.name for t in reg if t.kind == "write"})
    return guard.check([ToolCall(name=call_name, arguments=args)], ctx)[0]
