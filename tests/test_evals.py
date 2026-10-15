import json
import sys
import tempfile
import types
import unittest
from math import comb
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from helpers import EMAIL, UID

from pydantic import BaseModel

from supportagent.bench.local import ScriptedUser, goal_hash, run_trial
from supportagent.bench.local_tasks import TASKS
from supportagent.evals import classify_failure, failure_breakdown, pass_at_k, pass_hat_k, summarize, tool_accuracy
from supportagent.evals.metrics import Trial, bootstrap_ci, escalation_stats, pass_k_curve
from supportagent.evals.tau2_results import load_tau2_run
from supportagent.llm import ScriptedLLM

CANCEL = {"order_id": "#W2378156", "reason": "no longer needed"}
EXP = [{"name": "get_order_details", "arguments": {"order_id": "#W2378156"}},
       {"name": "cancel_pending_order", "arguments": CANCEL}]


def T(task, trial, reward, calls=(), expected=EXP, **kw):
    return Trial(task, trial, reward, [dict(c) for c in calls], list(expected), **kw)


class TestPassK(unittest.TestCase):
    def test_formulas(self):
        self.assertEqual(pass_hat_k(4, 4, 4), 1.0)
        self.assertEqual(pass_hat_k(4, 3, 4), 0.0)
        self.assertAlmostEqual(pass_hat_k(4, 3, 2), comb(3, 2) / comb(4, 2))
        self.assertAlmostEqual(pass_at_k(4, 1, 2), 1 - comb(3, 2) / comb(4, 2))
        self.assertEqual(pass_hat_k(5, 2, 1), 0.4)               # pass^1 == success rate
        with self.assertRaises(ValueError):
            pass_hat_k(2, 1, 3)

    def test_curve_monotone_and_gap(self):
        trials = [T("a", i, r) for i, r in enumerate([1, 1, 1, 0])] + [T("b", i, 1) for i in range(4)]
        c = pass_k_curve(trials)
        hat, at = c["pass^k"], c["pass@k"]
        self.assertEqual(hat[1], 0.875)
        self.assertTrue(all(hat[k] >= hat[k + 1] for k in range(1, 4)))
        self.assertTrue(all(at[k] <= at[k + 1] for k in range(1, 4)))
        self.assertEqual((hat[4], at[4]), (0.5, 1.0))


class TestToolAccuracy(unittest.TestCase):
    def test_exact(self):
        calls = [{"name": "get_order_details", "arguments": {"order_id": "#W2378156"}},
                 {"name": "cancel_pending_order", "arguments": CANCEL}]
        a = tool_accuracy(T("a", 0, 1, calls))
        self.assertEqual((a["write_precision"], a["write_recall"], a["read_recall"]), (1.0, 1.0, 1.0))
        self.assertTrue(a["exact_write_sequence"])

    def test_wrong_args_and_extra(self):
        calls = [{"name": "cancel_pending_order", "arguments": {**CANCEL, "reason": "ordered by mistake"}},
                 {"name": "modify_user_address", "arguments": {"user_id": UID}}]
        a = tool_accuracy(T("a", 0, 0, calls))
        self.assertEqual((a["write_precision"], a["write_recall"], a["read_recall"]), (0.0, 0.0, 0.0))

    def test_compare_args_and_case(self):
        exp = [{"name": "cancel_pending_order", "arguments": CANCEL, "compare_args": ["order_id"]}]
        calls = [{"name": "cancel_pending_order", "arguments": {"order_id": "#w2378156", "reason": "x"}}]
        self.assertEqual(tool_accuracy(T("a", 0, 1, calls, exp))["write_recall"], 1.0)

    def test_blocked_and_errors(self):
        calls = [{"name": "cancel_pending_order", "arguments": CANCEL, "blocked": True},
                 {"name": "cancel_pending_order", "arguments": CANCEL, "is_error": True},
                 {"name": "cancel_pending_order", "arguments": CANCEL}]
        a = tool_accuracy(T("a", 0, 1, calls))
        self.assertAlmostEqual(a["blocked_rate"], 1 / 3)
        self.assertAlmostEqual(a["tool_error_rate"], 1 / 2)
        self.assertEqual(a["write_precision"], 1.0)    # errored write isn't counted as a change

    def test_no_writes_expected(self):
        a = tool_accuracy(T("a", 0, 1, [], []))
        self.assertEqual((a["write_precision"], a["write_recall"]), (1.0, 1.0))


class TestEscalationAndFailures(unittest.TestCase):
    def test_escalation(self):
        tr = {"name": "transfer_to_human_agents", "arguments": {}}
        trials = [T("a", 0, 1, [tr], [tr]), T("b", 0, 0, [tr]), T("c", 0, 0, [], [tr]), T("d", 0, 1)]
        e = escalation_stats(trials)
        self.assertEqual(e["escalation_rate"], 0.5)
        self.assertEqual(e["appropriate_escalation_rate"], 0.5)
        self.assertEqual((e["unnecessary_escalations"], e["missed_escalations"]), (1, 1))

    def test_taxonomy(self):
        ok = {"name": "cancel_pending_order", "arguments": CANCEL}
        bad = {"name": "cancel_pending_order", "arguments": {**CANCEL, "reason": "ordered by mistake"}}
        cases = {
            "no_action": T("a", 0, 0, []),
            "wrong_arguments": T("a", 0, 0, [bad]),
            "wrong_tool": T("a", 0, 0, [{"name": "modify_user_address", "arguments": {}}]),
            "extra_action": T("a", 0, 0, [ok, {"name": "modify_user_address", "arguments": {}}]),
            "communication": T("a", 0, 0, [ok]),
            "missed_escalation": T("a", 0, 0, [], [{"name": "transfer_to_human_agents", "arguments": {}}]),
            "guard_escalation": T("a", 0, 0, [], guard_events=[{"type": "escalation"}]),
            "approval_denied": T("a", 0, 0, [], approvals=[{"approved": False}]),
            "max_steps_or_error": T("a", 0, 0, [], termination_reason="max_steps"),
        }
        for label, trial in cases.items():
            self.assertEqual(classify_failure(trial), label, label)
        self.assertEqual(classify_failure(T("a", 0, 1)), "success")
        fb = failure_breakdown(list(cases.values()))
        self.assertEqual(fb["n_failures"], len(cases))

    def test_summary_and_ci(self):
        trials = [T(t, i, float((i + j) % 3 != 0), [{"name": "cancel_pending_order", "arguments": CANCEL}])
                  for j, t in enumerate("abcd") for i in range(3)]
        s = summarize(trials)
        self.assertEqual((s["n_tasks"], s["n_trials"], s["trials_per_task"]), (4, 12, [3]))
        self.assertIn(3, s["pass^k"])
        lo, hi = bootstrap_ci([0.0, 0.5, 1.0, 1.0])
        self.assertLessEqual(lo, hi)


class TestTau2Loader(unittest.TestCase):
    def test_load(self):
        run = {"info": {"environment_info": {"domain_name": "retail"}},
               "tasks": [{"id": "7", "evaluation_criteria": {"actions": [
                   {"action_id": "7_0", "name": "cancel_pending_order", "arguments": CANCEL}]}}],
               "simulations": [
                   {"task_id": "7", "trial": 0, "reward_info": {"reward": 1.0}, "termination_reason": "user_stop",
                    "agent_cost": 0.02, "messages": [
                        {"role": "user", "content": "cancel"},
                        {"role": "assistant", "content": None, "tool_calls": [
                            {"id": "c1", "name": "cancel_pending_order", "arguments": CANCEL}]},
                        {"role": "tool", "id": "c1", "content": "{}", "error": False}]},
                   {"task_id": "7", "trial": 1, "reward_info": {"reward": 0.0}, "messages": []}]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "run.json"
            p.write_text(json.dumps(run))
            (Path(d) / "run.guard.jsonl").write_text(json.dumps(
                {"task_id": "7", "trial": 1, "guard_events": [{"type": "tool_blocked", "rule": "schema"}],
                 "approvals": [], "blocked_calls": [{"name": "cancel_pending_order", "arguments": {}}]}) + "\n")
            trials = load_tau2_run(p)
        self.assertEqual([t.reward for t in trials], [1.0, 0.0])
        self.assertEqual(trials[0].tool_calls[0]["name"], "cancel_pending_order")
        self.assertEqual(trials[0].cost, 0.02)
        self.assertTrue(trials[1].tool_calls[0]["blocked"])
        self.assertEqual(summarize(trials)["guardrails"]["blocks_by_rule"], {"schema": 1})


class TestLocalBench(unittest.TestCase):
    def task(self, tid):
        return next(t for t in TASKS if t["id"] == tid)

    def test_goal_hashes(self):
        hashes = {t["id"]: goal_hash(t) for t in TASKS}   # every reference action executes cleanly
        self.assertEqual(hashes["refuse_cancel_delivered"], hashes["refuse_other_users_order"])
        self.assertNotEqual(hashes["cancel_pending"], hashes["refuse_cancel_delivered"])

    def test_success_and_refusal(self):
        llm = ScriptedLLM([("find_user_id_by_email", {"email": EMAIL}), ("get_user_details", {"user_id": UID}),
                           "Cancel order #W2378156 (no longer needed)?", ("cancel_pending_order", CANCEL),
                           "Order #W2378156 has been cancelled."])
        t = run_trial(self.task("cancel_pending"), llm, ScriptedUser([f"I'm {EMAIL}, cancel #W2378156", "yes"]))
        self.assertEqual(t.reward, 1.0)
        self.assertEqual(len(t.approvals), 1)
        # refusal task: agent tries anyway; the tool refuses; DB unchanged -> success by not acting
        llm = ScriptedLLM([("find_user_id_by_email", {"email": EMAIL}), ("get_user_details", {"user_id": UID}),
                           "Cancel order #W4776164?",
                           ("cancel_pending_order", {"order_id": "#W4776164", "reason": "no longer needed"}),
                           "Sorry, delivered orders can't be cancelled, but I can start a return."])
        t = run_trial(self.task("refuse_cancel_delivered"), llm,
                      ScriptedUser([f"I'm {EMAIL}, cancel #W4776164", "yes"]))
        self.assertEqual(t.reward, 1.0)

    def test_communicate_requirement(self):
        task = self.task("exchange_tshirt")
        mei = "mei.kovacs8232@example.com"
        args = {"order_id": "#W6390527", "item_ids": ["8124970213"], "new_item_ids": ["9612497925"],
                "payment_method_id": "paypal_7644869"}
        base = [("find_user_id_by_email", {"email": mei}), ("get_user_details", {"user_id": "mei_kovacs_8020"}),
                ("get_order_details", {"order_id": "#W6390527"}), ("get_product_details", {"product_id": "6086499569"}),
                "Exchange item 8124970213 on #W6390527 for 9612497925 via paypal_7644869?",
                ("exchange_delivered_order_items", args)]
        user = lambda: ScriptedUser([f"I'm {mei}, exchange my shirt on #W6390527", "yes"])  # noqa: E731
        silent = run_trial(task, ScriptedLLM(base + ["Your exchange has been requested."]), user())
        told = run_trial(task, ScriptedLLM(base + ["Exchange requested; you'll pay a $1.21 difference."]), user())
        self.assertEqual((silent.reward, told.reward), (0.0, 1.0))
        self.assertEqual(classify_failure(silent), "communication")


# ------------------------------------------------------------------ tau2 adapter against a stand-in
def _install_fake_tau2():
    """Minimal stand-in for tau2's documented agent interface, so the adapter's logic is tested
    without the real package. Mirrors: HalfDuplexAgent(tools, domain_policy), message models
    with role/content/tool_calls, ToolMessage(id, content, error), MultiToolMessage(tool_messages)."""
    S = TypeVar("S")

    class HalfDuplexAgent(Generic[S]):
        def __init__(self, tools, domain_policy):
            self.tools, self.domain_policy = tools, domain_policy

    class ToolCall(BaseModel):
        id: str = ""
        name: str
        arguments: dict
        requestor: str = "assistant"

    class _Msg(BaseModel):
        role: str
        content: Optional[str] = None
        tool_calls: Optional[list[ToolCall]] = None

    class AssistantMessage(_Msg):
        pass

    class UserMessage(_Msg):
        pass

    class SystemMessage(BaseModel):
        role: str
        content: Optional[str] = None

    class ToolMessage(BaseModel):
        id: str
        role: str = "tool"
        content: Optional[str] = None
        error: bool = False

    class MultiToolMessage(BaseModel):
        role: str = "tool"
        tool_messages: list[ToolMessage]

    mods = {n: types.ModuleType(n) for n in ["tau2", "tau2.agent", "tau2.agent.base_agent", "tau2.data_model",
                                              "tau2.data_model.message"]}
    mods["tau2.agent.base_agent"].HalfDuplexAgent = HalfDuplexAgent
    for c in (AssistantMessage, UserMessage, SystemMessage, ToolMessage, MultiToolMessage, ToolCall):
        setattr(mods["tau2.data_model.message"], c.__name__, c)
    sys.modules.update(mods)
    return mods["tau2.data_model.message"]


class FakeTool:
    def __init__(self, spec):
        self.openai_schema = {"type": "function", "function": {"name": spec.name, "description": spec.description,
                                                               "parameters": spec.input_schema}}


class TestTau2Adapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = "tau2" in sys.modules
        cls.M = _install_fake_tau2()

    @classmethod
    def tearDownClass(cls):
        if not cls.real:
            for k in [k for k in sys.modules if k == "tau2" or k.startswith("tau2.")]:
                del sys.modules[k]
            sys.modules.pop("supportagent.bench.tau2_agent", None)

    def test_guarded_loop(self):
        from helpers import RetailDB, build_tools

        from supportagent.bench.tau2_agent import GuardedReActAgent

        reg = build_tools(RetailDB())
        tools = [FakeTool(s) for s in reg]
        M = self.M
        llm = ScriptedLLM([
            ("cancel_pending_order", CANCEL),                         # blocked internally: not authed/confirmed
            ("find_user_id_by_email", {"email": EMAIL}),              # emitted to env
            ("get_user_details", {"user_id": UID}),
            "Cancel order #W2378156 (no longer needed)?",
            ("cancel_pending_order", CANCEL),                         # confirmed -> approve (auto) -> emitted
            "Order #W2378156 has been cancelled.",
        ])
        agent = GuardedReActAgent(tools, "policy", chat=llm, domain="retail", approval_mode="auto_approve")
        st = agent.get_init_state()
        out, st = agent.generate_next_message(M.UserMessage(role="user", content=f"I'm {EMAIL}; cancel #W2378156"), st)
        self.assertEqual(out.tool_calls[0].name, "find_user_id_by_email")
        self.assertEqual(st.guard_events[0]["rule"], "auth_required")
        c1 = out.tool_calls[0].id
        out, st = agent.generate_next_message(M.ToolMessage(id=c1, content=UID), st)
        self.assertEqual(out.tool_calls[0].name, "get_user_details")
        user_json = json.dumps(RetailDB().users[UID])
        out, st = agent.generate_next_message(M.MultiToolMessage(tool_messages=[
            M.ToolMessage(id=out.tool_calls[0].id, content=user_json)]), st)
        self.assertIn("Cancel order", out.content)
        out, st = agent.generate_next_message(M.UserMessage(role="user", content="yes"), st)
        self.assertEqual(out.tool_calls[0].name, "cancel_pending_order")
        self.assertTrue(st.approvals[0]["approved"])
        out, st = agent.generate_next_message(M.ToolMessage(id=out.tool_calls[0].id, content="{}"), st)
        self.assertEqual(out.content, "Order #W2378156 has been cancelled.")
        # tau2 never saw the blocked attempt
        emitted = [c.name for m in st.messages if getattr(m, "tool_calls", None) for c in m.tool_calls]
        self.assertEqual(emitted, ["find_user_id_by_email", "get_user_details", "cancel_pending_order"])
        self.assertEqual(len(st.blocked_calls), 1)

    def test_escalates_when_stuck(self):
        from helpers import RetailDB, build_tools

        from supportagent.bench.tau2_agent import GuardedReActAgent

        tools = [FakeTool(s) for s in build_tools(RetailDB())]
        llm = ScriptedLLM([("get_order_details", {"order_id": "#W2378156"})] * 10)
        agent = GuardedReActAgent(tools, "policy", chat=llm, domain="retail")
        out, st = agent.generate_next_message(self.M.UserMessage(role="user", content="order #W2378156?"),
                                              agent.get_init_state())
        self.assertEqual(out.tool_calls[0].name, "transfer_to_human_agents")
        self.assertEqual(st.guard_events[-1]["type"], "escalation")


if __name__ == "__main__":
    unittest.main()
