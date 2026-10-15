import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from helpers import EMAIL, UID, Message, RetailDB, ToolCall, build_tools, make_agent

from supportagent.config import settings
from supportagent.tools.mcp_client import MCPToolbox, spec_from_mcp
from supportagent.tools.mcp_server import build_server
from supportagent.tracing import Tracer, load_trace

CANCEL = {"order_id": "#W2378156", "reason": "no longer needed"}
AUTH = [("find_user_id_by_email", {"email": EMAIL}), ("get_user_details", {"user_id": UID})]


class TestRetailTools(unittest.TestCase):
    def setUp(self):
        self.db = RetailDB()
        self.reg = build_tools(self.db)

    def test_cancel_rules(self):
        out, err = self.reg.execute("cancel_pending_order", {"order_id": "#W4776164", "reason": "no longer needed"})
        self.assertTrue(err)
        self.assertIn("non-pending", out)
        out, err = self.reg.execute("cancel_pending_order", {"order_id": "#W2378156", "reason": "too expensive"})
        self.assertTrue(err)
        out, err = self.reg.execute("cancel_pending_order", CANCEL)
        self.assertFalse(err)
        self.assertEqual(self.db.orders["#W2378156"]["status"], "cancelled")
        self.assertEqual(self.db.orders["#W2378156"]["payment_history"][-1]["transaction_type"], "refund")

    def test_return_payment_rule(self):
        out, err = self.reg.execute("return_delivered_order_items", {
            "order_id": "#W6390527", "item_ids": ["8124970213"], "payment_method_id": "credit_card_9513926"})
        self.assertTrue(err)   # not this user's payment method
        out, err = self.reg.execute("return_delivered_order_items", {
            "order_id": "#W4776164", "item_ids": ["3453331371"], "payment_method_id": "gift_card_6118653"})
        self.assertFalse(err)

    def test_exchange_and_modify_once(self):
        out, err = self.reg.execute("exchange_delivered_order_items", {
            "order_id": "#W6390527", "item_ids": ["8124970213"], "new_item_ids": ["9612497925"],
            "payment_method_id": "paypal_7644869"})
        self.assertFalse(err)
        self.assertEqual(json.loads(out)["exchange_price_difference"], 1.21)
        out, err = self.reg.execute("exchange_delivered_order_items", {
            "order_id": "#W6390527", "item_ids": ["8124970213"], "new_item_ids": ["1176194968"],
            "payment_method_id": "paypal_7644869"})
        self.assertTrue(err)   # order no longer "delivered"
        args = {"order_id": "#W2378156", "item_ids": ["9612497925"], "new_item_ids": ["1176194968"],
                "payment_method_id": "credit_card_9513926"}
        out, err = self.reg.execute("modify_pending_order_items", args)
        self.assertIn("not available", out)
        args["new_item_ids"] = ["8124970213"]
        self.assertFalse(self.reg.execute("modify_pending_order_items", args)[1])
        self.assertTrue(self.reg.execute("modify_pending_order_items", args)[1])   # only once

    def test_db_hash_changes(self):
        h = self.db.hash()
        self.reg.execute("cancel_pending_order", CANCEL)
        self.assertNotEqual(h, self.db.hash())
        self.assertEqual(RetailDB().hash(), h)


class TestGraph(unittest.TestCase):
    def test_full_flow_with_approval(self):
        agent, db = make_agent(AUTH + [
            "Cancel order #W2378156 (no longer needed)?", ("cancel_pending_order", CANCEL),
            "Order #W2378156 has been cancelled."])
        r = agent.send("t", f"I'm {EMAIL}, cancel #W2378156")
        self.assertEqual(r.status, "awaiting_user")
        r = agent.send("t", "yes")
        self.assertEqual(r.status, "awaiting_approval")
        self.assertEqual(db.orders["#W2378156"]["status"], "pending")   # nothing executed while paused
        req = agent.pending_approval("t")
        self.assertEqual(req["actions"][0]["tool"], "cancel_pending_order")
        with self.assertRaises(RuntimeError):
            agent.graph.invoke(agent._Command(resume={}), {"configurable": {"thread_id": "nope"}})
        r = agent.resolve("t", {req["actions"][0]["call_id"]: {"approved": True, "reviewer": "alice"}})
        self.assertEqual(r.reply, "Order #W2378156 has been cancelled.")
        self.assertEqual(db.orders["#W2378156"]["status"], "cancelled")
        state = agent._state("t")
        self.assertEqual(state["approvals"][0]["reviewer"], "alice")
        # the approval node ran twice: once to interrupt, once on resume (LangGraph semantics)
        self.assertEqual(agent.graph.history["t"].count("approval"), 2)
        self.assertIsNone(agent.pending_approval("t"))

    def test_denied(self):
        agent, db = make_agent(AUTH + ["Cancel order #W2378156?", ("cancel_pending_order", CANCEL),
                                       "Sorry, I couldn't cancel it. Want a human agent?"])
        agent.send("t", f"I'm {EMAIL}, cancel #W2378156")
        r = agent.send("t", "yes")
        cid = r.approval_request["actions"][0]["call_id"]
        r = agent.resolve("t", {cid: {"approved": False, "reviewer": "bob", "reason": "fraud check"}})
        self.assertEqual(db.orders["#W2378156"]["status"], "pending")
        tool_msgs = [m for m in agent.messages("t") if m.role == "tool"]
        self.assertIn("supervisor declined", tool_msgs[-1].content)
        self.assertIn("fraud check", tool_msgs[-1].content)

    def test_auto_modes(self):
        script = AUTH + ["Cancel order #W2378156?", ("cancel_pending_order", CANCEL), "Done, it has been cancelled."]
        agent, db = make_agent(list(script), approval_mode="auto_approve")
        agent.send("t", f"I'm {EMAIL}")
        self.assertEqual(agent.send("t", "yes").status, "awaiting_user")
        self.assertEqual(db.orders["#W2378156"]["status"], "cancelled")
        agent, db = make_agent(list(script), approval_mode="auto_deny")
        agent.send("t", f"I'm {EMAIL}")
        agent.send("t", "yes")
        self.assertEqual(db.orders["#W2378156"]["status"], "pending")

    def test_blocked_retries_escalate(self):
        bad = [("get_order_details", {"order_id": f"#W00000{i}0"}) for i in range(6)]
        agent, db = make_agent(AUTH + bad)
        r = agent.send("t", f"I'm {EMAIL}, where's my order")
        self.assertEqual(r.status, "escalated")
        names = [c.name for m in agent.messages("t") for c in m.tool_calls]
        self.assertIn("transfer_to_human_agents", names)
        self.assertIn("transferring you to a human", r.reply)
        self.assertEqual(agent.send("t", "hello?").status, "escalated")   # conversation stays with the human

    def test_step_budget(self):
        old = settings.max_steps_per_turn
        object.__setattr__(settings, "max_steps_per_turn", 3)
        try:
            agent, _ = make_agent([("list_all_product_types", {})] * 10)
            r = agent.send("t", "what do you sell?")
            self.assertEqual(r.status, "escalated")
            self.assertIn("step budget", agent._state("t")["escalation_reason"])
        finally:
            object.__setattr__(settings, "max_steps_per_turn", old)

    def test_output_rewrite_is_ephemeral(self):
        agent, _ = make_agent(["I have refunded your order.", "Could you share your email so I can look it up?"])
        r = agent.send("t", "refund me")
        self.assertEqual(r.reply, "Could you share your email so I can look it up?")
        contents = [m.content for m in agent.messages("t")]
        self.assertNotIn("I have refunded your order.", contents)          # rejected draft never stored
        self.assertTrue(any("guardrail feedback" in m.content for m in agent.deps.llm.calls[-1]))
        self.assertTrue(any(e["type"] == "output_blocked" for e in agent._state("t")["guard_events"]))

    def test_transfer_status(self):
        agent, _ = make_agent(AUTH + [("transfer_to_human_agents", {"summary": "warranty"}),
                                      "You have been transferred to a human agent."])
        r = agent.send("t", f"I'm {EMAIL}, I want a human")
        self.assertEqual(r.status, "transferred")

    def test_trace_written_and_redacted(self):
        with tempfile.TemporaryDirectory() as d:
            agent, _ = make_agent(AUTH + ["Hello Yusuf."], tracer=Tracer("tr", Path(d)))
            agent.send("tr", f"I'm {EMAIL}")
            recs = load_trace(Path(d) / "tr.jsonl")
        names = [r["name"] for r in recs]
        self.assertIn("guard", names)
        self.assertIn("tool.find_user_id_by_email", names)
        self.assertNotIn(EMAIL, json.dumps(recs))


class TestMCP(unittest.TestCase):
    def test_roundtrip_and_metadata(self):
        db = RetailDB()
        box = MCPToolbox.connect(build_server(build_tools(db)))
        try:
            spec = box.registry.get("cancel_pending_order")
            self.assertEqual((spec.kind, spec.irreversible, spec.approve), ("write", True, True))
            self.assertEqual(spec.input_schema["properties"]["reason"]["enum"], ["no longer needed", "ordered by mistake"])
            self.assertEqual(box.execute("find_user_id_by_email", {"email": EMAIL}), (UID, False))
            out, err = box.execute("cancel_pending_order", {"order_id": "#W4776164", "reason": "no longer needed"})
            self.assertTrue(err)
            self.assertEqual(out, "Error: non-pending order cannot be cancelled")
            self.assertTrue(box.execute("nope", {})[1])
            self.assertFalse(box.execute("cancel_pending_order", CANCEL)[1])
            self.assertEqual(db.orders["#W2378156"]["status"], "cancelled")   # server mutated the backend
        finally:
            box.close()

    def test_annotation_fallback(self):
        from mcp.types import Tool, ToolAnnotations

        t = Tool(name="issue_voucher", description="", input_schema={"type": "object", "properties": {}},
                 annotations=ToolAnnotations(destructive_hint=True))
        s = spec_from_mcp(t)
        self.assertTrue(s.irreversible and s.approve)
        t2 = Tool(name="lookup_balance", description="", input_schema={"type": "object", "properties": {}},
                  annotations=ToolAnnotations(read_only_hint=True))
        self.assertEqual(spec_from_mcp(t2).kind, "read")
        t3 = Tool(name="adjust_points", description="", input_schema={"type": "object", "properties": {}})
        self.assertEqual(spec_from_mcp(t3).kind, "write")   # no hints, unknown verb -> fail closed

    def test_agent_over_mcp(self):
        db = RetailDB()
        box = MCPToolbox.connect(build_server(build_tools(db)))
        try:
            agent, _ = make_agent(AUTH + ["Cancel #W2378156?", ("cancel_pending_order", CANCEL), "It has been cancelled."],
                                  db=db, approval_mode="auto_approve")
            agent.deps.executor = box.execute       # nodes resolve the executor at call time
            agent.send("t", f"I'm {EMAIL}")
            agent.send("t", "yes")
            self.assertEqual(db.orders["#W2378156"]["status"], "cancelled")
        finally:
            box.close()


if __name__ == "__main__":
    unittest.main()
