import os
import unittest
from unittest import mock

import helpers  # noqa: F401  (path setup)

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi not installed")
class TestAPI(unittest.TestCase):
    def setUp(self):
        from helpers import EMAIL, UID, POLICY, Deps, RetailDB, ScriptedLLM, SupportAgent, Tracer, build_tools

        from supportagent.api import main

        self.main = main
        script = [("find_user_id_by_email", {"email": EMAIL}), ("get_user_details", {"user_id": UID}),
                  "Cancel order #W2378156 (no longer needed)?",
                  ("cancel_pending_order", {"order_id": "#W2378156", "reason": "no longer needed"}),
                  "Order #W2378156 has been cancelled."]
        self.db = RetailDB()

        def fake_agent(tid):
            return SupportAgent(Deps(ScriptedLLM(script), build_tools(self.db), POLICY, approval_mode="human",
                                     tracer=Tracer(tid, enabled=False)), backend="mini")

        self.patch = mock.patch.object(main, "_make_agent", fake_agent)
        self.patch.start()
        os.environ["OPERATOR_TOKEN"] = "secret"
        self.c = TestClient(main.app)

    def tearDown(self):
        self.patch.stop()

    def test_flow(self):
        from helpers import EMAIL

        tid = self.c.post("/v1/sessions", json={}).json()["thread_id"]
        r = self.c.post(f"/v1/sessions/{tid}/messages", json={"text": f"I'm {EMAIL}, cancel #W2378156"}).json()
        self.assertEqual(r["status"], "awaiting_user")
        r = self.c.post(f"/v1/sessions/{tid}/messages", json={"text": "yes"}).json()
        self.assertEqual(r["status"], "awaiting_approval")
        self.assertEqual(self.c.post(f"/v1/sessions/{tid}/messages", json={"text": "hello?"}).status_code, 409)
        self.assertEqual(self.c.get("/v1/approvals").status_code, 401)
        hdr = {"X-Operator-Token": "secret", "X-Reviewer": "alice"}
        pending = self.c.get("/v1/approvals", headers=hdr).json()["pending"]
        cid = pending[0]["actions"][0]["call_id"]
        self.assertEqual(self.c.post(f"/v1/approvals/{tid}", headers=hdr, json={"decisions": {}}).status_code, 422)
        r = self.c.post(f"/v1/approvals/{tid}", headers=hdr,
                        json={"decisions": {cid: {"approved": True, "reason": "ok"}}}).json()
        self.assertEqual(r["reply"], "Order #W2378156 has been cancelled.")
        s = self.c.get(f"/v1/sessions/{tid}").json()
        self.assertEqual(s["approvals"][0]["reviewer"], "alice")
        self.assertEqual(self.db.orders["#W2378156"]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
