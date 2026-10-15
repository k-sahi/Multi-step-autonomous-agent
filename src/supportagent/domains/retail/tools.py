"""Retail tools over RetailDB, with the same names and semantics as tau2-bench retail.

The tool functions enforce *hard* invariants (an order must be pending to cancel). Policy
rules that depend on the conversation (was the user authenticated? did they say yes?) are
enforced by the guardrail layer, because a tool can't see the conversation.
"""
from __future__ import annotations

import copy
import json
from typing import Literal

from ...tools.registry import ToolExecutionError, ToolRegistry, ToolSpec
from .data import RetailDB

S = lambda **p: {"type": "object", "properties": p, "required": list(p), "additionalProperties": False}  # noqa: E731
STR = {"type": "string"}
STR_LIST = {"type": "array", "items": {"type": "string"}}
CANCEL_REASONS = ["no longer needed", "ordered by mistake"]


def build_tools(db: RetailDB) -> ToolRegistry:
    reg = ToolRegistry()

    def order(order_id: str) -> dict:
        o = db.orders.get(order_id)
        if o is None:
            raise ToolExecutionError(f"order {order_id} not found")
        return o

    def user(user_id: str) -> dict:
        u = db.users.get(user_id)
        if u is None:
            raise ToolExecutionError(f"user {user_id} not found")
        return u

    def payment(u: dict, pm_id: str) -> dict:
        pm = u["payment_methods"].get(pm_id)
        if pm is None:
            raise ToolExecutionError(f"payment method {pm_id} not found for user")
        return pm

    # ------------------------------------------------------------------ reads
    def find_user_id_by_email(email: str) -> str:
        for uid, u in db.users.items():
            if u["email"].lower() == email.lower():
                return uid
        raise ToolExecutionError("user not found")

    def find_user_id_by_name_zip(first_name: str, last_name: str, zip: str) -> str:
        for uid, u in db.users.items():
            n = u["name"]
            if (n["first_name"].lower(), n["last_name"].lower(), u["address"]["zip"]) == \
               (first_name.lower(), last_name.lower(), zip):
                return uid
        raise ToolExecutionError("user not found")

    def get_user_details(user_id: str) -> dict:
        return user(user_id)

    def get_order_details(order_id: str) -> dict:
        return order(order_id)

    def get_product_details(product_id: str) -> dict:
        p = db.products.get(product_id)
        if p is None:
            raise ToolExecutionError("product not found")
        return p

    def list_all_product_types() -> dict:
        return {p["name"]: pid for pid, p in db.products.items()}

    def calculate(expression: str) -> str:
        if not set(expression) <= set("0123456789+-*/(). "):
            raise ToolExecutionError("invalid characters in expression")
        return str(round(eval(expression, {"__builtins__": {}}, {}), 2))  # noqa: S307 - charset-restricted

    # ------------------------------------------------------------------ writes
    def cancel_pending_order(order_id: str, reason: Literal["no longer needed", "ordered by mistake"]) -> dict:
        o = order(order_id)
        if o["status"] != "pending":
            raise ToolExecutionError("non-pending order cannot be cancelled")
        if reason not in CANCEL_REASONS:
            raise ToolExecutionError(f"invalid reason; must be one of {CANCEL_REASONS}")
        refunds = [{"transaction_type": "refund", "amount": p["amount"], "payment_method_id": p["payment_method_id"]}
                   for p in o["payment_history"] if p["transaction_type"] == "payment"]
        o["payment_history"] += refunds
        o["status"] = "cancelled"
        o["cancel_reason"] = reason
        for r in refunds:  # gift card refunds are immediate
            pm = db.users[o["user_id"]]["payment_methods"].get(r["payment_method_id"], {})
            if pm.get("source") == "gift_card":
                pm["balance"] = round(pm["balance"] + r["amount"], 2)
        return o

    def modify_pending_order_address(order_id: str, address1: str, address2: str, city: str, state: str,
                                     country: str, zip: str) -> dict:
        o = order(order_id)
        if o["status"] != "pending":
            raise ToolExecutionError("non-pending order cannot be modified")
        o["address"] = {"address1": address1, "address2": address2, "city": city, "state": state,
                        "country": country, "zip": zip}
        return o

    def modify_pending_order_items(order_id: str, item_ids: list[str], new_item_ids: list[str],
                                   payment_method_id: str) -> dict:
        o = order(order_id)
        if o["status"] != "pending":
            raise ToolExecutionError("non-pending order cannot be modified")
        diff = _swap_items(o, item_ids, new_item_ids)
        pm = payment(db.users[o["user_id"]], payment_method_id)
        _charge(pm, diff)
        o["payment_history"].append({"transaction_type": "payment" if diff > 0 else "refund",
                                     "amount": abs(diff), "payment_method_id": payment_method_id})
        o["status"] = "pending (item modified)"
        return o

    def return_delivered_order_items(order_id: str, item_ids: list[str], payment_method_id: str) -> dict:
        o = order(order_id)
        if o["status"] != "delivered":
            raise ToolExecutionError("non-delivered order cannot be returned")
        u = db.users[o["user_id"]]
        pm = payment(u, payment_method_id)
        original = {p["payment_method_id"] for p in o["payment_history"]}
        if pm["source"] != "gift_card" and payment_method_id not in original:
            raise ToolExecutionError("refund must go to the original payment method or a gift card")
        ids = [i["item_id"] for i in o["items"]]
        for iid in item_ids:
            if iid not in ids:
                raise ToolExecutionError(f"item {iid} not in order")
        o["status"] = "return requested"
        o["return_items"] = sorted(item_ids)
        o["return_payment_method_id"] = payment_method_id
        return o

    def exchange_delivered_order_items(order_id: str, item_ids: list[str], new_item_ids: list[str],
                                       payment_method_id: str) -> dict:
        o = order(order_id)
        if o["status"] != "delivered":
            raise ToolExecutionError("non-delivered order cannot be exchanged")
        probe = copy.deepcopy(o)
        diff = _swap_items(probe, item_ids, new_item_ids)
        payment(db.users[o["user_id"]], payment_method_id)
        o["status"] = "exchange requested"
        o["exchange_items"] = sorted(item_ids)
        o["exchange_new_items"] = sorted(new_item_ids)
        o["exchange_payment_method_id"] = payment_method_id
        o["exchange_price_difference"] = round(diff, 2)
        return o

    def modify_user_address(user_id: str, address1: str, address2: str, city: str, state: str,
                            country: str, zip: str) -> dict:
        u = user(user_id)
        u["address"] = {"address1": address1, "address2": address2, "city": city, "state": state,
                        "country": country, "zip": zip}
        return u

    def transfer_to_human_agents(summary: str) -> str:
        return "Transfer successful"

    def _swap_items(o: dict, item_ids: list[str], new_item_ids: list[str]) -> float:
        if len(item_ids) != len(new_item_ids):
            raise ToolExecutionError("the number of items to swap must match")
        diff = 0.0
        for old_id, new_id in zip(item_ids, new_item_ids):
            item = next((i for i in o["items"] if i["item_id"] == old_id), None)
            if item is None:
                raise ToolExecutionError(f"item {old_id} not found in order")
            variant = db.products[item["product_id"]]["variants"].get(new_id)
            if variant is None:
                raise ToolExecutionError(f"new item {new_id} is not a variant of the same product")
            if not variant["available"]:
                raise ToolExecutionError(f"new item {new_id} is not available")
            diff += variant["price"] - item["price"]
            item.update(item_id=new_id, price=variant["price"], options=variant["options"])
        return round(diff, 2)

    def _charge(pm: dict, amount: float) -> None:
        if pm["source"] == "gift_card":
            if amount > pm["balance"]:
                raise ToolExecutionError("insufficient gift card balance")
            pm["balance"] = round(pm["balance"] - amount, 2)

    addr = dict(address1=STR, address2=STR, city=STR, state=STR, country=STR, zip=STR)
    specs = [
        ToolSpec("find_user_id_by_email", "Find a user id by email. Use to authenticate the user.",
                 S(email=STR), find_user_id_by_email),
        ToolSpec("find_user_id_by_name_zip", "Find a user id by first name, last name and zip code.",
                 S(first_name=STR, last_name=STR, zip=STR), find_user_id_by_name_zip),
        ToolSpec("get_user_details", "Get a user's profile, addresses, payment methods and order ids.",
                 S(user_id=STR), get_user_details),
        ToolSpec("get_order_details", "Get the status, items and payments of an order, e.g. '#W0000000'.",
                 S(order_id=STR), get_order_details),
        ToolSpec("get_product_details", "Get a product and its item variants with prices and availability.",
                 S(product_id=STR), get_product_details),
        ToolSpec("list_all_product_types", "List product names and their product ids.",
                 {"type": "object", "properties": {}, "required": []}, list_all_product_types),
        ToolSpec("calculate", "Evaluate an arithmetic expression, e.g. '52.79 - 49.51'.",
                 S(expression=STR), calculate, kind="generic"),
        ToolSpec("cancel_pending_order", "Cancel a pending order. Refunds go to the original payment methods.",
                 S(order_id=STR, reason={"type": "string", "enum": CANCEL_REASONS}), cancel_pending_order,
                 kind="write", irreversible=True, confirm=True, approve=True,
                 risk_note="Irreversible: the order is cancelled and refunds issued."),
        ToolSpec("modify_pending_order_address", "Change the shipping address of a pending order.",
                 S(order_id=STR, **addr), modify_pending_order_address, kind="write", confirm=True),
        ToolSpec("modify_pending_order_items",
                 "Swap items in a pending order for other variants of the same product. Can only be done once.",
                 S(order_id=STR, item_ids=STR_LIST, new_item_ids=STR_LIST, payment_method_id=STR),
                 modify_pending_order_items, kind="write", irreversible=True, confirm=True, approve=True,
                 risk_note="Irreversible: an order's items can only be modified once; charges or refunds the difference."),
        ToolSpec("return_delivered_order_items", "Request a return of delivered items.",
                 S(order_id=STR, item_ids=STR_LIST, payment_method_id=STR), return_delivered_order_items,
                 kind="write", irreversible=True, confirm=True, approve=True,
                 risk_note="Irreversible: starts a refund to the chosen payment method."),
        ToolSpec("exchange_delivered_order_items", "Request an exchange of delivered items for other variants.",
                 S(order_id=STR, item_ids=STR_LIST, new_item_ids=STR_LIST, payment_method_id=STR),
                 exchange_delivered_order_items, kind="write", irreversible=True, confirm=True, approve=True,
                 risk_note="Irreversible: an order can only be exchanged once."),
        ToolSpec("modify_user_address", "Change the user's default address.",
                 S(user_id=STR, **addr), modify_user_address, kind="write", confirm=True),
        ToolSpec("transfer_to_human_agents",
                 "Transfer the user to a human agent, with a summary. Only if the request is out of scope "
                 "or the user explicitly asks for a human.", S(summary=STR), transfer_to_human_agents,
                 kind="transfer"),
    ]
    for s in specs:
        reg.add(s)
    return reg


def tool_output_json(x) -> str:
    return json.dumps(x, default=str)
