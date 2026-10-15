"""A small, self-contained retail database in the same shape as tau2-bench's retail domain
(users, orders, products, payment methods), used for offline tests and the demo.

tau2's real retail domain has hundreds of users and products; this one has a handful,
chosen so that every policy rule has a case that passes and a case that should be refused.
"""
from __future__ import annotations

import copy
import hashlib
import json


def _item(item_id, product_id, name, price, **options):
    return {"item_id": item_id, "product_id": product_id, "name": name, "price": price, "options": options}


PRODUCTS = {
    "6086499569": {"name": "T-Shirt", "product_id": "6086499569", "variants": {
        "9612497925": {"item_id": "9612497925", "options": {"color": "blue", "size": "M"}, "available": True, "price": 50.88},
        "8124970213": {"item_id": "8124970213", "options": {"color": "purple", "size": "XL"}, "available": True, "price": 49.67},
        "1176194968": {"item_id": "1176194968", "options": {"color": "black", "size": "S"}, "available": False, "price": 52.88},
    }},
    "1656367028": {"name": "Mechanical Keyboard", "product_id": "1656367028", "variants": {
        "1151293680": {"item_id": "1151293680", "options": {"switch": "linear", "layout": "full"}, "available": True, "price": 272.33},
        "7706410293": {"item_id": "7706410293", "options": {"switch": "clicky", "layout": "full"}, "available": True, "price": 269.16},
    }},
    "4768869376": {"name": "Water Bottle", "product_id": "4768869376", "variants": {
        "3453331371": {"item_id": "3453331371", "options": {"capacity": "500ml", "color": "black"}, "available": True, "price": 52.79},
        "2439754078": {"item_id": "2439754078", "options": {"capacity": "1000ml", "color": "red"}, "available": True, "price": 49.51},
    }},
}

USERS = {
    "yusuf_rossi_9620": {
        "user_id": "yusuf_rossi_9620", "name": {"first_name": "Yusuf", "last_name": "Rossi"},
        "email": "yusuf.rossi7301@example.com",
        "address": {"address1": "763 Broadway", "address2": "Suite 135", "city": "Philadelphia",
                    "state": "PA", "zip": "19122", "country": "USA"},
        "payment_methods": {
            "credit_card_9513926": {"source": "credit_card", "brand": "mastercard", "last_four": "2478",
                                    "id": "credit_card_9513926"},
            "gift_card_6118653": {"source": "gift_card", "balance": 41.0, "id": "gift_card_6118653"},
        },
        "orders": ["#W2378156", "#W4776164"],
    },
    "mei_kovacs_8020": {
        "user_id": "mei_kovacs_8020", "name": {"first_name": "Mei", "last_name": "Kovacs"},
        "email": "mei.kovacs8232@example.com",
        "address": {"address1": "317 Elm Street", "address2": "Suite 461", "city": "Charlotte",
                    "state": "NC", "zip": "28236", "country": "USA"},
        "payment_methods": {"paypal_7644869": {"source": "paypal", "id": "paypal_7644869"}},
        "orders": ["#W6390527"],
    },
}

ORDERS = {
    "#W2378156": {
        "order_id": "#W2378156", "user_id": "yusuf_rossi_9620", "status": "pending",
        "address": copy.deepcopy(USERS["yusuf_rossi_9620"]["address"]),
        "items": [_item("9612497925", "6086499569", "T-Shirt", 50.88, color="blue", size="M"),
                  _item("1151293680", "1656367028", "Mechanical Keyboard", 272.33, switch="linear", layout="full")],
        "payment_history": [{"transaction_type": "payment", "amount": 323.21, "payment_method_id": "credit_card_9513926"}],
    },
    "#W4776164": {
        "order_id": "#W4776164", "user_id": "yusuf_rossi_9620", "status": "delivered",
        "address": copy.deepcopy(USERS["yusuf_rossi_9620"]["address"]),
        "items": [_item("3453331371", "4768869376", "Water Bottle", 52.79, capacity="500ml", color="black")],
        "payment_history": [{"transaction_type": "payment", "amount": 52.79, "payment_method_id": "credit_card_9513926"}],
    },
    "#W6390527": {
        "order_id": "#W6390527", "user_id": "mei_kovacs_8020", "status": "delivered",
        "address": copy.deepcopy(USERS["mei_kovacs_8020"]["address"]),
        "items": [_item("8124970213", "6086499569", "T-Shirt", 49.67, color="purple", size="XL")],
        "payment_history": [{"transaction_type": "payment", "amount": 49.67, "payment_method_id": "paypal_7644869"}],
    },
}


class RetailDB:
    def __init__(self, data: dict | None = None):
        d = data or {"users": USERS, "orders": ORDERS, "products": PRODUCTS}
        self.users = copy.deepcopy(d["users"])
        self.orders = copy.deepcopy(d["orders"])
        self.products = copy.deepcopy(d["products"])

    def dump(self) -> dict:
        return {"users": self.users, "orders": self.orders, "products": self.products}

    def hash(self) -> str:
        """Same idea as tau2's DB check: success = the final DB equals the expected one."""
        return hashlib.sha256(json.dumps(self.dump(), sort_keys=True).encode()).hexdigest()[:16]

    def copy(self) -> "RetailDB":
        return RetailDB(self.dump())
