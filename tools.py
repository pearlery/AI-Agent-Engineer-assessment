"""Mock backend tools for the e-commerce support agent."""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_TODAY = date(2026, 6, 14)

# Explicit customer registry (emails with zero orders are listed with [])
CUSTOMERS: dict[str, list[str]] = {
    "jane@example.com": ["1042", "1055", "1018", "1077"],
    "john@example.com": ["1071", "1060", "1038"],
    "angry@example.com": ["1099"],
    "nobody@example.com": [],
}

_ORDERS: dict[str, dict[str, Any]] = {
    "1042": {
        "order_id": "1042",
        "customer_email": "jane@example.com",
        "status": "delivered",
        "items": [{"name": "Wireless Headphones", "qty": 1, "price": 89.99}],
        "total": 89.99,
        "order_date": (_TODAY - timedelta(days=10)).isoformat(),
        "delivered_date": (_TODAY - timedelta(days=5)).isoformat(),
        "refundable": True,
        "damaged": True,
    },
    "1055": {
        "order_id": "1055",
        "customer_email": "jane@example.com",
        "status": "delivered",
        "items": [{"name": "USB-C Cable", "qty": 2, "price": 12.99}],
        "total": 25.98,
        "order_date": (_TODAY - timedelta(days=20)).isoformat(),
        "delivered_date": (_TODAY - timedelta(days=15)).isoformat(),
        "refundable": True,
        "damaged": False,
    },
    "1018": {
        "order_id": "1018",
        "customer_email": "jane@example.com",
        "status": "delivered",
        "items": [{"name": "Laptop Stand", "qty": 1, "price": 34.99}],
        "total": 34.99,
        "order_date": (_TODAY - timedelta(days=55)).isoformat(),
        "delivered_date": (_TODAY - timedelta(days=40)).isoformat(),  # > 30 days
        "refundable": True,
        "damaged": False,
    },
    "1077": {
        "order_id": "1077",
        "customer_email": "jane@example.com",
        "status": "shipped",
        "items": [{"name": "Screen Protector", "qty": 1, "price": 14.99}],
        "total": 14.99,
        "order_date": (_TODAY - timedelta(days=3)).isoformat(),
        "delivered_date": None,
        "refundable": True,
        "damaged": False,
    },
    "1099": {
        "order_id": "1099",
        "customer_email": "angry@example.com",
        "status": "delivered",
        "items": [{"name": "Gaming Mouse", "qty": 1, "price": 79.99}],
        "total": 79.99,
        "order_date": (_TODAY - timedelta(days=8)).isoformat(),
        "delivered_date": (_TODAY - timedelta(days=3)).isoformat(),
        "refundable": True,
        "damaged": False,
    },
    "1038": {
        "order_id": "1038",
        "customer_email": "john@example.com",
        "status": "delivered",
        "items": [{"name": "Desk Lamp", "qty": 1, "price": 45.00}],
        "total": 45.00,
        "order_date": (_TODAY - timedelta(days=60)).isoformat(),
        "delivered_date": (_TODAY - timedelta(days=45)).isoformat(),  # > 30 days
        "refundable": True,
        "damaged": False,
    },
    "1060": {
        "order_id": "1060",
        "customer_email": "john@example.com",
        "status": "delivered",
        "items": [{"name": "Phone Case", "qty": 1, "price": 19.99}],
        "total": 19.99,
        "order_date": (_TODAY - timedelta(days=25)).isoformat(),
        "delivered_date": (_TODAY - timedelta(days=20)).isoformat(),
        "refundable": False,
        "damaged": False,
    },
    "1071": {
        "order_id": "1071",
        "customer_email": "john@example.com",
        "status": "delivered",
        "items": [{"name": "Bluetooth Speaker", "qty": 1, "price": 59.99}],
        "total": 59.99,
        "order_date": (_TODAY - timedelta(days=12)).isoformat(),
        "delivered_date": (_TODAY - timedelta(days=7)).isoformat(),
        "refundable": True,
        "damaged": False,
    },
}

_REFUND_POLICY = (
    "Refund Policy:\n"
    "- Refunds are allowed within 30 days of delivery.\n"
    "- Damaged items are always eligible for a full refund regardless of delivery date.\n"
    "- Non-refundable items (final sale) cannot be refunded unless damaged.\n"
    "- All refunds require manager approval for amounts over $500."
)

_CONFIRMATION_COUNTER = 1000
_TICKET_COUNTER = 5000

# Test hooks — override in eval to control flaky behaviour
_force_timeout_order_ids: set[str] = set()
_disable_random_failures = False


def set_force_timeout_order_ids(order_ids: set[str]) -> None:
    global _force_timeout_order_ids
    _force_timeout_order_ids = order_ids


def set_disable_random_failures(disable: bool) -> None:
    global _disable_random_failures
    _disable_random_failures = disable


def reset_counters() -> None:
    global _CONFIRMATION_COUNTER, _TICKET_COUNTER
    _CONFIRMATION_COUNTER = 1000
    _TICKET_COUNTER = 5000


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def search_orders(customer_email: str) -> list[dict[str, Any]]:
    """Return all orders for a customer email."""
    if customer_email not in CUSTOMERS:
        return []

    results = [
        {
            "order_id": o["order_id"],
            "status": o["status"],
            "items": o["items"],
            "total": o["total"],
            "order_date": o["order_date"],
        }
        for o in _ORDERS.values()
        if o["customer_email"] == customer_email
    ]
    return sorted(results, key=lambda x: x["order_date"], reverse=True)


def get_order(order_id: str) -> dict[str, Any]:
    """Return full order detail. Fails ~20% of the time with a timeout."""
    order_id = order_id.lstrip("#")

    if order_id in _force_timeout_order_ids:
        raise TimeoutError("Service timeout — order service unavailable")

    if not _disable_random_failures and random.random() < 0.20:
        raise TimeoutError("Service timeout — order service unavailable")

    if order_id not in _ORDERS:
        raise ValueError(f"Order {order_id} not found")

    return dict(_ORDERS[order_id])


def get_refund_policy() -> str:
    """Return the current refund policy text."""
    return _REFUND_POLICY


def issue_refund(order_id: str, amount: float, reason: str) -> dict[str, Any]:
    """IRREVERSIBLE — Issue a refund and return a confirmation number."""
    global _CONFIRMATION_COUNTER
    order_id = order_id.lstrip("#")

    if order_id not in _ORDERS:
        raise ValueError(f"Order {order_id} not found")

    _CONFIRMATION_COUNTER += 1
    return {
        "confirmation_number": f"REF-{_CONFIRMATION_COUNTER}",
        "order_id": order_id,
        "amount": amount,
        "reason": reason,
        "status": "refund_issued",
    }


def escalate_to_human(order_id: str | None, note: str) -> dict[str, Any]:
    """File a support ticket for a human agent."""
    global _TICKET_COUNTER
    _TICKET_COUNTER += 1
    return {
        "ticket_id": f"TKT-{_TICKET_COUNTER}",
        "order_id": order_id,
        "note": note,
        "status": "escalated",
    }


# ---------------------------------------------------------------------------
# Tool schemas (for LLM function-calling)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_orders",
        "description": "Search for all orders belonging to a customer by email address.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_email": {
                    "type": "string",
                    "description": "Customer email address",
                }
            },
            "required": ["customer_email"],
        },
    },
    {
        "name": "get_order",
        "description": "Get full details for a single order including delivery date and refund eligibility flags.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID (e.g. '1042' or '#1042')",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "get_refund_policy",
        "description": "Retrieve the current refund policy. Call this before issuing any refund.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "issue_refund",
        "description": (
            "IRREVERSIBLE — Issue a monetary refund for an order. "
            "This action cannot be undone. Only call after verifying eligibility "
            "via get_order and get_refund_policy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID to refund"},
                "amount": {"type": "number", "description": "Refund amount in USD"},
                "reason": {"type": "string", "description": "Reason for the refund"},
            },
            "required": ["order_id", "amount", "reason"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate a case to a human support agent when the request is ambiguous or tools fail.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Related order ID, if known",
                },
                "note": {"type": "string", "description": "Context for the human agent"},
            },
            "required": ["note"],
        },
    },
]
