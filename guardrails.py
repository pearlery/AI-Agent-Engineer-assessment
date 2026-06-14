"""Hard-coded guardrails — cannot be bypassed by the LLM."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


class RefundNotEligibleError(Exception):
    """Raised when an order fails refund eligibility checks."""


def days_since_delivery(order: dict[str, Any], today: date | None = None) -> int:
    """Return number of days since the order was delivered."""
    if today is None:
        today = date.today()
    delivered = order["delivered_date"]
    if isinstance(delivered, str):
        delivered = datetime.fromisoformat(delivered).date()
    return (today - delivered).days


def validate_refund(order: dict[str, Any], today: date | None = None) -> bool:
    """
    Hard check in code — not bypassable by the LLM.

    Returns True if the order is eligible for refund.
    Raises RefundNotEligibleError with a clear reason otherwise.
    """
    if order.get("damaged"):
        return True  # always ok

    if days_since_delivery(order, today) > 30:
        raise RefundNotEligibleError("Outside 30-day window")

    if not order.get("refundable"):
        raise RefundNotEligibleError("Order not refundable")

    return True
