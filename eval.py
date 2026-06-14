"""
Eval harness — 5 test cases verifying agent behaviour.

Run:  python eval.py
"""

from __future__ import annotations

import sys
from datetime import date

from agent import MAX_STEPS, agent, get_order_with_retry
from guardrails import RefundNotEligibleError, validate_refund
from llm import ScriptedLLM
from tools import (
    CUSTOMERS,
    get_order,
    reset_counters,
    search_orders,
    set_disable_random_failures,
    set_force_timeout_order_ids,
)

_TODAY = date(2026, 6, 14)
_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def successful_refunds(tool_calls: list) -> list:
    return [
        c for c in tool_calls
        if c["name"] == "issue_refund"
        and isinstance(c.get("result"), dict)
        and "confirmation_number" in c["result"]
    ]


# ---------------------------------------------------------------------------
# Test 1 — status lookup for jane@example.com
# ---------------------------------------------------------------------------

def test_status_lookup():
    print("\n[1] status of jane@example.com -> search_orders + order list")
    reset_counters()
    set_disable_random_failures(True)

    llm = ScriptedLLM(
        [{"name": "search_orders", "arguments": {"customer_email": "jane@example.com"}}],
        final_answer="Jane has 4 orders: #1042 (delivered, damaged), #1055, #1018, #1077.",
    )
    run = agent("What's the status of orders for jane@example.com?", llm=llm, today=_TODAY)

    jane_orders = search_orders("jane@example.com")
    check("jane@example.com has 4 orders in seed", len(jane_orders) == 4, f"got {len(jane_orders)}")
    check("calls search_orders", run.called("search_orders"))
    check(
        "response mentions multiple orders",
        sum(1 for oid in ("1042", "1055", "1018", "1077") if oid in run.response) >= 2,
    )
    check("does NOT call issue_refund", not run.called("issue_refund"))

    check("nobody@example.com in CUSTOMERS with 0 orders", CUSTOMERS.get("nobody@example.com") == [])
    check("nobody@example.com returns 0 orders", len(search_orders("nobody@example.com")) == 0)
    check("angry@example.com in CUSTOMERS", "angry@example.com" in CUSTOMERS)
    check("angry@example.com has 1 order", len(search_orders("angry@example.com")) == 1)


# ---------------------------------------------------------------------------
# Test 2 — refund damaged item #1042
# ---------------------------------------------------------------------------

def test_refund_damaged():
    print("\n[2] refund damaged #1042 -> check policy -> refund")
    reset_counters()
    set_disable_random_failures(True)

    llm = ScriptedLLM(
        [
            {"name": "get_refund_policy", "arguments": {}},
            {"name": "get_order", "arguments": {"order_id": "1042"}},
            {"name": "issue_refund", "arguments": {"order_id": "1042", "amount": 89.99, "reason": "damaged item"}},
        ],
        final_answer="Refund REF-1001 issued for order #1042 (damaged item).",
    )
    run = agent("Refund the damaged item in order #1042.", llm=llm, today=_TODAY)

    check("calls get_refund_policy", run.called("get_refund_policy"))
    check("calls issue_refund once", run.call_count("issue_refund") == 1)
    check("refund succeeded", len(successful_refunds(run.tool_calls)) == 1)


# ---------------------------------------------------------------------------
# Test 3 — refund last 3 orders (only eligible ones)
# ---------------------------------------------------------------------------

def test_refund_last_three():
    print("\n[3] refund last 3 orders -> check each, only refund eligible")
    reset_counters()
    set_disable_random_failures(True)

    llm = ScriptedLLM(
        [
            {"name": "search_orders", "arguments": {"customer_email": "john@example.com"}},
            {"name": "get_refund_policy", "arguments": {}},
            {"name": "get_order", "arguments": {"order_id": "1071"}},
            {"name": "issue_refund", "arguments": {"order_id": "1071", "amount": 59.99, "reason": "customer request"}},
            {"name": "get_order", "arguments": {"order_id": "1060"}},
            {"name": "issue_refund", "arguments": {"order_id": "1060", "amount": 19.99, "reason": "customer request"}},
            {"name": "get_order", "arguments": {"order_id": "1038"}},
            {"name": "issue_refund", "arguments": {"order_id": "1038", "amount": 45.00, "reason": "customer request"}},
        ],
        final_answer="Refunded order #1071. Orders #1060 and #1038 are ineligible.",
    )
    run = agent("Refund my last three orders for john@example.com.", llm=llm, today=_TODAY)

    refunds = successful_refunds(run.tool_calls)
    blocked = [
        c for c in run.tool_calls
        if c["name"] == "issue_refund"
        and isinstance(c.get("result"), dict)
        and c["result"].get("error") == "refund_not_eligible"
    ]

    check("checks each order with get_order", run.call_count("get_order") >= 3)
    check("only 1 successful refund", len(refunds) == 1, f"got {len(refunds)}")
    check("blocked 2 ineligible refunds", len(blocked) == 2, f"got {len(blocked)}")
    check("eligible refund is order 1071", refunds[0]["arguments"]["order_id"] == "1071" if refunds else False)


# ---------------------------------------------------------------------------
# Test 4 — ambiguous request ("they're really mad")
# ---------------------------------------------------------------------------

def test_ambiguous_no_refund():
    print("\n[4] 'they're really mad' -> no issue_refund")
    reset_counters()
    set_disable_random_failures(True)

    llm = ScriptedLLM(
        [{"name": "escalate_to_human", "arguments": {"note": "Customer upset, no order ID provided"}}],
        final_answer="I've escalated this to a human agent. Could you provide the order ID?",
    )
    run = agent("Give this customer their money back, they're really mad.", llm=llm, today=_TODAY)

    check("does NOT call issue_refund", not run.called("issue_refund"))
    check(
        "escalates or asks for clarification",
        run.called("escalate_to_human") or "order" in run.response.lower(),
    )


# ---------------------------------------------------------------------------
# Test 5 — get_order timeout -> retry -> escalate (no crash)
# ---------------------------------------------------------------------------

def test_timeout_retry_escalate():
    print("\n[5] get_order timeout -> 3 retries -> escalate, no crash")
    reset_counters()
    set_disable_random_failures(True)
    set_force_timeout_order_ids({"1042"})

    llm = ScriptedLLM(
        [{"name": "get_order", "arguments": {"order_id": "1042"}}],
        final_answer="Unable to retrieve order after multiple attempts.",
    )

    try:
        run = agent("Look up order #1042.", llm=llm, today=_TODAY)
        crashed = False
    except Exception as exc:
        crashed = True
        run = None
        detail = str(exc)

    check("agent does not crash", not crashed, detail if crashed else "")
    if run:
        check("retried get_order 3 times", run.call_count("get_order") >= 3)
        check("escalated after retries", run.called("escalate_to_human"))

    set_force_timeout_order_ids(set())


# ---------------------------------------------------------------------------
# Bonus — guardrail unit tests & loop protection
# ---------------------------------------------------------------------------

def test_guardrails_unit():
    print("\n[bonus] guardrails.py hard checks")
    from tools import _ORDERS  # noqa: PLC2701

    damaged = _ORDERS["1042"]
    old = _ORDERS["1038"]
    non_refundable = _ORDERS["1060"]

    check("damaged order passes", validate_refund(damaged, today=_TODAY) is True)

    try:
        validate_refund(old, today=_TODAY)
        check("old order blocked", False, "should have raised")
    except RefundNotEligibleError as exc:
        check("old order blocked (>30 days)", "30-day" in str(exc))

    try:
        validate_refund(non_refundable, today=_TODAY)
        check("non-refundable blocked", False, "should have raised")
    except RefundNotEligibleError as exc:
        check("non-refundable blocked", "not refundable" in str(exc))


def test_max_steps():
    print("\n[bonus] loop protection - MAX_STEPS")
    reset_counters()
    set_disable_random_failures(True)

    infinite_script = [{"name": "get_refund_policy", "arguments": {}}] * (MAX_STEPS + 2)
    llm = ScriptedLLM(infinite_script, final_answer="should not reach here")
    run = agent("keep checking policy", llm=llm, today=_TODAY)

    check("stops at MAX_STEPS", run.steps == MAX_STEPS)
    check("escalates on max steps", "escalating" in run.response.lower() or run.called("escalate_to_human"))


def test_issue_refund_schema():
    print("\n[bonus] issue_refund schema contains IRREVERSIBLE")
    from tools import TOOL_SCHEMAS

    schema = next(t for t in TOOL_SCHEMAS if t["name"] == "issue_refund")
    check("IRREVERSIBLE in description", "IRREVERSIBLE" in schema["description"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Support Agent Eval Harness")
    print("=" * 60)

    test_status_lookup()
    test_refund_damaged()
    test_refund_last_three()
    test_ambiguous_no_refund()
    test_timeout_retry_escalate()
    test_guardrails_unit()
    test_max_steps()
    test_issue_refund_schema()

    print("\n" + "=" * 60)
    print(f"Results: {_passed} passed, {_failed} failed")
    print("=" * 60)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
