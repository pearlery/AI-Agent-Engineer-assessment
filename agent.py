"""ReAct agent loop with guardrails, retry logic, and observability."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from typing import Any

from guardrails import RefundNotEligibleError, validate_refund
from logging_config import clear_run_id, set_run_id
from llm import LLM, GeminiLLM, ScriptedLLM, get_default_llm
from tools import (
    TOOL_SCHEMAS,
    escalate_to_human,
    get_order,
    get_refund_policy,
    issue_refund,
    search_orders,
    set_disable_random_failures,
)

logger = logging.getLogger(__name__)

MAX_STEPS = 10

SYSTEM_PROMPT = (
    "You are an internal e-commerce support agent. "
    "Use tools to resolve support requests. "
    "If a customer email is mentioned but no order ID, call search_orders first to find their orders. "
    "Always check get_refund_policy before issuing any refund. "
    "Verify each order's eligibility with get_order before calling issue_refund. "
    "If the request mentions multiple orders, check each one individually with get_order. "
    "If the request is ambiguous (no email and no order ID), ask for clarification or escalate — never guess."
)


class AgentRun:
    """Result of a single agent run with full observability."""

    def __init__(self, response: str, tool_calls: list[dict[str, Any]], steps: int, history: list[dict[str, Any]]):
        self.response = response
        self.tool_calls = tool_calls
        self.steps = steps
        self.history = history

    def called(self, tool_name: str) -> bool:
        return any(c["name"] == tool_name for c in self.tool_calls)

    def call_count(self, tool_name: str) -> int:
        return sum(1 for c in self.tool_calls if c["name"] == tool_name)


def get_order_with_retry(
    order_id: str,
    tool_calls_log: list[dict[str, Any]],
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """Retry get_order up to 3 times; escalate on persistent timeout."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = get_order(order_id)
            tool_calls_log.append(
                {"name": "get_order", "arguments": {"order_id": order_id}, "result": result, "attempt": attempt + 1}
            )
            logger.info("get_order succeeded", extra={"order_id": order_id, "attempt": attempt + 1})
            return result
        except TimeoutError as exc:
            last_error = exc
            tool_calls_log.append(
                {
                    "name": "get_order",
                    "arguments": {"order_id": order_id},
                    "result": {"error": str(exc)},
                    "attempt": attempt + 1,
                }
            )
            logger.warning("get_order timeout", extra={"order_id": order_id, "attempt": attempt + 1})
            if attempt == 2:
                ticket = escalate_to_human(order_id, "get_order timeout after 3 retries")
                tool_calls_log.append(
                    {
                        "name": "escalate_to_human",
                        "arguments": {"order_id": order_id, "note": "get_order timeout after 3 retries"},
                        "result": ticket,
                    }
                )
                return {"error": "escalated", "ticket": ticket, "order_id": order_id}

    raise last_error  # unreachable, satisfies type checker


def _coerce_args(name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Best-effort type coercion for common LLM argument mistakes."""
    args = dict(args or {})
    # order_id can arrive as int from some models
    if "order_id" in args and args["order_id"] is not None:
        args["order_id"] = str(args["order_id"])
    # amount can arrive as a string "89.99"
    if name == "issue_refund" and "amount" in args:
        try:
            args["amount"] = float(args["amount"])
        except (ValueError, TypeError):
            pass  # let downstream raise a clear error
    return args


def execute_tool(
    action: dict[str, Any],
    tool_calls_log: list[dict[str, Any]],
    *,
    today: date | None = None,
    order_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a tool call with guardrails and logging."""
    name = action.get("name") or ""
    if not name:
        result = {"error": "missing_tool_name", "message": "LLM returned tool call with no name"}
        logger.error("missing_tool_name", extra={"action": action})
        tool_calls_log.append({"name": "unknown", "arguments": {}, "result": result})
        return {"role": "tool", "name": "unknown", "result": result}

    args = _coerce_args(name, action.get("arguments"))
    if order_cache is None:
        order_cache = {}

    logger.info("tool_call", extra={"tool": name, "tool_args": args})

    try:
        if name == "search_orders":
            result = search_orders(args["customer_email"])
        elif name == "get_order":
            oid = args["order_id"]
            if oid not in order_cache:
                order_cache[oid] = get_order_with_retry(args["order_id"], tool_calls_log, today=today)
            result = order_cache[oid]
            return {"role": "tool", "name": name, "result": result}
        elif name == "get_refund_policy":
            result = get_refund_policy()
        elif name == "issue_refund":
            oid = args["order_id"]
            order = order_cache.get(oid) or get_order_with_retry(oid, tool_calls_log, today=today)
            order_cache[oid] = order
            if "error" in order:
                result = order
            else:
                validate_refund(order, today=today)
                result = issue_refund(args["order_id"], args["amount"], args["reason"])
        elif name == "escalate_to_human":
            result = escalate_to_human(args.get("order_id"), args["note"])
        else:
            result = {"error": f"Unknown tool: {name}"}
    except RefundNotEligibleError as exc:
        result = {"error": "refund_not_eligible", "message": str(exc)}
        logger.warning("guardrail blocked refund", extra={"order_id": args.get("order_id"), "reason": str(exc)})
    except (KeyError, ValueError, TypeError) as exc:
        result = {"error": "invalid_arguments", "message": str(exc)}
        logger.error("malformed tool call", extra={"tool": name, "error": str(exc)})
    except Exception as exc:
        result = {"error": "tool_failure", "message": str(exc)}
        logger.error("tool failure", extra={"tool": name, "error": str(exc)})

    tool_calls_log.append({"name": name, "arguments": args, "result": result})
    return {"role": "tool", "name": name, "result": result}


def run(
    user_message: str,
    llm: LLM | None = None,
    *,
    today: date | None = None,
    disable_random_failures: bool = False,
) -> AgentRun:
    """
    Main ReAct loop:
        while not done and steps < MAX_STEPS:
            action = llm.decide(history, tools)
            result = execute_tool(action)
            history.append(result)
            steps += 1
    """
    if llm is None:
        llm = get_default_llm()

    run_id = str(uuid.uuid4())
    set_run_id(run_id)
    logger.info("agent_run_started", extra={"user_message": user_message})

    set_disable_random_failures(disable_random_failures)
    try:
        history: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        tool_calls_log: list[dict[str, Any]] = []
        order_cache: dict[str, Any] = {}
        steps = 0
        done = False
        response = ""

        while not done and steps < MAX_STEPS:
            action = llm.decide(history, TOOL_SCHEMAS)
            logger.info("llm_action", extra={"step": steps, "action_type": action["type"]})

            if action["type"] == "final_answer":
                response = action["content"]
                history.append({"role": "assistant", "content": response})
                done = True
                break

            if action["type"] != "tool_call":
                logger.warning("unknown_action_type", extra={"step": steps, "action_type": action.get("type")})
                response = "Unexpected response from model — escalating to human"
                done = True
                break

            history.append(
                {
                    "role": "tool_call",
                    "id": f"call_{steps}",
                    "name": action["name"],
                    "arguments": action["arguments"],
                }
            )

            tool_result = execute_tool(action, tool_calls_log, today=today, order_cache=order_cache)
            history.append(tool_result)
            steps += 1

        if not done and steps >= MAX_STEPS:
            ticket = escalate_to_human(None, "Max steps reached — agent loop exceeded limit")
            tool_calls_log.append(
                {
                    "name": "escalate_to_human",
                    "arguments": {"note": "Max steps reached"},
                    "result": ticket,
                }
            )
            response = "Max steps reached — escalating to human"
            logger.warning("max_steps_reached", extra={"steps": steps})

        logger.info(
            "agent_run_finished",
            extra={"steps": steps, "tool_call_count": len(tool_calls_log)},
        )
        return AgentRun(response=response, tool_calls=tool_calls_log, steps=steps, history=history)
    finally:
        set_disable_random_failures(False)  # always restore default
        clear_run_id()


# Convenience alias used by eval.py
def agent(user_message: str, llm: LLM | None = None, **kwargs: Any) -> AgentRun:
    return run(user_message, llm=llm, **kwargs)
