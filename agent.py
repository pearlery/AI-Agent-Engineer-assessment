"""ReAct agent loop with guardrails, retry logic, and observability."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from guardrails import RefundNotEligibleError, validate_refund
from llm import LLM, OpenAILLM, ScriptedLLM
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
    "Always check get_refund_policy before issuing a refund. "
    "Verify each order's eligibility with get_order before calling issue_refund. "
    "If the request is ambiguous (no order ID), ask for clarification or escalate — never guess."
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


def execute_tool(
    action: dict[str, Any],
    tool_calls_log: list[dict[str, Any]],
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """Execute a tool call with guardrails and logging."""
    name = action["name"]
    args = action.get("arguments", {})

    logger.info("tool_call", extra={"tool": name, "args": args})

    try:
        if name == "search_orders":
            result = search_orders(args["customer_email"])
        elif name == "get_order":
            result = get_order_with_retry(args["order_id"], tool_calls_log, today=today)
            return {"role": "tool", "name": name, "result": result}
        elif name == "get_refund_policy":
            result = get_refund_policy()
        elif name == "issue_refund":
            order = get_order_with_retry(args["order_id"], tool_calls_log, today=today)
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
    if disable_random_failures:
        set_disable_random_failures(True)

    if llm is None:
        llm = OpenAILLM()

    history: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    tool_calls_log: list[dict[str, Any]] = []
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

        history.append(
            {
                "role": "tool_call",
                "id": f"call_{steps}",
                "name": action["name"],
                "arguments": action["arguments"],
            }
        )

        tool_result = execute_tool(action, tool_calls_log, today=today)
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

    return AgentRun(response=response, tool_calls=tool_calls_log, steps=steps, history=history)


# Convenience alias used by eval.py
def agent(user_message: str, llm: LLM | None = None, **kwargs: Any) -> AgentRun:
    return run(user_message, llm=llm, **kwargs)
