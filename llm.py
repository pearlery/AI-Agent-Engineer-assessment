"""LLM adapters — OpenAI for production, ScriptedLLM for deterministic eval."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any


class LLM(ABC):
    @abstractmethod
    def decide(self, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Return one of:
          {"type": "tool_call", "name": str, "arguments": dict}
          {"type": "final_answer", "content": str}
        """


class OpenAILLM(LLM):
    """Real LLM via OpenAI function-calling."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def decide(self, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        messages = _history_to_openai_messages(history)
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if message.tool_calls:
            call = message.tool_calls[0]
            return {
                "type": "tool_call",
                "name": call.function.name,
                "arguments": json.loads(call.function.arguments),
            }

        return {"type": "final_answer", "content": message.content or ""}


class ScriptedLLM(LLM):
    """
    Deterministic LLM for eval — replays a scripted sequence of tool calls
    then returns a final answer.
    """

    def __init__(self, script: list[dict[str, Any]], final_answer: str = "Done."):
        self.script = list(script)
        self.final_answer = final_answer
        self._step = 0

    def decide(self, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if self._step < len(self.script):
            action = self.script[self._step]
            self._step += 1
            return {"type": "tool_call", **action}
        return {"type": "final_answer", "content": self.final_answer}


def _history_to_openai_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an internal e-commerce support agent. "
                "Use tools to look up orders, check refund policy before refunding, "
                "and never issue a refund without verifying eligibility. "
                "If a request is ambiguous (no order ID), ask a clarifying question "
                "or escalate to a human — do not guess."
            ),
        }
    ]

    for entry in history:
        role = entry.get("role")
        if role == "user":
            messages.append({"role": "user", "content": entry["content"]})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": entry.get("content", "")})
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": entry.get("tool_call_id", "call_0"),
                    "content": json.dumps(entry["result"]),
                }
            )
        elif role == "tool_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": entry.get("id", "call_0"),
                            "type": "function",
                            "function": {
                                "name": entry["name"],
                                "arguments": json.dumps(entry["arguments"]),
                            },
                        }
                    ],
                }
            )

    return messages


def extract_email(text: str) -> str | None:
    match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
    return match.group(0) if match else None


def extract_order_id(text: str) -> str | None:
    match = re.search(r"#?(\d{4,})", text)
    return match.group(1) if match else None
