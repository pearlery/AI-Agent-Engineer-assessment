"""LLM adapters — Gemini 2.0 Flash for production, ScriptedLLM for eval."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "You are an internal e-commerce support agent. "
    "Use tools to resolve support requests. "
    "If a customer email is mentioned but no order ID, call search_orders first to find their orders. "
    "Always check get_refund_policy before issuing any refund. "
    "Verify each order's eligibility with get_order before calling issue_refund. "
    "If the request mentions multiple orders, check each one individually with get_order. "
    "If the request is ambiguous (no email and no order ID), ask for clarification or escalate — never guess."
)


class LLM(ABC):
    @abstractmethod
    def decide(self, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Return one of:
          {"type": "tool_call", "name": str, "arguments": dict}
          {"type": "final_answer", "content": str}
        """


class GeminiLLM(LLM):
    """Gemini 2.0 Flash via google-genai function calling."""

    def __init__(self, model: str = GEMINI_MODEL, api_key: str | None = None):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("Install google-genai: pip install google-genai") from exc

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

        self._genai = genai
        self._types = types
        self.client = genai.Client(api_key=key)
        self.model = model

    def decide(self, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        system, contents = _history_to_gemini(history, self._types)
        gemini_tools = _tools_to_gemini(tools, self._types)

        config = self._types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=system or SYSTEM_INSTRUCTION,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return _parse_gemini_response(response)


class OpenAILLM(LLM):
    """Optional OpenAI adapter (fallback)."""

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


def get_default_llm() -> LLM:
    """Return the production LLM (Gemini 2.0 Flash)."""
    return GeminiLLM()


def _history_to_gemini(history: list[dict[str, Any]], types: Any) -> tuple[str | None, list[Any]]:
    system: str | None = None
    contents: list[Any] = []

    for entry in history:
        role = entry.get("role")
        if role == "system":
            system = entry["content"]
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=entry["content"])]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=entry.get("content", ""))]))
        elif role == "tool_call":
            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name=entry["name"],
                                args=entry["arguments"],
                            )
                        )
                    ],
                )
            )
        elif role == "tool":
            result = entry["result"]
            if not isinstance(result, dict):
                result = {"result": result}
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=entry["name"],
                                response=result,
                            )
                        )
                    ],
                )
            )

    return system, contents


def _tools_to_gemini(tools: list[dict[str, Any]], types: Any) -> list[Any]:
    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
        )
        for t in tools
    ]
    return [types.Tool(function_declarations=declarations)]


def _parse_gemini_response(response: Any) -> dict[str, Any]:
    if not response.candidates:
        return {"type": "final_answer", "content": ""}

    content = response.candidates[0].content
    if not content or not content.parts:
        return {"type": "final_answer", "content": ""}

    for part in content.parts:
        if part.function_call:
            args = part.function_call.args or {}
            return {
                "type": "tool_call",
                "name": part.function_call.name,
                "arguments": dict(args),
            }
        if part.text:
            return {"type": "final_answer", "content": part.text}

    return {"type": "final_answer", "content": ""}


def _history_to_openai_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

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
