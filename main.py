"""Interactive CLI — support rep types requests in plain English."""

from __future__ import annotations

import json
import logging
import sys

from dotenv import load_dotenv
load_dotenv()

from agent import agent
from llm import get_default_llm
from logging_config import setup_logging

logger = logging.getLogger(__name__)


def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "API rate limit reached. Please wait a moment and try again."
    if "503" in msg or "UNAVAILABLE" in msg:
        return "LLM service temporarily unavailable. Please try again shortly."
    if "401" in msg or "403" in msg or "API_KEY" in msg.upper():
        return "API key error. Please check your GEMINI_API_KEY."
    return msg.split("\n")[0]  # first line only for other errors


def run_once(message: str, *, show_json: bool = False) -> int:
    run = agent(message, llm=get_default_llm())

    if show_json:
        print(
            json.dumps(
                {
                    "response": run.response,
                    "steps": run.steps,
                    "tool_calls": run.tool_calls,
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(f"\nAgent> {run.response}")
        if run.tool_calls:
            print("\n--- Tool calls ---")
            for call in run.tool_calls:
                print(f"  {call['name']}({call.get('arguments', {})})")

    return 0


def interactive_loop() -> int:
    print("E-commerce Support Agent (Gemini 2.5 Flash)")
    print("Type a support request in plain English.")
    print("Commands: quit / exit / q to leave\n")

    while True:
        try:
            message = input("Rep> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not message:
            continue
        if message.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        logger.info("user_request", extra={"user_message": message})
        try:
            run_once(message)
        except Exception as exc:
            logger.exception("agent_error", extra={"error": str(exc)})
            friendly = _friendly_error(exc)
            print(f"\nAgent> Sorry, something went wrong: {friendly}")

    return 0


def main() -> int:
    setup_logging()

    # Optional one-shot mode for scripting: python main.py "message" [--json]
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        message = args[0]
        show_json = "--json" in args[1:]
        return run_once(message, show_json=show_json)

    return interactive_loop()


if __name__ == "__main__":
    sys.exit(main())
