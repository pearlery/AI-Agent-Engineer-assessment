"""CLI entry point for live agent runs."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from agent import agent
from llm import OpenAILLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="E-commerce support agent")
    parser.add_argument("message", nargs="?", help="Support request in plain English")
    parser.add_argument("--json", action="store_true", help="Output full run as JSON")
    args = parser.parse_args()

    if not args.message:
        parser.print_help()
        return 1

    run = agent(args.message, llm=OpenAILLM())

    if args.json:
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
        print(run.response)
        if run.tool_calls:
            print("\n--- Tool calls ---")
            for call in run.tool_calls:
                print(f"  {call['name']}({call.get('arguments', {})})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
