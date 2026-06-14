# AI Agent Engineer Assessment

E-commerce internal support agent with ReAct loop, hard-coded guardrails, and eval harness.

## Seed data

| Entity | Orders | Notes |
|--------|--------|-------|
| `jane@example.com` | 4 (#1042, #1055, #1018, #1077) | #1042 damaged, #1018 >30 days |
| `john@example.com` | 3 (#1071, #1060, #1038) | Mixed eligibility for refund test |
| `angry@example.com` | 1 (#1099) | Upset-customer scenario |
| `nobody@example.com` | 0 | Explicit empty customer in `CUSTOMERS` |

## Quick start

```bash
pip install -r requirements.txt
python eval.py          # run all tests (no API key needed)
```

### Live agent (requires Gemini API key)

```bash
set GEMINI_API_KEY=your-key-here
python main.py          # interactive loop — rep types requests
python main.py "What's the status of orders for jane@example.com?"  # one-shot
```

Uses **Gemini 2.0 Flash** (`gemini-2.0-flash`). Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).
Also accepts `GOOGLE_API_KEY` as an alias.

JSON structured logs go to **stderr**; agent response goes to **stdout**.

## Project structure

| File | Purpose |
|------|---------|
| `tools.py` | Mock backend — 5 tools with seed data & 20% flaky `get_order` |
| `guardrails.py` | Hard refund eligibility checks (not bypassable by LLM) |
| `agent.py` | ReAct loop, retry logic, max-step protection |
| `llm.py` | Gemini 2.0 Flash adapter + `ScriptedLLM` for deterministic eval |
| `eval.py` | 5 test cases + bonus guardrail/loop tests |
| `main.py` | Interactive CLI (`input()` loop) + optional one-shot mode |
| `logging_config.py` | JSON structured logging with `run_id` per agent run |
| `PART_C.md` | Whiteboard discussion answers |

## Test cases (eval.py)

1. **Status lookup** — `jane@example.com` → calls `search_orders`, returns order list
2. **Damaged refund** — `#1042` → checks policy, issues exactly one refund
3. **Last 3 orders** — checks each order, only refunds eligible (#1071)
4. **Ambiguous request** — "they're really mad" → no `issue_refund`
5. **Timeout handling** — `get_order` fails 3× → escalates, no crash

## Architecture

```
User message
    ↓
ReAct loop (max 10 steps)
    ↓
LLM decides → tool_call or final_answer
    ↓
execute_tool()
    ├── get_order → retry 3× → escalate on persistent timeout
    └── issue_refund → validate_refund() guardrail (hard code)
```

## Part C

See [PART_C.md](PART_C.md) for production discussion answers.
