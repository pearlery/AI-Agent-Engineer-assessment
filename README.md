# AI Agent Engineer Assessment

E-commerce internal support agent with ReAct loop, hard-coded guardrails, and eval harness.

## Quick start

```bash
pip install -r requirements.txt
python eval.py          # run all tests (no API key needed)
```

### Live agent (requires OpenAI API key)

```bash
set OPENAI_API_KEY=sk-...
python main.py "What's the status of orders for jane@example.com?"
```

## Project structure

| File | Purpose |
|------|---------|
| `tools.py` | Mock backend — 5 tools with seed data & 20% flaky `get_order` |
| `guardrails.py` | Hard refund eligibility checks (not bypassable by LLM) |
| `agent.py` | ReAct loop, retry logic, max-step protection |
| `llm.py` | OpenAI adapter + `ScriptedLLM` for deterministic eval |
| `eval.py` | 5 test cases + bonus guardrail/loop tests |
| `main.py` | CLI entry point for live runs |
| `PART_C.md` | Whiteboard discussion answers |

## Seed data

| Entity | Detail |
|--------|--------|
| `jane@example.com` | 2 orders (#1042 damaged, #1055 normal) |
| `john@example.com` | 3 orders — mixed eligibility for "refund last 3" test |
| `nobody@example.com` | 0 orders |
| Order #1038 | Delivered 45 days ago → outside 30-day window |
| Order #1042 | `damaged=True` → always refundable |
| Order #1060 | `refundable=False` → blocked by guardrail |

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
