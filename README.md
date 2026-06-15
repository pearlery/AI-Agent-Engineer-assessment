# E-Commerce Internal Support Agent

AI Agent Engineer Technical Assessment — production-grade ReAct support agent for resolving e-commerce support requests.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-...   # primary LLM (Qwen3-235B via OpenRouter)
GEMINI_API_KEY=...                  # fallback if OpenRouter key is absent
```

The agent auto-selects the LLM: OpenRouter when `OPENROUTER_API_KEY` is set, Gemini otherwise.

---

## Running

### Interactive mode

```bash
python main.py
```

### One-shot mode

```bash
python main.py "Refund the damaged item in order #1042."
python main.py "jane@example.com wants refunds on all eligible orders." --json
```

### Eval suite (no API calls required)

```bash
python -m pytest eval.py -v
```

---

## Agent Architecture

This agent implements the **ReAct loop** (Reason → Act → Observe) without any agent framework — the loop is written directly in `agent.py`.

```
User message
     │
     ▼
┌─────────────────────────────────────────┐
│              ReAct Loop                 │
│                                         │
│  1. llm.decide(history, tools)          │
│       → tool_call  OR  final_answer     │
│                                         │
│  2. execute_tool(action)                │
│       → guardrail check (issue_refund)  │
│       → retry logic (get_order x3)      │
│       → order cache (dedup calls)       │
│                                         │
│  3. append result to history            │
│  4. repeat until done or MAX_STEPS=10   │
└─────────────────────────────────────────┘
     │
     ▼
AgentRun(response, tool_calls, steps, tokens)
```

### Tools available to the agent

| Tool | Purpose |
|---|---|
| `search_orders` | Look up orders by customer email |
| `get_order` | Fetch order details by ID |
| `get_refund_policy` | Retrieve current refund policy |
| `issue_refund` | Issue a refund (irreversible — guarded) |
| `escalate_to_human` | Create a support ticket for human review |

### Robustness features

| Feature | Implementation |
|---|---|
| **Guardrails** | `validate_refund()` in `guardrails.py` — code-level, not bypassable by LLM |
| **Retry logic** | `get_order_with_retry()` — 3 attempts, escalates on persistent timeout |
| **Loop protection** | `MAX_STEPS = 10` hard cap + escalate |
| **Token budget** | `MAX_INPUT_TOKENS = 100,000` ceiling + escalate |
| **Bad model output** | `_coerce_args()` fixes int order_id, string amount |
| **Order cache** | Per-run dict prevents duplicate `get_order` API calls |
| **Dynamic date** | Today's date injected into system prompt at runtime |
| **Structured logs** | JSON lines to stderr with `run_id` per run |
| **Deterministic eval** | `ScriptedLLM` replays fixed tool sequences — zero API calls |

---

## LLM Provider

**Primary:** [Qwen3-235B-A22B-Instruct-2507](https://openrouter.ai/qwen/qwen3-235b-a22b-2507) via [OpenRouter](https://openrouter.ai)

- OpenAI-compatible API endpoint
- 262K context window
- $0.09/M input · $0.10/M output
- Strong instruction following, tool use, and multi-step reasoning

**Fallback:** Gemini 2.5 Flash (`gemini-2.5-flash`) via Google GenAI SDK

### LLM adapter design

```
LLM (ABC)
├── OpenRouterLLM   ← default (uses openai SDK, base_url=openrouter.ai)
├── GeminiLLM       ← fallback (uses google-genai SDK)
└── ScriptedLLM     ← eval only (deterministic, no API)
```

Each adapter returns a uniform `{"type": "tool_call"|"final_answer", ...}` dict — the agent loop is provider-agnostic.

---

## Project Structure

```
agent.py          # ReAct loop, retry logic, token tracking
llm.py            # LLM adapters (OpenRouter, Gemini, Scripted)
tools.py          # Tool implementations + TOOL_SCHEMAS
guardrails.py     # Refund eligibility validation
eval.py           # 10 deterministic tests (ScriptedLLM)
main.py           # CLI entry point (interactive + one-shot)
logging_config.py # JSON structured logging with run_id
PART_C.md         # Production design discussion
```

## Seed data

| Customer | Orders | Notes |
|---|---|---|
| `jane@example.com` | #1042, #1055, #1018, #1077 | #1042 damaged, #1018 >30 days old |
| `john@example.com` | #1071, #1060, #1038 | Mixed eligibility |
| `angry@example.com` | #1099 | Upset-customer scenario |
| `nobody@example.com` | — | Empty order list |

## Part C

See [PART_C.md](PART_C.md) for production design discussion.
