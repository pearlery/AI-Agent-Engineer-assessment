# Part C — Whiteboard Discussion

## 1. "This now serves 500 reps and 10k requests/day. What breaks first?"

**Rate limits break first — within the first business-hour spike.**

### Back-of-envelope

| Metric | Value |
|--------|-------|
| 10k req/day evenly spread | ~7 req/min average |
| 500 reps, morning rush 8–9am | ~500 req in 60 min = **8 req/min peak** |
| Each request → avg 4 LLM calls (ReAct loop) | **32 LLM calls/min at peak** |
| Gemini 2.5 Flash free tier | 20 req/min — **already blown** |
| Gemini 2.5 Flash paid tier | 2,000 req/min — headroom until ~15k reps |

### Fix sequence (in priority order)

**1. Cache `get_refund_policy()`** — today it's called on every refund request (1 extra LLM round-trip). It's static content. One `functools.lru_cache(maxsize=1, ttl=3600)` call saves ~30% of LLM calls with zero code risk.

**2. Rate-limit with exponential backoff** — already partially covered by the `tenacity` retry in `google-genai`, but we need per-rep concurrency caps (max 2 concurrent runs per rep) to prevent a single power user starving others.

**3. Model routing** — simple status lookups ("what are my orders?") don't need a flagship model. Route by intent:
```
intent = classify(user_message)  # tiny local classifier or regex
if intent == "lookup":    use gemini-2.0-flash-lite  # ~10x cheaper
if intent == "refund":    use gemini-2.5-flash        # needs reasoning
```
Estimated 40–50% cost reduction at 10k req/day.

**4. Request-level `order_cache`** — already implemented. Prevents double `get_order` calls within a single run.

**5. Token budget ceiling** — already implemented (`MAX_INPUT_TOKENS = 100_000`). Prevents runaway loops from burning quota on a single bad request.

### Cost estimate at 10k req/day
- Avg 4 LLM calls × avg 800 input tokens = 32M input tokens/day
- Gemini 2.5 Flash: $0.15/1M tokens → **~$5/day** — acceptable
- With model routing: **~$2–3/day**

---

## 2. "How would you measure whether this agent is actually good in production?"

Three layers: automated signals (daily), sampled quality (weekly), regression gate (per deploy).

### Automated signals

| Metric | Target | Alert threshold |
|--------|--------|----------------|
| **Refund error rate** — refunds on ineligible orders / total refunds | 0% (guardrails make this hard) | Any > 0 pages on-call |
| **Guardrail block rate** — `refund_not_eligible` errors / refund attempts | Baseline ~15% | 2× baseline = prompt regression |
| **Escalation rate** — runs ending in `escalate_to_human` | Baseline ~5% | > 3× baseline = tool health issue |
| **Avg steps per run** (p50/p95) | p50 ≤ 3, p95 ≤ 7 | p95 > 9 = approaching MAX_STEPS |
| **get_order timeout rate** | ~20% by design | > 40% = backend degraded |
| **Token cost per run** | < 5k input tokens | p99 > 20k = runaway loop |

### Sampled quality review

- **LLM-as-judge on 5% of runs**: score each run on (a) correct tool sequence, (b) policy followed, (c) tone appropriate. Store score in structured log. Alert if 7-day rolling average drops > 10%.
- **Human override rate**: refunds reversed by a supervisor within 24h — lagging but gold-standard signal.

### Regression gate (per deploy)

`eval.py` runs in CI on every push — 32 deterministic tests with `ScriptedLLM`, zero API calls, < 1 second. If any test fails, deploy is blocked.

---

## 3. "A refund was issued that shouldn't have been. Walk me through debugging it."

### Step-by-step using structured logs

```
1. Get the bad confirmation number (e.g. REF-1001)
         ↓
2. grep logs for "issue_refund" where result.confirmation_number = REF-1001
   → get run_id (e.g. "2a0d9619-ee0b-413c-8b9e-46f6387b4224")
         ↓
3. grep all logs for that run_id, sort by timestamp
   → full trace: agent_run_started → llm_action steps → tool_calls → agent_run_finished
         ↓
4. Find the issue_refund tool_call entry
   → check: what order_id, amount, reason were passed?
         ↓
5. Find the preceding get_order call for that order_id
   → check: what did delivered_date, refundable, damaged show?
         ↓
6. Classify root cause:
```

| Root cause | What to look for in logs |
|-----------|--------------------------|
| **Policy fault** | `validate_refund()` returned True but shouldn't — e.g. timezone bug in 30-day calc |
| **Prompt fault** | `get_refund_policy` was never called in the run — LLM skipped the step |
| **Tool fault** | `get_order` returned wrong `refundable=True` or `delivered_date` |
| **Data fault** | Seed/DB data was incorrect at time of run |
| **Guardrail bypass** | Would appear as `validate_refund()` not in the call chain — should be impossible by code design |

```
7. Reproduce with eval.py: add a ScriptedLLM test that replays the exact sequence
         ↓
8. Fix, add the test to the suite, deploy, verify refund-error-rate metric drops
```

Key log fields that make this tractable: `run_id`, `step`, `tool`, `tool_args`, `result`, `order_id`, `reason` (in guardrail block log).

---

## 4. "When would you NOT use an agent here, and just write deterministic code?"

**When the logic has no ambiguity and the input is structured.**

| Request | Agent? | Better approach |
|---------|--------|-----------------|
| "Status of order #1042" | ❌ | Direct DB lookup + template string — 0 LLM calls |
| "Is order #1042 refundable?" | ❌ | `validate_refund()` already does this deterministically |
| Bulk refund a CSV of order IDs | ❌ | Script with guardrail checks, no NL needed |
| "Refund my last 3 orders for john@example.com" | ✅ | Requires NL understanding + multi-step reasoning |
| "They're really mad, sort it out" | ✅ | Ambiguity resolution requires language understanding |
| Nightly SLA report | ❌ | SQL query + email template |

**Rule of thumb:** if you can write an `if/else` tree that covers 95% of cases correctly and the inputs are structured, use deterministic code. The agent earns its cost only for the cases that require understanding ambiguous natural language and chaining multiple steps whose order isn't known in advance.

**The real maturity signal:** the guardrails in `guardrails.py` exist precisely because some logic — like refund eligibility — must NEVER be delegated to the LLM. The agent handles the "what does the user want?" question; deterministic code handles "is this allowed?"
