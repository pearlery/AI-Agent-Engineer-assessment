# Part C — Whiteboard Discussion Answers

## 1. "This now serves 500 reps and 10k requests/day. What breaks first?"

**LLM latency and cost** break first.

At 10k requests/day with a multi-step ReAct loop (3–8 LLM calls per request), you're looking at 30k–80k API calls daily. Two immediate fixes:

| Problem | Fix |
|---------|-----|
| `get_refund_policy` called on every refund request | **Cache it** — static content, TTL 24h or invalidate on policy change |
| Simple lookups ("status of jane@…") routed through full agent | **Route to a smaller/cheaper model** (e.g. gpt-4o-mini) or bypass agent entirely |
| Repeated `get_order` for same ID within a session | **Request-level cache** with 5-min TTL |
| Rate limits under burst traffic | **Queue + backpressure**, per-rep concurrency limits |

Secondary bottlenecks: observability storage (structured logs at 10k/day), and escalation queue depth if flaky tools spike.

---

## 2. "How would you measure whether this agent is actually good?"

Track **outcome metrics** + **quality sampling**:

### Outcome metrics (automated, daily)
- **Refund error rate** — refunds issued on ineligible orders (should be ~0% with guardrails)
- **Escalation rate** — % of runs ending in `escalate_to_human` (baseline then alert on spikes)
- **Human override rate** — % of agent actions reversed by a human within 24h
- **Avg steps per run** — creeping upward signals prompt drift or new edge cases

### Quality sampling
- **LLM-as-judge** on 5% of runs — score: correct tools called, policy followed, tone appropriate
- **Regression eval suite** (`eval.py`) run on every deploy
- **Weekly human review** of 20 random runs, focused on refund decisions

### Alerting
- Refund error rate > 0.1% → page on-call
- Escalation rate > 2× baseline → investigate tool health or prompt regression

---

## 3. "A refund was issued that shouldn't have been. Walk me through debugging it."

```
1. Get confirmation number from the bad refund
        ↓
2. Search structured logs by confirmation_number / order_id / timestamp
        ↓
3. Reconstruct the full run: user message → each LLM step → tool calls → guardrail results
        ↓
4. Find the step where issue_refund was called
        ↓
5. Check: did validate_refund() run? What order data did get_order return?
        ↓
6. Classify root cause:
   • Policy fault  — guardrail logic wrong (e.g. timezone bug in 30-day check)
   • Prompt fault  — LLM skipped get_refund_policy; guardrail didn't catch it
   • Tool fault    — get_order returned wrong refundable/damaged flags
   • Data fault    — seed/production data out of sync
        ↓
7. Reproduce with eval.py test case, fix, deploy, verify metric drops
```

Key log fields to capture per step: `run_id`, `step`, `tool_name`, `arguments`, `result`, `guardrail_outcome`, `llm_model`, `latency_ms`.

---

## 4. "When would you NOT use an agent here?"

**Don't use an agent when the logic is fully deterministic:**

| Task | Better approach |
|------|----------------|
| Order status lookup by email | Direct API call + formatted response — no LLM needed |
| Eligibility check | `guardrails.py` rules — already coded, no inference |
| Issuing refund after eligibility confirmed | Workflow engine with explicit approval step |

**Use an agent when:**
- User intent is ambiguous natural language ("they're really mad")
- Multi-step reasoning across tools is needed
- The rep's request doesn't map 1:1 to a single API call

**Rule of thumb:** If you can write a `if/else` that covers 95% of cases correctly, use deterministic code. Reserve the agent for the 5% that need language understanding and multi-step orchestration — and keep guardrails around anything irreversible.
