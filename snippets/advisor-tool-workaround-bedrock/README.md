# Claude Advisor Tool Workaround for Amazon Bedrock

## Problem

The [Claude Advisor Tool](https://docs.anthropic.com/en/docs/build-with-claude/advisor) is available on the Anthropic API (via `betas: ["advisor-tool-2026-03-01"]`) and Claude Platform on AWS, but **not** on Amazon Bedrock's Converse or InvokeModel APIs.

The advisor tool pairs a strong model (e.g. Opus 4.7) as a planning/review advisor with a fast model (e.g. Sonnet 4.6) as the executor. The executor decides when to consult the advisor; the advisor sees the full transcript and returns guidance; the executor continues with that context.

## Workaround Approach

This snippet implements the same **principal-agent pattern** using two Bedrock `converse()` calls orchestrated client-side:

1. **Executor (Sonnet)** generates a response with a custom `consult_advisor` tool available
2. When the executor calls `consult_advisor`, we intercept and route the full conversation to the **Advisor (Opus)**
3. The advisor's response is returned as a tool result to the executor
4. The executor continues generating, now informed by the advisor's guidance

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Client Orchestrator                │
│                                                     │
│  ┌───────────┐    consult_advisor()   ┌──────────┐  │
│  │  Executor │ ───────────────────▶   │ Advisor  │  │
│  │ (Sonnet)  │ ◀─────────────────── │ (Opus)   │  │
│  │           │    advisor guidance    │          │  │
│  └───────────┘                       └──────────┘  │
│       │                                             │
│       ▼                                             │
│  Final Response                                     │
└─────────────────────────────────────────────────────┘
```

### Key Differences from Native Advisor Tool

| Aspect | Native (Anthropic API) | Workaround (Bedrock) |
|--------|----------------------|---------------------|
| API calls | 1 request (server-side orchestration) | 2+ requests (client-side) |
| Advisor input | Server constructs from full transcript | Client passes full transcript |
| Executor awareness | Built-in `server_tool_use` block | Custom tool definition |
| Caching | `ephemeral` TTL on advisor | Prompt caching via Bedrock (if available) |
| Billing | Split in `usage.iterations[]` | Separate per-request billing |

## Files

- `advisor_workaround.py` — Main implementation with Bedrock Converse API
- `example_usage.py` — Demo showing the advisor pattern in action
- `requirements.txt` — Dependencies

## Usage

```bash
pip install -r requirements.txt
python example_usage.py
```

## Configuration

Set your preferred models via environment variables:

```bash
export ADVISOR_MODEL_ID="us.anthropic.claude-opus-4-7-v1"
export EXECUTOR_MODEL_ID="us.anthropic.claude-sonnet-4-6-v1"
export AWS_REGION="us-east-1"
```

## When to Consult the Advisor

Following Anthropic's own guidance for the native tool:

1. **Before substantive work** — Not for orientation, but before writing/answering
2. **When stuck** — Recurring errors, approach not converging
3. **Before declaring done** — Final review of the completed artifact
4. **On disagreement** — When evidence contradicts prior advisor guidance, surface the conflict

## Cost Optimization

- The advisor (Opus) is only called when the executor decides it needs help
- Typical advisor responses are 400-700 tokens
- All bulk generation stays at executor (Sonnet/Haiku) rates
- Add `max_advisor_calls` to cap costs per request
