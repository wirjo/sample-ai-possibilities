# Claude Advisor Tool — Workaround for Amazon Bedrock

The [Claude Advisor Tool](https://docs.anthropic.com/en/docs/build-with-claude/advisor) pairs Opus (advisor) with Sonnet (executor) in a single API call. It's available on the Anthropic API and Claude Platform on AWS, but **not yet on Amazon Bedrock**.

This notebook implements the same pattern client-side using the Bedrock Converse API.

## Quick Start

```bash
pip install boto3
jupyter notebook advisor_tool_workaround.ipynb
```

## What's Inside

The notebook walks through:
1. **Why server-side (native) is better** — and why we still need this workaround
2. **Defining the advisor tool** for the executor to call
3. **The orchestration loop** — intercepting advisor calls and routing to Opus
4. **Running it** on a real task
5. **Cost breakdown** — showing the savings vs all-Opus

## Prerequisites

- AWS credentials with access to Claude Sonnet 4.6 and Opus 4.7 on Bedrock
- Python 3.9+
- `boto3`
