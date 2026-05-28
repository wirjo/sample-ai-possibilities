"""
Claude Advisor Tool Workaround for Amazon Bedrock

Implements the advisor-executor pattern (similar to Anthropic's advisor tool)
using client-side orchestration with Amazon Bedrock's Converse API.

The pattern:
- Executor (fast model like Sonnet) handles the main generation
- Advisor (strong model like Opus) is consulted at key decision points
- Client orchestrates the two-model conversation loop
"""

import os
import json
import boto3
from typing import Optional


# Default model configuration
DEFAULT_EXECUTOR_MODEL = "us.anthropic.claude-sonnet-4-6-v1"
DEFAULT_ADVISOR_MODEL = "us.anthropic.claude-opus-4-7-v1"
DEFAULT_REGION = "us-east-1"

# The tool definition that the executor can call to consult the advisor
ADVISOR_TOOL_DEFINITION = {
    "toolSpec": {
        "name": "consult_advisor",
        "description": (
            "Consult a senior advisor for guidance on complex decisions, "
            "planning, architecture, or when you're stuck. The advisor sees "
            "the full conversation context and provides strategic guidance. "
            "Call this: (1) before substantive work, (2) when stuck or an "
            "approach isn't converging, (3) before declaring done for a final "
            "review, (4) when your evidence contradicts prior advice."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "What specific guidance do you need from the advisor? "
                            "Be specific about the decision point or challenge."
                        ),
                    }
                },
                "required": ["question"],
            }
        },
    }
}


class AdvisorWorkAround:
    """
    Implements the Claude Advisor pattern on Amazon Bedrock.

    Pairs a fast executor model with a strong advisor model,
    orchestrating the conversation client-side to replicate
    the behavior of Anthropic's native advisor tool.
    """

    def __init__(
        self,
        executor_model_id: Optional[str] = None,
        advisor_model_id: Optional[str] = None,
        region: Optional[str] = None,
        max_advisor_calls: int = 3,
        executor_system_prompt: Optional[str] = None,
        advisor_system_prompt: Optional[str] = None,
    ):
        self.executor_model_id = (
            executor_model_id
            or os.environ.get("EXECUTOR_MODEL_ID", DEFAULT_EXECUTOR_MODEL)
        )
        self.advisor_model_id = (
            advisor_model_id
            or os.environ.get("ADVISOR_MODEL_ID", DEFAULT_ADVISOR_MODEL)
        )
        self.region = region or os.environ.get("AWS_REGION", DEFAULT_REGION)
        self.max_advisor_calls = max_advisor_calls

        self.client = boto3.client("bedrock-runtime", region_name=self.region)

        self.executor_system_prompt = executor_system_prompt or (
            "You are a capable AI assistant. You have access to a senior advisor "
            "that you can consult for complex decisions. Use the consult_advisor "
            "tool when you need strategic guidance, are stuck, or want a review "
            "before finalizing important work. Don't consult for trivial questions."
        )

        self.advisor_system_prompt = advisor_system_prompt or (
            "You are a senior advisor reviewing work done by a junior executor. "
            "Provide concise, actionable guidance. Focus on: correctness, approach "
            "quality, edge cases missed, and strategic direction. Be direct and "
            "specific. Keep responses under 500 words."
        )

        # Usage tracking
        self.usage = {
            "executor": {"input_tokens": 0, "output_tokens": 0},
            "advisor": {"input_tokens": 0, "output_tokens": 0},
            "advisor_calls": 0,
        }

    def _call_executor(
        self,
        messages: list,
        tools: Optional[list] = None,
    ) -> dict:
        """Call the executor model via Bedrock Converse."""
        kwargs = {
            "modelId": self.executor_model_id,
            "messages": messages,
            "system": [{"text": self.executor_system_prompt}],
            "inferenceConfig": {"maxTokens": 4096},
        }
        if tools:
            kwargs["toolConfig"] = {"tools": tools}

        response = self.client.converse(**kwargs)

        # Track usage
        usage = response.get("usage", {})
        self.usage["executor"]["input_tokens"] += usage.get("inputTokens", 0)
        self.usage["executor"]["output_tokens"] += usage.get("outputTokens", 0)

        return response

    def _call_advisor(self, messages: list, question: str) -> str:
        """
        Call the advisor model with the full conversation context.

        The advisor sees:
        1. The full conversation history (messages)
        2. The specific question from the executor
        """
        # Build the advisor's view: full transcript + the executor's question
        advisor_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            "Here is the full conversation transcript so far:\n\n"
                            f"{json.dumps(messages, indent=2, default=str)}\n\n"
                            "---\n\n"
                            f"The executor is asking for your guidance:\n{question}"
                        )
                    }
                ],
            }
        ]

        response = self.client.converse(
            modelId=self.advisor_model_id,
            messages=advisor_messages,
            system=[{"text": self.advisor_system_prompt}],
            inferenceConfig={"maxTokens": 1024},
        )

        # Track usage
        usage = response.get("usage", {})
        self.usage["advisor"]["input_tokens"] += usage.get("inputTokens", 0)
        self.usage["advisor"]["output_tokens"] += usage.get("outputTokens", 0)
        self.usage["advisor_calls"] += 1

        # Extract advisor response text
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        advisor_text = ""
        for block in content:
            if "text" in block:
                advisor_text += block["text"]

        return advisor_text

    def run(self, user_message: str, additional_tools: Optional[list] = None) -> dict:
        """
        Run the advisor-executor pattern for a given user message.

        Args:
            user_message: The user's request
            additional_tools: Optional extra tools for the executor (beyond advisor)

        Returns:
            dict with 'response' (final text), 'usage' (token counts), 'advisor_calls' (count)
        """
        # Reset usage tracking
        self.usage = {
            "executor": {"input_tokens": 0, "output_tokens": 0},
            "advisor": {"input_tokens": 0, "output_tokens": 0},
            "advisor_calls": 0,
        }

        # Build tool list: advisor + any additional tools
        tools = [ADVISOR_TOOL_DEFINITION]
        if additional_tools:
            tools.extend(additional_tools)

        # Initialize conversation
        messages = [{"role": "user", "content": [{"text": user_message}]}]

        # Agentic loop: executor generates, we intercept advisor calls
        while True:
            response = self._call_executor(messages, tools=tools)

            stop_reason = response.get("stopReason", "")
            output = response.get("output", {})
            assistant_message = output.get("message", {})

            # Add assistant response to conversation
            messages.append(assistant_message)

            # If the model stopped naturally (not a tool call), we're done
            if stop_reason == "end_turn":
                break

            # If the model wants to use a tool
            if stop_reason == "tool_use":
                content = assistant_message.get("content", [])
                tool_results = []

                for block in content:
                    if "toolUse" in block:
                        tool_use = block["toolUse"]
                        tool_name = tool_use.get("name", "")
                        tool_use_id = tool_use.get("toolUseId", "")
                        tool_input = tool_use.get("input", {})

                        if tool_name == "consult_advisor":
                            # Check advisor call limit
                            if self.usage["advisor_calls"] >= self.max_advisor_calls:
                                advisor_response = (
                                    "[Advisor call limit reached. "
                                    "Proceed with your best judgment.]"
                                )
                            else:
                                question = tool_input.get("question", "")
                                advisor_response = self._call_advisor(
                                    messages, question
                                )

                            tool_results.append(
                                {
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": advisor_response}],
                                    }
                                }
                            )
                        else:
                            # For other tools, return a placeholder
                            # (in real usage, you'd handle these)
                            tool_results.append(
                                {
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [
                                            {
                                                "text": (
                                                    f"[Tool '{tool_name}' not "
                                                    "implemented in this example]"
                                                )
                                            }
                                        ],
                                    }
                                }
                            )

                # Add tool results to conversation
                messages.append({"role": "user", "content": tool_results})
            else:
                # Unknown stop reason, break
                break

        # Extract final response text
        final_text = ""
        final_content = assistant_message.get("content", [])
        for block in final_content:
            if "text" in block:
                final_text += block["text"]

        return {
            "response": final_text,
            "usage": self.usage,
            "advisor_calls": self.usage["advisor_calls"],
            "messages": messages,  # Full conversation history
        }

    def print_usage(self):
        """Print a formatted usage summary."""
        print("\n─── usage ────────────────────────────────────────────")
        print(
            f"executor  in={self.usage['executor']['input_tokens']} "
            f"out={self.usage['executor']['output_tokens']}"
        )
        print(
            f"advisor   in={self.usage['advisor']['input_tokens']} "
            f"out={self.usage['advisor']['output_tokens']} "
            f"calls={self.usage['advisor_calls']}"
        )
        print("──────────────────────────────────────────────────────\n")
