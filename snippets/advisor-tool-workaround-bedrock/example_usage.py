"""
Example: Using the Advisor Tool Workaround on Amazon Bedrock

Demonstrates the advisor-executor pattern where Sonnet handles execution
and Opus provides strategic guidance when consulted.
"""

from advisor_workaround import AdvisorWorkAround


def main():
    # Initialize with defaults (Sonnet executor + Opus advisor)
    advisor = AdvisorWorkAround(
        executor_model_id="us.anthropic.claude-sonnet-4-6-v1",
        advisor_model_id="us.anthropic.claude-opus-4-7-v1",
        max_advisor_calls=3,
    )

    # Example: Complex architecture question where the executor
    # should consult the advisor for planning guidance
    task = """
    Design a serverless event-driven architecture for a real-time 
    fraud detection system that processes 10,000 transactions per second.
    
    Requirements:
    - Sub-100ms latency for scoring
    - ML model inference (XGBoost + neural network ensemble)
    - Real-time feature engineering from streaming data
    - Explainability for flagged transactions
    - 99.99% availability
    
    Provide the architecture with AWS services, data flow, 
    and key design decisions.
    """

    print("🧠 Running advisor-executor pattern...")
    print(f"   Executor: {advisor.executor_model_id}")
    print(f"   Advisor:  {advisor.advisor_model_id}")
    print(f"   Max advisor calls: {advisor.max_advisor_calls}")
    print("─" * 55)

    result = advisor.run(task)

    print("\n📋 Final Response:")
    print("─" * 55)
    print(result["response"])

    # Show usage breakdown
    advisor.print_usage()

    print(f"✅ Advisor was consulted {result['advisor_calls']} time(s)")


def example_with_custom_prompts():
    """Example with customized system prompts for a specific domain."""
    advisor = AdvisorWorkAround(
        executor_system_prompt=(
            "You are a Python developer. You have access to a senior architect "
            "advisor. Consult them before making design decisions or when you're "
            "unsure about the best approach. Write clean, production-ready code."
        ),
        advisor_system_prompt=(
            "You are a principal engineer reviewing code and architecture decisions. "
            "Focus on: security, scalability, error handling, and maintainability. "
            "Point out anti-patterns. Suggest better alternatives concisely."
        ),
        max_advisor_calls=2,
    )

    result = advisor.run(
        "Write a Python class for rate-limiting API calls using a token bucket "
        "algorithm. It needs to support both per-user and global limits, with "
        "Redis as the backing store for distributed deployments."
    )

    print(result["response"])
    advisor.print_usage()


if __name__ == "__main__":
    main()
