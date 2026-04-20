"""
Test script for the MCP Template.

Simple end-to-end test that searches for information about Figma.

Usage:
    python tests/test.py
    python tests/test.py --question "What does Figma do?"
    python tests/test.py --max-loops 5
"""

import os
import sys
import json
import asyncio
import argparse

# Ensure the parent directory is on the path for relative imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from mcp_client import Chat


SAMPLE_QUESTION = (
    "Research the company Figma. "
    "What is the annual revenue of Figma?"
)


async def main(question: str = SAMPLE_QUESTION, max_loops: int = 4):
    """Run the MCP client with a sample company research query."""
    print("=" * 60)
    print("MCP Template Test - Company Research: Figma")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print(f"Max loops: {max_loops}")
    print("-" * 60)

    chat = Chat()
    result = await chat.run(
        question=question,
        max_loops=max_loops,
    )

    if not result:
        print("\n[FAIL] ERROR: No result returned from MCP client.")
        sys.exit(1)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Final answer
    if "message" in result:
        message = result["message"]
        if isinstance(message, dict):
            print(f"\nFinal Answer:")
            print(json.dumps(message, indent=2))
        else:
            print(f"\nFinal Answer: {message}")

    # Tools used
    if "tools_used" in result:
        print(f"\nTools used ({len(result['tools_used'])}):")
        for i, tool in enumerate(result["tools_used"], 1):
            print(f"  {i}. {tool}")

    # Log data
    if "log_data" in result:
        log = result["log_data"]
        print(f"\nStats:")
        print(f"  Loops used: {log.get('total_loops', '?')}/{log.get('max_loops', '?')}")
        print(f"  Total duration: {log.get('total_duration', 0):.2f}s")
        print(f"  LLM call duration: {log.get('total_duration_for_llm_calls', 0):.2f}s")
        print(f"  Sampling calls: {log.get('sampling_call_count', 0)}")
        print(f"  Model: {log.get('model_id', '?')}")

        if "token_usage" in log:
            tokens = log["token_usage"]
            print(f"\nToken Usage:")
            print(f"  Input:  {tokens.get('total_input_tokens', 0):,}")
            print(f"  Output: {tokens.get('total_output_tokens', 0):,}")
            print(f"  Total:  {tokens.get('total_tokens', 0):,}")

        if "cost_usd" in log:
            cost = log["cost_usd"]
            print(f"\nCost:")
            print(f"  Total:    ${cost.get('total_cost', 0):.6f}")
            print(f"  Main API: ${cost.get('main_api_cost', 0):.6f}")
            print(f"  Sampling: ${cost.get('sampling_cost', 0):.6f}")

        if "tool_durations" in log:
            print(f"\nTool Durations:")
            for tool, duration in log["tool_durations"].items():
                print(f"  {tool}: {duration:.2f}s")

    print("\n" + "=" * 60)
    print("[OK] Test completed successfully.")
    print("=" * 60)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the MCP Template with a company research query."
    )
    parser.add_argument(
        "--question", "-q", type=str, default=SAMPLE_QUESTION,
        help="The question to ask the MCP client.",
    )
    parser.add_argument(
        "--max-loops", type=int, default=4,
        help="Maximum number of agentic loops (default: 4).",
    )
    args = parser.parse_args()

    asyncio.run(main(question=args.question, max_loops=args.max_loops))
