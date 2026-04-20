"""
Pricing & Token Tracking
-------------------------
Provides cost calculation for Anthropic Claude models and a
TokenTracker dataclass that encapsulates all token/cache accounting
for both main API calls and MCP sampling calls.

Usage:
    from pricing import TokenTracker, calculate_cost

    tracker = TokenTracker()
    tracker.accumulate_usage(anthropic_response)
    summary = tracker.build_summary(model_id="claude-haiku-4-5")
    # summary == {"token_usage": {...}, "cost_usd": {...}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


# ─────────────────────────────────────────────────────────────────────
# Model Pricing (USD per 1 million tokens)
# ─────────────────────────────────────────────────────────────────────

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Claude 4.5 Sonnet
    "claude-4-5-sonnet-20250110": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-4-5-sonnet-latest": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    # Claude Sonnet 4
    "claude-sonnet-4": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-sonnet-4-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    # Claude Haiku 4.5
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
}

# Default model used when a model_id is not found in MODEL_PRICING.
DEFAULT_MODEL = "claude-haiku-4-5"


def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """
    Calculate the cost in USD for the given token usage.

    Args:
        model_id: The Anthropic model identifier (e.g. "claude-haiku-4-5").
        input_tokens: Number of non-cached input tokens.
        output_tokens: Number of output tokens.
        cache_creation_tokens: Tokens written to the prompt cache.
        cache_read_tokens: Tokens read from the prompt cache.

    Returns:
        Total cost in USD as a float.
    """
    pricing = MODEL_PRICING.get(model_id, MODEL_PRICING[DEFAULT_MODEL])

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    cache_write_cost = (cache_creation_tokens / 1_000_000) * pricing["cache_write"]
    cache_read_cost = (cache_read_tokens / 1_000_000) * pricing["cache_read"]

    return input_cost + output_cost + cache_write_cost + cache_read_cost


# ─────────────────────────────────────────────────────────────────────
# Token Tracker
# ─────────────────────────────────────────────────────────────────────


@dataclass
class TokenTracker:
    """
    Accumulates token usage across multiple Anthropic API calls.

    Tracks both *main* (agentic-loop) calls and *sampling* calls
    (triggered by the MCP server) separately, so you can attribute
    cost to each channel.

    Typical flow:
        tracker = TokenTracker()
        tracker.accumulate_usage(response)          # main API call
        tracker.accumulate_sampling_usage(response)  # sampling call
        summary = tracker.build_summary("claude-haiku-4-5")
    """

    # ── Totals (main + sampling combined) ──
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0

    # ── Sampling-only counters ──
    sampling_input_tokens: int = 0
    sampling_output_tokens: int = 0
    sampling_cache_creation_tokens: int = 0
    sampling_cache_read_tokens: int = 0
    sampling_call_count: int = 0

    def accumulate_usage(self, response: Any) -> None:
        """
        Extract token counts from an Anthropic Messages API response
        and add them to the **total** counters.

        Safe to call even if `response` has no `usage` attribute.
        """
        usage = getattr(response, "usage", None)
        if not usage:
            return

        self.total_input_tokens += getattr(usage, "input_tokens", 0)
        self.total_output_tokens += getattr(usage, "output_tokens", 0)

        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        self.total_cache_creation_tokens += cache_creation
        self.total_cache_read_tokens += cache_read

    def accumulate_sampling_usage(self, response: Any) -> None:
        """
        Extract token counts from an Anthropic Messages API response
        and add them to **both** the total and sampling counters.

        Also increments `sampling_call_count`.
        """
        usage = getattr(response, "usage", None)
        if not usage:
            self.sampling_call_count += 1
            return

        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        # Add to totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_creation_tokens += cache_creation
        self.total_cache_read_tokens += cache_read

        # Add to sampling-specific
        self.sampling_input_tokens += input_tokens
        self.sampling_output_tokens += output_tokens
        self.sampling_cache_creation_tokens += cache_creation
        self.sampling_cache_read_tokens += cache_read

        self.sampling_call_count += 1

    # ── Derived counters ──

    @property
    def main_input_tokens(self) -> int:
        """Input tokens from main API calls only (excludes sampling)."""
        return self.total_input_tokens - self.sampling_input_tokens

    @property
    def main_output_tokens(self) -> int:
        """Output tokens from main API calls only (excludes sampling)."""
        return self.total_output_tokens - self.sampling_output_tokens

    @property
    def main_cache_creation_tokens(self) -> int:
        """Cache creation tokens from main API calls only."""
        return self.total_cache_creation_tokens - self.sampling_cache_creation_tokens

    @property
    def main_cache_read_tokens(self) -> int:
        """Cache read tokens from main API calls only."""
        return self.total_cache_read_tokens - self.sampling_cache_read_tokens

    def build_summary(self, model_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Build a complete summary dict with token_usage and cost_usd.

        Args:
            model_id: The model identifier used for pricing lookup.

        Returns:
            Dict with keys "token_usage" and "cost_usd", each containing
            detailed breakdowns suitable for logging or JSON serialization.
        """
        total_cost = calculate_cost(
            model_id,
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cache_creation_tokens,
            self.total_cache_read_tokens,
        )
        sampling_cost = calculate_cost(
            model_id,
            self.sampling_input_tokens,
            self.sampling_output_tokens,
            self.sampling_cache_creation_tokens,
            self.sampling_cache_read_tokens,
        )
        main_api_cost = calculate_cost(
            model_id,
            self.main_input_tokens,
            self.main_output_tokens,
            self.main_cache_creation_tokens,
            self.main_cache_read_tokens,
        )

        return {
            "token_usage": {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "sampling_input_tokens": self.sampling_input_tokens,
                "sampling_output_tokens": self.sampling_output_tokens,
                "main_api_input_tokens": self.main_input_tokens,
                "main_api_output_tokens": self.main_output_tokens,
                "total_cache_creation_tokens": self.total_cache_creation_tokens,
                "total_cache_read_tokens": self.total_cache_read_tokens,
                "sampling_cache_creation_tokens": self.sampling_cache_creation_tokens,
                "sampling_cache_read_tokens": self.sampling_cache_read_tokens,
                "main_api_cache_creation_tokens": self.main_cache_creation_tokens,
                "main_api_cache_read_tokens": self.main_cache_read_tokens,
            },
            "cost_usd": {
                "total_cost": round(total_cost, 6),
                "sampling_cost": round(sampling_cost, 6),
                "main_api_cost": round(main_api_cost, 6),
            },
        }
