"""List prices, for the calls whose route does not report what it charged.

A metered route (OpenRouter's `usage.cost`, the Codex CLI's per-turn cost)
tells you the real number and should always win. This table is the fallback,
and it is a *fallback*: it is a published rate card applied to token counts,
so it misses discounts, provider surcharges and anything the route does with
caching that the usage block did not break out. Every cost this module
produces is therefore stamped `cost_source="list_price"`, so a ledger can be
read without guessing which numbers were measured.

Prices are USD per 1,000,000 tokens, as OpenRouter lists them (the `price`
block of its API and `/api/v1/models`; `openai/gpt-5.6-sol` matches
`llm/codex_cli.py`'s `LIST_PRICES`).

A model with no published rate card here reports `cost_source="unpriced"`
rather than a price inferred from tokens and totals, which would be a guess
wearing a table's authority.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float = 0.0


LIST_PRICES_PER_1M: dict[str, Price] = {
    "anthropic/claude-opus-5": Price(5.0, 25.0, 0.5, 6.25),
    "anthropic/claude-sonnet-5": Price(2.0, 10.0, 0.2, 2.5),
    "anthropic/claude-haiku-4.5": Price(1.0, 5.0, 0.1, 1.25),
    "openai/gpt-5.6-luna": Price(0.2, 1.2, 0.02, 0.25),
    "openai/gpt-5.6-sol": Price(2.0, 10.0, 0.2, 0.0),
    # OpenRouter /api/v1/models: sol-pro and luna-pro price like their base
    # tiers; terra is 2 / 12 / 0.2.
    "openai/gpt-5.6-sol-pro": Price(2.0, 10.0, 0.2, 0.0),
    "openai/gpt-5.6-luna-pro": Price(0.2, 1.2, 0.02, 0.25),
    "openai/gpt-5.6-terra": Price(2.0, 12.0, 0.2, 0.0),
    "openai/gpt-5.6-terra-pro": Price(2.0, 12.0, 0.2, 0.0),
    "minimax/minimax-m3": Price(0.3, 1.2, 0.06, 0.0),
}

# What the Claude CLI is asked for, and what its `modelUsage` block answers
# with, are both short of the OpenRouter slug the table is keyed on.
_ALIASES: dict[str, str] = {
    "opus": "anthropic/claude-opus-5",
    "sonnet": "anthropic/claude-sonnet-5",
    "haiku": "anthropic/claude-haiku-4.5",
    "claude-opus-5": "anthropic/claude-opus-5",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
}


def price_for(model: str) -> Price | None:
    """The rate card for `model`, or None if this table does not have one."""

    name = (model or "").strip().lower()
    if not name:
        return None
    if name in LIST_PRICES_PER_1M:
        return LIST_PRICES_PER_1M[name]
    if name in _ALIASES:
        return LIST_PRICES_PER_1M[_ALIASES[name]]
    # A dated release id (`claude-sonnet-5-20260514`) prices as its family.
    trimmed = name.rsplit("-", 1)[0] if name.rsplit("-", 1)[-1].isdigit() else name
    if trimmed in _ALIASES:
        return LIST_PRICES_PER_1M[_ALIASES[trimmed]]
    return LIST_PRICES_PER_1M.get(f"anthropic/{trimmed}")


def list_price_usd(
    model: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """List-price cost of one call, or None when the model has no price.

    `prompt_tokens` is the whole prompt as the provider counts it, cached
    portions included — that is what both the OpenAI-compatible and the
    Anthropic usage blocks report — so the cached and cache-written parts are
    subtracted out and billed at their own rates. Billing the whole prompt at
    the write rate whenever a breakpoint was present is the specific mistake
    that put `run_judge_probe.py`'s first estimate 11% high.
    """

    price = price_for(model)
    if price is None:
        return None
    fresh = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    total = (
        fresh * price.input
        + cache_read_tokens * price.cache_read
        + cache_write_tokens * price.cache_write
        + completion_tokens * price.output
    ) / 1_000_000
    return round(total, 6)


def cost_from_list_price(
    model: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> tuple[float, str]:
    """`(cost, cost_source)` — the shape every client records.

    `("...", "unpriced")` rather than a silent zero when the model is absent
    from the table: 0.0 is a plausible cost and would be summed into a run's
    total as if it were one.
    """

    cost = list_price_usd(
        model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    if cost is None:
        logger.warning("no list price for %s; reporting cost 0.0 as unpriced", model)
        return 0.0, "unpriced"
    return cost, "list_price"
