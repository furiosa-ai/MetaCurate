"""Every client says what a call cost and where the number came from.

Per-role spend is one of the loop's reported metrics, and it can only be
summed if a 0.0 means "free" rather than "nobody knew". So each client fills
`cost_usd` and `cost_source`, and the source names the method: measured by the
route, reported by a harness that was itself the thing being billed, computed
from a rate card, or missing.

The Claude CLI case is the one with a trap in it. `total_cost_usd` is the
CLI's own accounting of an Anthropic bill. Point `ANTHROPIC_BASE_URL` at
OpenRouter — which is how an Opus proposer is billed through one account —
and the CLI keeps reporting a number computed from Anthropic's rate
card for a call Anthropic never billed.
"""

from __future__ import annotations

import json
import types
from unittest.mock import patch

import pytest

from meta_filter.llm.anthropic import AnthropicClient
from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient
from meta_filter.llm.mock import MockLLMClient
from meta_filter.llm.pricing import cost_from_list_price, list_price_usd, price_for

# --- the rate card ----------------------------------------------------------


def test_the_table_prices_the_models_the_runs_use() -> None:
    for model in (
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5.6-luna",
        "openai/gpt-5.6-sol",
        "minimax/minimax-m3",
    ):
        assert price_for(model) is not None


def test_cli_aliases_and_dated_ids_price_as_their_family() -> None:
    opus = price_for("anthropic/claude-opus-5")
    assert price_for("opus") == opus
    assert price_for("claude-haiku-4-5-20251001") == price_for("anthropic/claude-haiku-4.5")


def test_an_unknown_model_has_no_price() -> None:
    assert price_for("z-ai/glm-5.3") is None
    assert list_price_usd("z-ai/glm-5.3", prompt_tokens=100, completion_tokens=10) is None


def test_an_unpriced_model_reports_a_source_not_a_number() -> None:
    cost, source = cost_from_list_price("z-ai/glm-5.3", prompt_tokens=1000, completion_tokens=100)
    assert cost == 0.0
    assert source == "unpriced"


def test_output_and_cached_input_are_billed_at_their_own_rates() -> None:
    # Opus 5: $5 in, $25 out, $0.50 cache read, per 1M.
    plain = list_price_usd(
        "anthropic/claude-opus-5", prompt_tokens=1_000_000, completion_tokens=1_000_000
    )
    assert plain == pytest.approx(30.0)

    cached = list_price_usd(
        "anthropic/claude-opus-5",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_tokens=800_000,
    )
    assert cached == pytest.approx(0.2 * 5.0 + 0.8 * 0.5)


# --- the mock ---------------------------------------------------------------


def test_the_mock_costs_nothing_and_says_that_is_measured() -> None:
    resp = MockLLMClient().generate("META_FILTER::PROPOSER hello")
    assert resp.cost_usd == 0.0
    assert resp.cost_source == "mock"


# --- the Anthropic SDK client -----------------------------------------------


class _Message:
    def __init__(self, usage):
        self.content = [type("B", (), {"text": "hi"})()]
        self.usage = usage


def _anthropic_usage(**kwargs):
    defaults = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    defaults.update(kwargs)
    return type("U", (), defaults)()


def _run_anthropic(model: str, usage) -> object:
    client = AnthropicClient(model=model, api_key="k")
    messages = types.SimpleNamespace(create=lambda **kwargs: _Message(usage))
    fake_sdk = types.SimpleNamespace(
        Anthropic=lambda **kwargs: types.SimpleNamespace(messages=messages)
    )
    with patch.dict("sys.modules", {"anthropic": fake_sdk}):
        return client.generate("prompt")


def test_the_anthropic_client_prices_a_call_from_the_rate_card() -> None:
    resp = _run_anthropic("anthropic/claude-sonnet-5", _anthropic_usage())
    # Sonnet 5: $2/1M in, $10/1M out.
    assert resp.cost_usd == pytest.approx(1000 * 2e-6 + 100 * 10e-6)
    assert resp.cost_source == "list_price"


def test_the_anthropic_client_counts_cache_tokens_in_the_prompt() -> None:
    """Anthropic reports cache tokens outside `input_tokens`, so a client that
    ignores them undercounts both the prompt and the bill."""

    usage = _anthropic_usage(cache_read_input_tokens=4000, cache_creation_input_tokens=500)
    resp = _run_anthropic("anthropic/claude-sonnet-5", usage)
    assert resp.prompt_tokens == 1000 + 4000 + 500
    assert resp.cached_tokens == 4000
    assert resp.cost_usd == pytest.approx(1000 * 2e-6 + 4000 * 0.2e-6 + 500 * 2.5e-6 + 100 * 10e-6)


# --- the Claude CLI client --------------------------------------------------


def _cli_payload(**extra) -> dict:
    payload = {
        "type": "result",
        "is_error": False,
        "result": "hello",
        "usage": {"input_tokens": 1000, "output_tokens": 100},
        "total_cost_usd": 0.42,
    }
    payload.update(extra)
    return payload


def _run_cli(tmp_path, model: str, monkeypatch, base_url: str | None, **extra):
    if base_url is None:
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    client = ClaudeCodeHeadlessClient(model=model, scratch_dir=tmp_path)

    class _R:
        returncode = 0
        stdout = json.dumps(_cli_payload(**extra))
        stderr = ""

    with patch("meta_filter.llm.claude_code.subprocess.run", return_value=_R()):
        return client.generate("prompt")


def test_the_cli_cost_is_trusted_when_anthropic_is_the_biller(tmp_path, monkeypatch):
    resp = _run_cli(tmp_path, "opus", monkeypatch, None)
    assert resp.cost_usd == pytest.approx(0.42)
    assert resp.cost_source == "harness"


def test_an_anthropic_base_url_is_still_the_biller(tmp_path, monkeypatch):
    resp = _run_cli(tmp_path, "opus", monkeypatch, "https://api.anthropic.com")
    assert resp.cost_source == "harness"


def test_the_cli_cost_is_discarded_when_the_route_is_openrouter(tmp_path, monkeypatch):
    """Opus billed through OpenRouter."""

    resp = _run_cli(
        tmp_path,
        "anthropic/claude-opus-5",
        monkeypatch,
        "https://openrouter.ai/api",
        modelUsage={"anthropic/claude-opus-5": {}},
    )
    assert resp.cost_source == "list_price"
    assert resp.cost_usd == pytest.approx(1000 * 5e-6 + 100 * 25e-6)
    assert resp.cost_usd != pytest.approx(0.42)


def test_a_redirected_call_on_an_unpriced_model_reports_unpriced(tmp_path, monkeypatch):
    resp = _run_cli(
        tmp_path,
        "some/unknown-model",
        monkeypatch,
        "https://openrouter.ai/api",
        modelUsage={"some/unknown-model": {}},
    )
    assert resp.cost_source == "unpriced"
    assert resp.cost_usd == 0.0
