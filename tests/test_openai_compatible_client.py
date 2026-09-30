"""What the OpenAI-compatible client sends and what it records.

Two things the loop has to see. It has to pin a model to one serving
provider and turn a reasoning model's thinking off, or "minimax/minimax-m3"
means whichever provider and quantization OpenRouter picked that minute — and
on an Anthropic-family model OpenRouter enables thinking by default, which is
how a scoring call burns its budget reasoning and returns nothing. And it has
to record cost, or per-role spend must be reconstructed from token counts
after the fact.

The client is exercised against a stub `openai` module: what matters here is
the request body it builds and the fields it lifts out of the response, both
of which are ours.
"""

from __future__ import annotations

import sys
import types

import pytest

from meta_filter.llm.openai_compatible import OpenAICompatibleClient


class _Capture:
    """Stands in for `openai.OpenAI`, remembering the call it was given."""

    last_kwargs: dict = {}
    last_init: dict = {}
    response: object = None

    def __init__(self, **kwargs):
        type(self).last_init = kwargs
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        type(self).last_kwargs = kwargs
        return type(self).response


def _response(
    *,
    text: str = "ok",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cost: float | None = 0.000123,
    provider: str | None = "MiniMax",
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
):
    usage = types.SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning_tokens),
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached_tokens),
    )
    if cost is not None:
        usage.cost = cost
    resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))],
        usage=usage,
    )
    if provider is not None:
        resp.provider = provider
    return resp


@pytest.fixture
def stub_openai(monkeypatch):
    module = types.ModuleType("openai")
    module.OpenAI = _Capture
    monkeypatch.setitem(sys.modules, "openai", module)
    _Capture.last_kwargs = {}
    _Capture.response = _response()
    return _Capture


def _client(**kwargs) -> OpenAICompatibleClient:
    kwargs.setdefault("model", "minimax/minimax-m3")
    kwargs.setdefault("api_key", "test-key")
    return OpenAICompatibleClient(**kwargs)


# --- the request ------------------------------------------------------------


def test_usage_accounting_is_requested_even_without_extra_body(stub_openai):
    """`usage.include` is what makes OpenRouter report what it charged."""

    _client().generate("hi")
    assert stub_openai.last_kwargs["extra_body"] == {"usage": {"include": True}}


def test_extra_body_is_passed_through(stub_openai):
    pin = {
        "provider": {
            "order": ["MiniMax"],
            "allow_fallbacks": False,
            "quantizations": ["fp8"],
        },
        "reasoning": {"enabled": False},
    }
    _client(extra_body=pin).generate("hi")

    body = stub_openai.last_kwargs["extra_body"]
    assert body["provider"] == pin["provider"]
    assert body["reasoning"] == {"enabled": False}
    assert body["usage"] == {"include": True}


def test_a_caller_may_override_the_usage_block(stub_openai):
    _client(extra_body={"usage": {"include": False}}).generate("hi")
    assert stub_openai.last_kwargs["extra_body"]["usage"] == {"include": False}


def test_extra_body_does_not_leak_between_calls(stub_openai):
    """The client must not accumulate into its own constructor argument."""

    client = _client(extra_body={"reasoning": {"enabled": False}})
    client.generate("hi")
    client.generate("hi again")
    assert client.extra_body == {"reasoning": {"enabled": False}}


# --- the response -----------------------------------------------------------


def test_a_reported_cost_is_used_as_measured(stub_openai):
    resp = _client().generate("hi")
    assert resp.cost_usd == pytest.approx(0.000123)
    assert resp.cost_source == "openrouter"
    assert resp.provider == "MiniMax"
    assert resp.prompt_tokens == 100
    assert resp.completion_tokens == 20


def test_reasoning_and_cached_tokens_are_recorded(stub_openai):
    stub_openai.response = _response(reasoning_tokens=1349, cached_tokens=64)
    resp = _client().generate("hi")
    assert resp.reasoning_tokens == 1349
    assert resp.cached_tokens == 64


def test_without_a_reported_cost_the_rate_card_is_used(stub_openai):
    stub_openai.response = _response(cost=None, prompt_tokens=1_000_000, completion_tokens=0)
    resp = _client().generate("hi")
    # minimax/minimax-m3 input list price: $0.30 per 1M.
    assert resp.cost_usd == pytest.approx(0.30)
    assert resp.cost_source == "list_price"


def test_an_unpriced_model_says_so_rather_than_reporting_zero(stub_openai):
    stub_openai.response = _response(cost=None)
    resp = _client(model="some-vendor/unknown-model").generate("hi")
    assert resp.cost_usd == 0.0
    assert resp.cost_source == "unpriced"


def test_a_cached_prompt_is_not_billed_as_fresh_input(stub_openai):
    stub_openai.response = _response(
        cost=None, prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=1_000_000
    )
    resp = _client().generate("hi")
    # All of it read from cache: $0.06 per 1M, not $0.30.
    assert resp.cost_usd == pytest.approx(0.06)


def test_a_route_that_names_no_provider_is_not_an_error(stub_openai):
    stub_openai.response = _response(provider=None)
    resp = _client().generate("hi")
    assert resp.provider == ""


# --- the CLI flags that fill it ---------------------------------------------


def test_a_flag_becomes_client_kwargs() -> None:
    from meta_filter.cli import _extra_body_kwargs

    body = '{"reasoning": {"enabled": false}}'
    assert _extra_body_kwargs(body, "openai", "--solver-extra-body") == {
        "extra_body": {"reasoning": {"enabled": False}}
    }
    assert _extra_body_kwargs(None, "openai", "--solver-extra-body") == {}


def test_a_pin_aimed_at_a_provider_that_cannot_apply_it_stops_the_run() -> None:
    """The quiet failure: the Claude and Codex CLIs build their own request
    body, so a pin passed to them leaves the run unpinned while looking
    pinned. Exit rather than warn."""

    import typer

    from meta_filter.cli import _extra_body_kwargs

    with pytest.raises(typer.Exit):
        _extra_body_kwargs('{"reasoning": {"enabled": false}}', "claude-code", "--flag")


def test_malformed_json_stops_the_run() -> None:
    import typer

    from meta_filter.cli import _extra_body_kwargs

    with pytest.raises(typer.Exit):
        _extra_body_kwargs("{not json", "openai", "--flag")
    with pytest.raises(typer.Exit):
        _extra_body_kwargs('["a", "list"]', "openai", "--flag")
