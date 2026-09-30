"""OpenAI-compatible chat completions adapter.

The heavy client library is imported lazily so the mock demo does not depend
on it. Set OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL to use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from meta_filter.llm.base import LLMResponse
from meta_filter.llm.pricing import cost_from_list_price


@dataclass
class OpenAICompatibleClient:
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    # Proposer responses rewrite three files at once, so 4K is far too small.
    # Claude Sonnet 4 supports output up to 64K on Anthropic/OpenRouter.
    max_tokens: int = 32768
    # Extra top-level JSON body fields, merged into every request. This is
    # where OpenRouter's non-OpenAI controls live, and two of them decide
    # whether a run's numbers mean anything:
    #
    #   {"provider": {"order": ["MiniMax"], "allow_fallbacks": false,
    #                 "quantizations": ["fp8"]}}
    #       pins the serving provider. Unpinned, one model slug is several
    #       different models — different providers serve different
    #       quantizations, and the choice changes call to call.
    #   {"reasoning": {"enabled": false}}
    #       turns thinking off. OpenRouter enables it by default on the
    #       Anthropic family, where a scoring call can then spend its whole
    #       output budget reasoning and return an empty answer.
    #
    # Per role rather than per run: the solver and the judge are usually
    # different models, so `cli.py` takes one of these for each.
    extra_body: dict | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = self.base_url or os.environ.get("OPENAI_BASE_URL") or None
        self.model = self.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Use --llm-provider mock for offline runs."
            )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> LLMResponse:
        # Lazy import so the offline demo doesn't require the SDK to be installed.
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The `openai` package is required for the openai provider.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages: list[dict] = []
        if system:
            # For OpenRouter+Anthropic (and Anthropic direct via a compatible
            # gateway), attaching cache_control to a system content block turns
            # on prompt caching. Any provider that doesn't support the marker
            # simply drops it — falling back to normal behavior.
            if cache_system and self._supports_prompt_cache():
                messages.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
            extra_body=self._request_body(),
        )
        choice = resp.choices[0].message
        usage = resp.usage
        prompt_tokens = int(_field(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(_field(usage, "completion_tokens", 0) or 0)
        cached = int(_field(_field(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
        reasoning = int(
            _field(_field(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        )

        # OpenRouter's own number when `usage: {include: true}` produced one:
        # it is what the account was actually charged, discounts and provider
        # surcharges included, and no rate card can reproduce it.
        reported = _field(usage, "cost", None)
        if reported is not None:
            cost_usd, cost_source = round(float(reported), 6), "openrouter"
        else:
            cost_usd, cost_source = cost_from_list_price(
                self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_tokens=cached,
            )

        return LLMResponse(
            text=choice.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
            cost_usd=cost_usd,
            cost_source=cost_source,
            provider=str(_field(resp, "provider", "") or ""),
            reasoning_tokens=reasoning,
            cached_tokens=cached,
        )

    def _request_body(self) -> dict:
        """`extra_body` plus the usage block, without mutating either.

        `usage: {include: true}` is what makes OpenRouter return `usage.cost`;
        a caller that passes its own `usage` key keeps it.
        """

        body = {"usage": {"include": True}}
        body.update(self.extra_body or {})
        return body

    def _supports_prompt_cache(self) -> bool:
        """Best-effort: Anthropic models exposed via OpenRouter honor
        `cache_control`. Other providers ignore it."""

        model = (self.model or "").lower()
        return model.startswith("anthropic/") or "claude" in model


def _field(obj: object, name: str, default: object = None) -> object:
    """Read `name` off a response object, a dict, or a pydantic extra.

    OpenRouter returns fields the OpenAI SDK's models do not declare —
    `provider`, `usage.cost` — and where they land depends on the SDK
    version: an attribute, `model_extra`, or a plain dict when the caller
    swapped the transport. Reading all three is cheaper than pinning a
    version, and a missing field is normal rather than exceptional (a
    non-OpenRouter gateway names no provider).
    """

    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict) and name in extra:
        return extra[name]
    return default
