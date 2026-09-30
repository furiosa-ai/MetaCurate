"""Anthropic Messages adapter (lazy import; unused in the offline demo)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from meta_filter.llm.base import LLMResponse
from meta_filter.llm.pricing import cost_from_list_price


@dataclass
class AnthropicClient:
    model: str = ""
    api_key: str | None = None
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = self.model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Use --llm-provider mock for offline runs."
            )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> LLMResponse:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The `anthropic` package is required for the anthropic provider."
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system:
            if cache_system:
                # Native Anthropic prompt caching — mark the system prefix
                # so all subsequent calls within TTL reuse it.
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        text = "".join(getattr(block, "text", "") for block in resp.content)
        usage = getattr(resp, "usage", None)
        # Anthropic reports cache traffic *beside* `input_tokens`, not inside
        # it, so a prompt served from cache reads as a tiny one unless the two
        # cache counters are added back.
        fresh_input = getattr(usage, "input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        output = getattr(usage, "output_tokens", 0) or 0
        prompt_tokens = fresh_input + cache_read + cache_write
        # This route reports no cost of its own, so the rate card is all there
        # is. `list_price`, never `openrouter`: the number is an estimate.
        cost_usd, cost_source = cost_from_list_price(
            self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=output,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=output,
            model=self.model,
            cost_usd=cost_usd,
            cost_source=cost_source,
            cached_tokens=cache_read,
        )
