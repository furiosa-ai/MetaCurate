"""LLM adapters. The mock client makes the full pipeline runnable offline."""

from meta_filter.llm.base import LLMClient, LLMResponse
from meta_filter.llm.mock import MockLLMClient

__all__ = ["LLMClient", "LLMResponse", "MockLLMClient", "build_client"]


def build_client(provider: str, **kwargs):
    """Factory that returns the appropriate client. Provider names are case-insensitive."""

    provider = provider.lower()
    if provider == "mock":
        return MockLLMClient(**kwargs)
    if provider == "openai":
        from meta_filter.llm.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(**kwargs)
    if provider == "anthropic":
        from meta_filter.llm.anthropic import AnthropicClient

        return AnthropicClient(**kwargs)
    if provider in ("codex", "codex-cli", "codex_cli"):
        from meta_filter.llm.codex_cli import CodexCliClient

        return CodexCliClient(**kwargs)
    if provider in ("claude-code", "claude_code", "claude-cli"):
        from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient

        return ClaudeCodeHeadlessClient(**kwargs)
    raise ValueError(f"unknown LLM provider: {provider}")
