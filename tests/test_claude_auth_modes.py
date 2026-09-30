"""Which identity a headless call runs as.

Both directions can cost a run. Under OAuth, a sourced `.env` would hand the
CLI a second identity and every call would exit 1. The opposite mistake is
possible too: a rented machine has no OAuth login, so stripping the key there
would fail every call instead.
"""

from __future__ import annotations

import pytest

from meta_filter.llm import claude_code


def test_oauth_drops_a_sourced_key_so_the_identity_cannot_be_swapped(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-reach-the-child")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf-unrelated-but-needed")

    env = claude_code._auth_env(claude_code.OAUTH)

    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDE_CODE_SIMPLE" not in env
    # The CLI child gets LEAST PRIVILEGE. An agentic call can run arbitrary
    # python via Bash; an HF token in its environment would let it download
    # exactly the files verification isolation keeps off the machine.
    assert "HUGGING_FACE_HUB_TOKEN" not in env


def test_api_key_mode_keeps_the_key_and_disables_the_login_path(monkeypatch):
    """On a rented machine the strip is the bug: there is no OAuth login to
    fall back to."""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-this-machines-identity")

    env = claude_code._auth_env(claude_code.API_KEY)

    assert env["ANTHROPIC_API_KEY"] == "sk-ant-this-machines-identity"
    assert env["CLAUDE_CODE_SIMPLE"] == "1"


def test_api_key_mode_without_a_key_fails_at_setup_not_per_call(monkeypatch):
    """The OAuth version of this mistake failed one call at a time and got a
    long way in before anyone noticed."""

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        claude_code._auth_env(claude_code.API_KEY)


def test_a_typo_in_the_auth_mode_is_not_silently_oauth(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    with pytest.raises(ValueError, match="unknown auth mode"):
        claude_code._auth_env("apikey")


def test_the_default_is_oauth_and_the_environment_can_select_otherwise(monkeypatch):
    """A machine selects its mode once, in its own environment, rather than
    every launch command growing a flag."""

    monkeypatch.delenv("META_FILTER_CLAUDE_AUTH", raising=False)
    assert claude_code.ClaudeCodeHeadlessClient().auth == claude_code.OAUTH

    monkeypatch.setenv("META_FILTER_CLAUDE_AUTH", claude_code.API_KEY)
    assert claude_code.ClaudeCodeHeadlessClient().auth == claude_code.API_KEY
