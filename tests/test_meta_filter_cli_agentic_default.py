"""The agentic default is safety-critical, so pin it in both directions.

Forgetting `--agentic` would swap the proposer without failing, and two runs
that differ only in that would not be comparable.
"""

from __future__ import annotations

import pytest

from meta_filter.cli import resolve_agentic


@pytest.mark.parametrize(
    ("flag", "provider", "expected"),
    [
        (None, "claude-code", True),
        (None, "codex", True),
        (None, "codex-cli", True),
        (None, "openai", False),
        (None, "mock", False),
        (None, "openrouter", False),
        (True, "claude-code", True),
        (False, "claude-code", False),
        (True, "mock", True),
    ],
)
def test_resolve_agentic(flag: bool | None, provider: str, expected: bool) -> None:
    assert resolve_agentic(flag, provider) is expected
