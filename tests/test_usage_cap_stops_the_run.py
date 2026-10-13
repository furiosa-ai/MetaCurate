"""A spend cap must stop a run at once, and must not read as configuration.

A shared key hitting its monthly cap can end many loops in seconds, and two
separate defects would turn that one billing event into misleading run
records:

  * the optimizer counts three consecutive proposer failures before
    declaring a run broken, and then calls the cause "a configuration
    failure, not a transient one". For a cap the first failure is already
    conclusive, and that message sends the reader to the auth flags instead
    of the billing page.
  * a cap on the last iteration makes the handler's `continue` fall out of
    the loop and the run return normally: one summary row short, exit **0**.
"""

from __future__ import annotations

import pytest

from meta_filter.llm.base import UsageLimitError
from meta_filter.llm.claude_code import _usage_limit_detail

# Verbatim from pcsA_seed41's log at 22:09:18.6Z, trimmed to the fields that
# matter. A synthesised body would prove only that the matcher matches itself.
CAP_BODY = (
    '{"type":"result","subtype":"error_during_execution","is_error":true,'
    '"api_error_status":400,"result":"API Error: 400 '
    '{\\"type\\":\\"error\\",\\"error\\":{\\"type\\":\\"invalid_request_error\\",'
    '\\"message\\":\\"You have reached your specified API usage limits. You will '
    'regain access on 2026-10-01 at 00:00 UTC.\\"}}"}'
)


def test_a_cap_body_is_recognised_and_keeps_the_regain_date() -> None:
    detail = _usage_limit_detail(CAP_BODY)
    assert detail, "the cap body must be recognised as a cap"
    # The regain date is the one fact a reader needs from the last log line.
    assert "2026-10-01" in detail


@pytest.mark.parametrize(
    "body",
    [
        "",
        None,
        '{"is_error":true,"api_error_status":529,"result":"Overloaded"}',
        # A 400 that is NOT a cap: a malformed request really is a
        # configuration error and must keep the streak handling.
        '{"is_error":true,"api_error_status":400,"result":"API Error: 400 '
        '{\\"message\\":\\"max_tokens: must be greater than 0\\"}"}',
    ],
)
def test_other_failures_are_not_caps(body: str | None) -> None:
    assert _usage_limit_detail(body) == ""


def test_the_error_is_not_a_plain_runtimeerror_to_the_optimizer() -> None:
    """It must be catchable ahead of the generic handler, and still an Exception."""

    assert issubclass(UsageLimitError, RuntimeError)
    assert UsageLimitError is not RuntimeError


def test_the_marker_match_ignores_case() -> None:
    """The Codex client always folded case; this one now shares its matcher."""

    assert _usage_limit_detail(CAP_BODY.replace("API usage limits", "api usage limits"))
