"""ClaudeCodeHeadlessClient — subprocess-mocked contract tests."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from meta_filter.llm import build_client
from meta_filter.llm.base import LLMResponse
from meta_filter.llm.claude_code import (
    ClaudeCodeHeadlessClient,
    is_supported_alias,
)


def _fake_run_result(payload: dict, *, returncode: int = 0, stderr: str = ""):
    class _R:
        pass

    r = _R()
    r.returncode = returncode
    r.stdout = json.dumps(payload)
    r.stderr = stderr
    return r


def test_supported_aliases():
    for name in ("haiku", "sonnet", "opus"):
        assert is_supported_alias(name)
    assert not is_supported_alias("gpt-4o")


def test_generate_parses_result_and_usage(tmp_path):
    client = ClaudeCodeHeadlessClient(model="haiku", scratch_dir=tmp_path)
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "hello world",
        "usage": {"input_tokens": 42, "output_tokens": 7},
        "modelUsage": {"claude-haiku-4-5-20251001": {}},
    }
    with patch(
        "meta_filter.llm.claude_code.subprocess.run",
        return_value=_fake_run_result(payload),
    ) as m:
        resp = client.generate("some prompt", system="you are helpful")
    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello world"
    assert resp.prompt_tokens == 42
    assert resp.completion_tokens == 7
    assert resp.model.startswith("claude-haiku-")

    args, kwargs = m.call_args
    argv = args[0]
    # Sanity: model alias + JSON output + no CLAUDE.md leakage + system prompt
    # forwarded.
    assert "--model" in argv and "haiku" in argv
    assert "--output-format" in argv and "json" in argv
    assert "--disable-slash-commands" in argv
    assert "--exclude-dynamic-system-prompt-sections" in argv
    assert "--system-prompt" in argv
    assert kwargs["input"] == "some prompt"
    assert str(tmp_path) == kwargs["cwd"]


def test_generate_raises_on_nonzero_exit(tmp_path):
    client = ClaudeCodeHeadlessClient(model="haiku", scratch_dir=tmp_path)
    with patch(
        "meta_filter.llm.claude_code.subprocess.run",
        return_value=_fake_run_result({}, returncode=1, stderr="boom\n"),
    ):
        with pytest.raises(RuntimeError, match="exit 1"):
            client.generate("x")


def test_failure_message_reaches_for_stdout_when_stderr_is_silent(tmp_path):
    """The CLI reports most failures on stdout, so a stderr-only message is blank."""

    client = ClaudeCodeHeadlessClient(model="haiku", scratch_dir=tmp_path)
    r = _fake_run_result({}, returncode=1, stderr="")
    r.stdout = "Invalid API key - please run /login"
    with patch("meta_filter.llm.claude_code.subprocess.run", return_value=r):
        with pytest.raises(RuntimeError, match="Invalid API key"):
            client.generate("x")


def test_a_stray_api_key_is_kept_out_of_the_subprocess(tmp_path, monkeypatch):
    """This client authenticates as the OAuth session, not as whatever key
    happens to be exported.

    Sourcing a shared `.env` for an unrelated credential would hand the CLI a
    second identity, which it silently prefers — every call then exits 1
    before anyone reads the message.
    """

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-stray")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf-unrelated")
    client = ClaudeCodeHeadlessClient(model="haiku", scratch_dir=tmp_path)
    payload = {"type": "result", "result": "ok", "usage": {}}
    with patch(
        "meta_filter.llm.claude_code.subprocess.run",
        return_value=_fake_run_result(payload),
    ) as m:
        client.generate("x")

    child_env = m.call_args.kwargs["env"]
    assert "ANTHROPIC_API_KEY" not in child_env
    # Infrastructure secrets are scrubbed from the CLI child too (least
    # privilege: an agentic call can run arbitrary code).
    assert "HUGGING_FACE_HUB_TOKEN" not in child_env


def test_generate_raises_on_api_error_payload(tmp_path):
    client = ClaudeCodeHeadlessClient(model="haiku", scratch_dir=tmp_path)
    payload = {
        "type": "result",
        "is_error": True,
        "api_error_status": "rate_limit_error",
        "result": "hit the wall",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    with patch(
        "meta_filter.llm.claude_code.subprocess.run",
        return_value=_fake_run_result(payload),
    ):
        with pytest.raises(RuntimeError, match="api error"):
            client.generate("x")


def test_max_budget_usd_forwarded(tmp_path):
    client = ClaudeCodeHeadlessClient(model="sonnet", scratch_dir=tmp_path, max_budget_usd=2.5)
    payload = {
        "type": "result",
        "is_error": False,
        "result": "ok",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    with patch(
        "meta_filter.llm.claude_code.subprocess.run",
        return_value=_fake_run_result(payload),
    ) as m:
        client.generate("x")
    argv = m.call_args[0][0]
    assert "--max-budget-usd" in argv
    assert "2.5" in argv


def test_static_client_gets_no_tools(tmp_path):
    client = ClaudeCodeHeadlessClient(model="haiku", scratch_dir=tmp_path)
    assert not client.is_agentic
    payload = {"is_error": False, "result": "ok", "usage": {}}
    with patch(
        "meta_filter.llm.claude_code.subprocess.run",
        return_value=_fake_run_result(payload),
    ) as m:
        client.generate("x")
    argv = m.call_args[0][0]
    assert "--allowedTools" not in argv
    assert "--append-system-prompt" not in argv
    # --dangerously-skip-permissions pre-approves every tool, so only an
    # explicit deny keeps a static call away from the filesystem.
    denied = argv[argv.index("--disallowedTools") + 1 :]
    for tool in ("Bash", "Read", "Write", "Edit", "Glob", "Grep"):
        assert tool in denied


def test_agentic_client_is_denied_the_tools_it_did_not_ask_for(tmp_path):
    client = ClaudeCodeHeadlessClient(
        model="opus", scratch_dir=tmp_path, allowed_tools=("Read", "Glob")
    )
    payload = {"is_error": False, "result": "ok", "usage": {}}
    with patch(
        "meta_filter.llm.claude_code.subprocess.run",
        return_value=_fake_run_result(payload),
    ) as m:
        client.generate("x")
    argv = m.call_args[0][0]
    denied = argv[argv.index("--disallowedTools") + 1 :]
    assert "Read" not in denied and "Glob" not in denied
    for tool in ("Bash", "Write", "Edit", "Task", "WebFetch"):
        assert tool in denied


def test_agentic_client_forwards_tools_and_workspace(tmp_path):
    workspace = tmp_path / "run"
    workspace.mkdir()
    client = ClaudeCodeHeadlessClient(
        model="opus",
        scratch_dir=tmp_path,
        allowed_tools=("Read", "Glob", "Grep", "Bash"),
        append_system_prompt="layout notes",
    ).with_workspace(workspace)
    assert client.is_agentic
    payload = {"is_error": False, "result": "ok", "usage": {}}
    with patch(
        "meta_filter.llm.claude_code.subprocess.run",
        return_value=_fake_run_result(payload),
    ) as m:
        client.generate("x")
    argv, kwargs = m.call_args[0][0], m.call_args[1]
    assert argv[argv.index("--allowedTools") + 1 : argv.index("--allowedTools") + 5] == [
        "Read",
        "Glob",
        "Grep",
        "Bash",
    ]
    assert argv[argv.index("--append-system-prompt") + 1] == "layout notes"
    # The agent's sandbox is its cwd, so the workspace must be what it sees.
    assert kwargs["cwd"] == str(workspace)


def test_build_client_dispatch(tmp_path):
    c = build_client("claude-code", model="haiku", scratch_dir=tmp_path)
    assert isinstance(c, ClaudeCodeHeadlessClient)
    assert c.model == "haiku"

    c2 = build_client("claude-cli", model="opus", scratch_dir=tmp_path)
    assert isinstance(c2, ClaudeCodeHeadlessClient)
    assert c2.model == "opus"
