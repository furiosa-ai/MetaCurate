"""CodexCliClient — subprocess-mocked contract tests plus real-log parsing.

The `codex` binary is never invoked here. `_run_leg` is replaced by a fake
that writes a canned `--json` stream, which is exactly what the real one
does, so everything downstream of the subprocess is under test: config
rendering, AGENTS.md delivery, the stdin prompt, log parsing, cost, and the
boundary audit.

`tests/fixtures/codex_leg1.jsonl` is a stream from a real Codex 0.118
proposer call (iteration 9, routed through OpenRouter), with its workspace
path, thread id and timestamps replaced by synthetic ones and its command
outputs omitted. Its numbers are asserted literally: if a
Codex release changes the stream shape, these tests fail rather than the
token accounting silently reading zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_filter.llm import build_client
from meta_filter.llm import codex_cli as cc
from meta_filter.llm import harness_guard as hg
from meta_filter.llm.base import LLMResponse, UsageLimitError
from meta_filter.llm.codex_cli import CodexCliClient, parse_legs

FIXTURE = Path(__file__).parent / "fixtures" / "codex_leg1.jsonl"

# The real call's own numbers. `input` is the prompt minus the cached slice:
# 1_281_674 - 1_201_907.
REAL_USAGE = {"input": 79_767, "output": 9_080, "cache_read": 1_201_907, "cache_write": 0}
REAL_THREAD = "00000000-0000-4000-8000-000000000001"
REAL_TOOL_CALLS = 14
# The workspace that call actually ran in, so "inside the boundary" can be
# tested against real paths rather than invented ones.
REAL_WORKSPACE = Path("/tmp/meta-filter-workspace/run")


def _stream(*records: dict) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def _turn(input_tokens: int = 1000, cached: int = 400, output: int = 50) -> dict:
    return {
        "type": "turn.completed",
        "usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached,
            "output_tokens": output,
        },
    }


def _message(text: str) -> dict:
    return {
        "type": "item.completed",
        "item": {"id": "item_1", "type": "agent_message", "text": text},
    }


def _client(tmp_path: Path, **kwargs) -> CodexCliClient:
    """A client with a fake credential file, so no real key is ever read."""

    env_file = tmp_path / "fake.env"
    env_file.write_text("OPENROUTER_API_KEY=test-key-placeholder\n", encoding="utf-8")
    kwargs.setdefault("scratch_dir", tmp_path / "ws")
    kwargs.setdefault("codex_home", tmp_path / "home")
    return CodexCliClient(env_file=env_file, **kwargs)


class _FakeLeg:
    """Stands in for `_run_leg`, recording what it was handed.

    `body` is what the leg writes to its log; a callable receives the leg
    number so a timeout-then-resume pair can return different streams.
    """

    def __init__(self, body, *, timed_out=False, returncode=0):
        self.body = body
        self.timed_out = timed_out if isinstance(timed_out, list) else [timed_out]
        self.returncode = returncode
        self.calls: list[dict] = []

    def __call__(self, argv, *, cwd, env, log, timeout, stdin_file):
        n = len(self.calls) + 1
        body = self.body(n) if callable(self.body) else self.body
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(body, encoding="utf-8")
        agents = Path(cwd) / "AGENTS.md"
        self.calls.append(
            {
                "argv": list(argv),
                "cwd": Path(cwd),
                "env": dict(env),
                "timeout": timeout,
                "stdin": Path(stdin_file).read_text(encoding="utf-8") if stdin_file else None,
                "agents_md": agents.read_text(encoding="utf-8") if agents.exists() else None,
                "workspace_listing": sorted(p.name for p in Path(cwd).iterdir()),
            }
        )
        timed_out = self.timed_out[min(n - 1, len(self.timed_out) - 1)]
        return {
            "argv": list(argv),
            "timed_out": timed_out,
            "returncode": self.returncode,
            "wall_seconds": 1.0,
            "log": Path(log).name,
        }


@pytest.fixture
def leg(monkeypatch):
    def install(body, **kwargs):
        fake = _FakeLeg(body, **kwargs)
        monkeypatch.setattr(cc, "_run_leg", fake)
        return fake

    return install


# ------------------------------------------------------------------ config


def test_config_declares_the_openrouter_provider(tmp_path):
    config = _client(tmp_path).render_config()
    assert 'model_provider = "openrouter"' in config
    assert "[model_providers.openrouter]" in config
    assert 'base_url = "https://openrouter.ai/api/v1"' in config
    assert 'env_key = "OPENROUTER_API_KEY"' in config
    # Codex 0.118 speaks only the responses shape; OpenRouter serves both.
    assert 'wire_api = "responses"' in config
    # No credential is ever written into the config file.
    assert "test-key-placeholder" not in config


def test_argv_carries_the_openrouter_model_slug(tmp_path):
    client = _client(tmp_path, model="openai/gpt-5.6-sol")
    argv = client.argv(tmp_path / "ws")
    assert argv[:2] == ["codex", "exec"]
    assert argv[argv.index("-m") + 1] == "openai/gpt-5.6-sol"
    assert argv[argv.index("-C") + 1] == str(tmp_path / "ws")
    assert "--json" in argv and "--skip-git-repo-check" in argv
    assert 'approval_policy="never"' in argv


def test_sandbox_is_read_only_until_tools_are_granted(tmp_path):
    assert _client(tmp_path).sandbox_mode == "read-only"
    assert _client(tmp_path, allowed_tools=("Read",)).sandbox_mode == "workspace-write"


def test_keyless_auth_json_is_removed(tmp_path):
    """A present-but-keyless auth.json beats env_key and kills the call."""

    client = _client(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.json").write_text('{"auth_mode": "apikey"}', encoding="utf-8")
    client._prepare_home()
    assert not (home / "auth.json").exists()
    assert (home / "config.toml").is_file()


def test_child_env_keeps_one_credential_and_drops_the_rest(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-survive")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-should-not-survive")
    monkeypatch.setenv("HF_TOKEN", "hf-should-not-survive")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = _client(tmp_path).child_env()
    assert env["OPENROUTER_API_KEY"] == "test-key-placeholder"
    for dropped in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"):
        assert dropped not in env
    assert env["CODEX_HOME"] == str(tmp_path / "home")
    # The operator's own Codex config must be unreachable.
    assert env["XDG_CONFIG_HOME"].startswith(str(tmp_path / "home"))


def test_missing_credential_is_a_clear_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    empty = tmp_path / "empty.env"
    empty.write_text("SOMETHING_ELSE=1\n", encoding="utf-8")
    client = CodexCliClient(scratch_dir=tmp_path / "ws", env_file=empty)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is unset"):
        client.child_env()


def test_build_client_registers_the_provider(tmp_path):
    client = build_client("codex", scratch_dir=tmp_path / "ws")
    assert isinstance(client, CodexCliClient)


# --------------------------------------------------- prompt and system prompt


def test_prompt_is_delivered_on_stdin_not_argv(tmp_path, leg):
    fake = leg(_stream(_message("done"), _turn()))
    prompt = "x" * 200_000  # past MAX_ARG_STRLEN; must not be an argv token
    _client(tmp_path).generate(prompt)
    call = fake.calls[0]
    assert call["stdin"] == prompt
    assert prompt not in call["argv"]
    assert not any(len(a) > 1000 for a in call["argv"])


def test_system_prompt_arrives_as_agents_md_and_is_removed(tmp_path, leg):
    fake = leg(_stream(_message("done"), _turn()))
    client = _client(
        tmp_path,
        allowed_tools=("Read",),
        append_system_prompt="LAYOUT: candidates live under iterations/",
    )
    client.generate("go", system="You are the proposer.")
    delivered = fake.calls[0]["agents_md"]
    assert "You are the proposer." in delivered
    assert "LAYOUT: candidates live under iterations/" in delivered
    # Left behind, it would silently become the next call's system prompt.
    assert not (tmp_path / "ws" / "AGENTS.md").exists()


def test_a_preexisting_agents_md_is_restored(tmp_path, leg):
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("project instructions", encoding="utf-8")
    fake = leg(_stream(_message("done"), _turn()))
    client = _client(tmp_path, allowed_tools=("Read",))
    client.generate("go", system="call-scoped prompt")
    # The call's own prompt comes first; the client appends its scratch-dir
    # note (TMPDIR inside the workspace) after it.
    assert fake.calls[0]["agents_md"].startswith("call-scoped prompt")
    assert "TMPDIR" in fake.calls[0]["agents_md"]
    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == "project instructions"


def test_static_call_runs_in_a_workspace_holding_only_the_prompt(tmp_path, leg):
    """No tool allow-list exists, so emptiness is what makes static static."""

    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "secret_evidence.json").write_text("{}", encoding="utf-8")
    fake = leg(_stream(_message("OK"), _turn()))
    _client(tmp_path).generate("Reply with the single word OK", system="be terse")
    call = fake.calls[0]
    assert call["cwd"] != workspace
    assert "secret_evidence.json" not in call["workspace_listing"]
    assert call["workspace_listing"] == ["AGENTS.md"]
    assert call["argv"][call["argv"].index("-s") + 1] == "read-only"


def test_agentic_call_runs_in_the_run_directory(tmp_path, leg):
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "record.json").write_text("{}", encoding="utf-8")
    fake = leg(_stream(_message("done"), _turn()))
    _client(tmp_path, allowed_tools=("Read", "Bash")).generate("go")
    assert fake.calls[0]["cwd"] == workspace
    assert "record.json" in fake.calls[0]["workspace_listing"]


def test_with_workspace_moves_both_the_cwd_and_the_boundary(tmp_path):
    target = tmp_path / "run_dir"
    target.mkdir()
    client = _client(tmp_path).with_workspace(target)
    assert client.scratch_dir == target
    assert client.guard_root == target
    assert client.audit_path == target / ".agent_audit.jsonl"


# -------------------------------------------------------------- log parsing


def test_parses_a_real_codex_stream():
    parsed = parse_legs([FIXTURE])
    assert parsed["usage"] == REAL_USAGE
    assert parsed["usage_seen"] is True
    assert parsed["thread_ids"] == [REAL_THREAD]
    assert len(parsed["calls"]) == REAL_TOOL_CALLS
    assert sum(1 for c in parsed["calls"] if c["tool"] == "command_execution") == 13
    assert sum(1 for c in parsed["calls"] if c["tool"] == "file_change") == 1
    assert parsed["text"].strip()
    # A file_change the sandbox refused is an attempt, not a landed write.
    change = next(c for c in parsed["calls"] if c["tool"] == "file_change")
    assert change["status"] == "failed"
    assert change["writes"] == []


def test_cached_prompt_is_not_billed_twice():
    """`input_tokens` includes the cached slice; pricing it twice inflates."""

    parsed = parse_legs([FIXTURE])
    assert parsed["usage"]["input"] + parsed["usage"]["cache_read"] == 1_281_674


def test_cost_of_the_real_call_at_list_price(tmp_path):
    client = _client(tmp_path, model="openai/gpt-5.6-luna")
    cost, source = client._cost(parse_legs([FIXTURE])["usage"], 0.0)
    assert source == "list_price"
    assert cost == pytest.approx(0.050888, abs=1e-6)


def test_a_harness_reported_cost_wins_over_list_price(tmp_path):
    client = _client(tmp_path, model="openai/gpt-5.6-luna")
    cost, source = client._cost(parse_legs([FIXTURE])["usage"], 0.0421)
    assert (cost, source) == (0.0421, "harness")


def test_unpriced_model_reports_zero_rather_than_guessing(tmp_path):
    client = _client(tmp_path, model="vendor/not-in-the-table")
    assert client._cost({"input": 1, "output": 1, "cache_read": 0, "cache_write": 0}, 0.0) == (
        0.0,
        "unpriced",
    )


def test_generate_returns_text_usage_and_cost(tmp_path, leg):
    leg(FIXTURE.read_text(encoding="utf-8"))
    resp = _client(tmp_path, model="openai/gpt-5.6-luna", allowed_tools=("Read",)).generate("go")
    assert isinstance(resp, LLMResponse)
    assert resp.text.strip()
    assert resp.completion_tokens == REAL_USAGE["output"]
    assert resp.prompt_tokens == REAL_USAGE["input"] + REAL_USAGE["cache_read"]
    assert resp.cached_tokens == REAL_USAGE["cache_read"]
    assert resp.cost_usd == pytest.approx(0.050888, abs=1e-6)
    assert resp.tool_calls == REAL_TOOL_CALLS
    assert resp.thread_id == REAL_THREAD
    assert resp.model == "openai/gpt-5.6-luna"


def test_tool_call_log_is_saved_next_to_the_transcript(tmp_path, leg):
    leg(FIXTURE.read_text(encoding="utf-8"))
    resp = _client(tmp_path, allowed_tools=("Read",)).generate("go")
    call_dir = Path(resp.transcript_dir)
    assert (call_dir / "leg1.jsonl").is_file()
    assert (call_dir / "prompt.txt").read_text(encoding="utf-8") == "go"
    saved = json.loads((call_dir / "tool_calls.json").read_text(encoding="utf-8"))
    assert len(saved) == REAL_TOOL_CALLS
    assert json.loads((call_dir / "legs.json").read_text(encoding="utf-8"))[0]["log"] == (
        "leg1.jsonl"
    )


# ---------------------------------------------------------- boundary audit


def _outside_write_stream(path: Path) -> str:
    return _stream(
        {
            "type": "item.completed",
            "item": {
                "id": "item_3",
                "type": "file_change",
                "status": "completed",
                "changes": [{"path": str(path), "kind": "add"}],
            },
        },
        _message("wrote it"),
        _turn(),
    )


def test_a_write_outside_the_workspace_raises(tmp_path, leg):
    escapee = tmp_path / "elsewhere" / "stolen.py"
    leg(_outside_write_stream(escapee))
    client = _client(tmp_path, allowed_tools=("Write",))
    with pytest.raises(hg.BoundaryViolation, match="stolen.py"):
        client.generate("go")

    audit = json.loads(
        "[" + ",".join((tmp_path / "ws" / ".agent_audit.jsonl").read_text().splitlines()) + "]"
    )
    denied = [r for r in audit if r["decision"] == "deny"]
    assert [r["path"] for r in denied] == [str(escapee)]
    assert denied[0]["tool"] == "file_change"
    # Honest about what happened: the write landed, it was not blocked.
    assert denied[0]["enforced"] is False


def test_a_write_to_tmp_counts_as_outside(tmp_path, leg):
    leg(_outside_write_stream(Path("/tmp/scratch-escape.json")))
    with pytest.raises(hg.BoundaryViolation, match="/tmp/scratch-escape.json"):
        _client(tmp_path, allowed_tools=("Write",)).generate("go")


def test_a_shell_redirect_out_of_the_workspace_is_flagged_not_raised(tmp_path, leg):
    """Weaker evidence than a reported write, so it reports instead of aborting."""

    leg(
        _stream(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_3",
                    "type": "command_execution",
                    "command": "cat > /tmp/scratch_notes.json <<'EOF'\n{}\nEOF",
                    "exit_code": 0,
                },
            },
            _message("done"),
            _turn(),
        )
    )
    resp = _client(tmp_path, allowed_tools=("Bash",)).generate("go")
    assert resp.inferred_writes_outside_workspace == ("/tmp/scratch_notes.json",)
    # Not a read, and not fatal.
    assert resp.reads_outside_workspace == ()
    assert resp.text


def test_shell_redirect_targets_are_recognised():
    assert hg.shell_write_targets("echo hi > /tmp/a.txt") == ["/tmp/a.txt"]
    assert hg.shell_write_targets("echo hi >>/tmp/a.txt") == ["/tmp/a.txt"]
    assert hg.shell_write_targets("echo hi | tee -a /var/log/x") == ["/var/log/x"]
    assert hg.shell_write_targets("grep foo bar.txt") == []


def test_a_read_outside_the_workspace_is_recorded_not_raised(tmp_path, leg):
    leg(
        _stream(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_3",
                    "type": "command_execution",
                    "command": "cat /home/user/other_run/record.json",
                    "exit_code": 0,
                },
            },
            _message("read it"),
            _turn(),
        )
    )
    resp = _client(tmp_path, allowed_tools=("Bash",)).generate("go")
    assert resp.reads_outside_workspace == ("/home/user/other_run/record.json",)
    audit = (tmp_path / "ws" / ".agent_audit.jsonl").read_text(encoding="utf-8")
    assert '"decision": "warn"' in audit


def test_a_contained_call_writes_no_violations(tmp_path, leg):
    workspace = tmp_path / "ws"
    inside = workspace / "iterations" / "001" / "cand" / "policy.py"
    leg(
        _stream(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_3",
                    "type": "file_change",
                    "status": "completed",
                    "changes": [{"path": str(inside), "kind": "add"}],
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "item_4",
                    "type": "command_execution",
                    "command": "/usr/bin/grep -r pattern iterations/",
                    "exit_code": 0,
                },
            },
            _message("done"),
            _turn(),
        )
    )
    resp = _client(tmp_path, allowed_tools=("Write", "Bash")).generate("go")
    assert resp.reads_outside_workspace == ()
    records = json.loads(
        (Path(resp.transcript_dir) / "boundary_audit.json").read_text(encoding="utf-8")
    )
    assert records == []


def test_the_real_call_wrote_nothing_outside_but_did_reach_tmp():
    """The captured call, audited against the root it actually ran in.

    It edited nothing outside its workspace, and it read nothing outside
    either — but it did `cat > /tmp/iter009_diag.json`. Codex's
    `workspace-write` sandbox counts the system temp directory as writable,
    so that is a real reach the audit has to surface, not a parser artifact.
    """

    records = hg.audit_records(parse_legs([FIXTURE])["calls"], REAL_WORKSPACE)
    assert hg.writes_outside(records) == []
    assert hg.reads_outside(records) == []
    assert hg.inferred_writes_outside(records) == ["/tmp/iter009_diag.json"]


def test_a_brace_expansion_is_not_mistaken_for_an_absolute_path():
    """`iterations/{006,007}/proposer/x.json` is relative, all of it.

    A `}` is neither a word character nor a dot, so the obvious lookbehind
    reads the tail as the absolute path `/proposer/x.json` and reports a
    phantom escape. The real call contains exactly this command.
    """

    command = "for f in iterations/{006,007,008}/proposer/response.json; do cat $f; done"
    assert hg.absolute_paths(command) == []
    assert hg.absolute_paths("cat /etc/passwd") == ["/etc/passwd"]
    assert hg.absolute_paths("cp a.py '/other/run/b.py'") == ["/other/run/b.py"]


# ------------------------------------------------------------ failure legs


def test_a_timeout_resumes_the_thread_once(tmp_path, leg):
    def body(n: int) -> str:
        if n == 1:
            return _stream(
                {"type": "thread.started", "thread_id": "th-42"}, {"type": "turn.started"}
            )
        return _stream(_message("finished on the second leg"), _turn())

    fake = leg(body, timed_out=[True, False])
    resp = _client(tmp_path, allowed_tools=("Read",)).generate("go")
    assert len(fake.calls) == 2
    second = fake.calls[1]["argv"]
    assert second[1:4] == ["exec", "resume", "th-42"]
    # The prompt is already in the thread; resending it would double the bill.
    assert fake.calls[1]["stdin"] is None
    assert cc.RESUME_NUDGE in second
    assert "finished on the second leg" in resp.text


def test_two_timeouts_with_no_output_raise(tmp_path, leg):
    leg(_stream({"type": "thread.started", "thread_id": "th-42"}), timed_out=[True, True])
    with pytest.raises(RuntimeError, match="timed out twice"):
        _client(tmp_path, allowed_tools=("Read",)).generate("go")


def test_a_timeout_before_the_thread_starts_cannot_resume(tmp_path, leg):
    leg("", timed_out=True)
    with pytest.raises(RuntimeError, match="nothing to resume"):
        _client(tmp_path).generate("go")


def test_a_spend_cap_raises_usage_limit_error(tmp_path, leg):
    leg(
        _stream({"type": "error", "message": "402 Payment Required: insufficient credits"}),
        returncode=1,
    )
    with pytest.raises(UsageLimitError, match="insufficient credits"):
        _client(tmp_path).generate("go")


def test_a_nonzero_exit_reports_the_log_tail(tmp_path, leg):
    leg("codex: unknown model provider 'openrouter'\n", returncode=2)
    with pytest.raises(RuntimeError, match="unknown model provider"):
        _client(tmp_path).generate("go")


def test_an_empty_stream_is_a_failure_not_an_empty_answer(tmp_path, leg):
    leg("")
    with pytest.raises(RuntimeError, match="neither text nor tool calls"):
        _client(tmp_path).generate("go")


def test_usage_recovered_from_the_rollout_when_the_stream_has_none(tmp_path, leg):
    """A killed leg emits no `turn.completed`; the rollout survives the kill."""

    home = tmp_path / "home"
    sessions = home / "sessions" / "2000" / "01" / "01"
    sessions.mkdir(parents=True)
    (sessions / "rollout-2000-01-01T00-00-00-th-42.jsonl").write_text(
        _stream(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"text": "partial answer"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 900,
                            "cached_input_tokens": 400,
                            "output_tokens": 20,
                        }
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    leg(_stream({"type": "thread.started", "thread_id": "th-42"}), timed_out=[True, True])
    resp = _client(tmp_path, model="openai/gpt-5.6-luna", allowed_tools=("Read",)).generate("go")
    assert resp.text == "partial answer"
    assert resp.completion_tokens == 20
    assert resp.cached_tokens == 400
    assert resp.prompt_tokens == 900


def test_call_directories_number_from_disk_across_workspace_copies(tmp_path):
    """Each `with_workspace` copy restarts its call counter at 0, so numbering
    from the counter would land every call in call001 and keep only the last
    transcript."""

    from meta_filter.llm.codex_cli import CodexCliClient

    base = CodexCliClient(model="openai/gpt-5.6-luna")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".codex_calls" / "call001").mkdir(parents=True)
    (ws / ".codex_calls" / "call002").mkdir(parents=True)
    fresh_copy = base.with_workspace(ws)
    assert fresh_copy._next_call_index() == 3
    assert base.with_workspace(tmp_path / "empty")._next_call_index() == 1


def test_child_env_points_tmpdir_inside_the_workspace(tmp_path, monkeypatch):
    """Two concurrent Codex runs can share /tmp file names; a per-call TMPDIR
    inside the workspace keeps scratch inside the boundary."""

    from meta_filter.llm.codex_cli import CodexCliClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    c = CodexCliClient(model="openai/gpt-5.6-luna").with_workspace(tmp_path)
    scratch = tmp_path / ".codex_calls" / "call001" / "tmp"
    env = c.child_env(scratch)
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == str(scratch)
    assert scratch.is_dir()
    assert "TMPDIR" not in c.child_env() or c.child_env()["TMPDIR"] != str(scratch)


def test_config_forces_landlock_so_apply_patch_works_here():
    """Codex's bundled bubblewrap cannot create a user namespace on this host
    (AppArmor restricts unprivileged userns), which made every native
    file_change fail. The Landlock sandbox does not need one; measured:
    file_change 'failed' without the flag, 'completed' with it, same prompt,
    same model."""

    from meta_filter.llm.codex_cli import CodexCliClient

    cfg = CodexCliClient(model="openai/gpt-5.6-luna").render_config()
    assert "[features]" in cfg and "use_legacy_landlock = true" in cfg
    assert cfg.index("[features]") < cfg.index("[model_providers.")
