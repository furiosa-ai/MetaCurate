"""Codex CLI subprocess LLM client, routed through OpenRouter.

Same shape as `claude_code.ClaudeCodeHeadlessClient` — `generate`,
`with_workspace`, `allowed_tools`, `append_system_prompt`, `audit_path` — so
`AgenticProposer` and the agentic critics take one without modification. What
differs is everything underneath:

    codex exec -m openai/gpt-5.6-luna -s workspace-write \\
        -c 'approval_policy="never"' --skip-git-repo-check \\
        -C <workspace> --json
    < prompt.txt

Why a second harness at all? A Claude Code proposer is billed on an
Anthropic account whose monthly cap can take down whole runs, and a loop that
can only be driven by one vendor cannot be compared against itself. Routing Codex at
OpenRouter puts a second family of models behind the identical proposer
interface at list price.

Five things this has to get right:

* **The prompt goes on stdin.** Linux caps a single argument at
  MAX_ARG_STRLEN (128 KiB) whatever ARG_MAX says, and a proposer prompt runs
  to 149 KB. As an argv positional it fails with E2BIG before the process
  starts; `codex exec` with no positional reads the prompt from stdin, and
  the bytes delivered are identical.
* **The system prompt goes in AGENTS.md.** Codex has no `--system-prompt`
  flag. It reads `AGENTS.md` from the working root, so that is the delivery
  path, and the file is removed afterwards so it cannot leak into the next
  call or into a run's artifacts.
* **CODEX_HOME must be isolated.** Left at the default this would inherit the
  operator's Codex login, their MCP servers, and their session history, and
  would write this run's rollouts into their history.
* **`auth.json` must not exist** inside that home. With `auth_mode="apikey"`
  and no key it takes precedence over the provider's `env_key` and Codex dies
  with "API key auth is missing a key" before it sends anything.
* **The sandbox confines writes, not reads.** There is no `PreToolUse` hook to
  block on, so containment is audited after the fact by `harness_guard` and an
  out-of-boundary *write* raises.

Limitations, in the order they will bite:

* `allowed_tools` selects agentic-vs-static and nothing finer. Codex has no
  per-tool allow-list, so the individual names are not enforced; a static
  call is made safe by the read-only sandbox and a workspace holding only the
  prompt, not by naming tools.
* Reads outside the workspace cannot be prevented, only recorded. Nor can a
  write the model performs with a shell redirect: Codex's `workspace-write`
  sandbox treats the system temp directory as writable, and the captured
  reference call used exactly that (`cat > /tmp/iter009_diag.json`). Those
  show up as `inferred_writes_outside_workspace` and are reported, not
  raised on, because the evidence is a pattern match on a command line.
* No temperature control, same as the Claude Code client.
* Cost is list price from `LIST_PRICES` unless the harness reports a cost of
  its own, so a promotional or negotiated rate will read high.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from meta_filter.llm import harness_guard as hg
from meta_filter.llm.base import LLMResponse, UsageLimitError, scrub_credentials

logger = logging.getLogger(__name__)

# OpenRouter list prices, USD per token: (input, output, cache_read,
# cache_write), as OpenRouter's /api/v1/models listed them.
LIST_PRICES: dict[str, tuple[float, float, float, float]] = {
    "openai/gpt-5.6-sol": (2e-6, 10e-6, 0.2e-6, 0.0),
    "openai/gpt-5.6-sol-pro": (2e-6, 10e-6, 0.2e-6, 0.0),
    "openai/gpt-5.6-luna": (0.2e-6, 1.2e-6, 0.02e-6, 0.0),
    "openai/gpt-5.6-luna-pro": (0.2e-6, 1.2e-6, 0.02e-6, 0.0),
    "openai/gpt-5.6-terra": (2e-6, 12e-6, 0.2e-6, 0.0),
    "openai/gpt-5.6-terra-pro": (2e-6, 12e-6, 0.2e-6, 0.0),
    "minimax/minimax-m3": (0.3e-6, 1.2e-6, 0.06e-6, 0.0),
}


def read_provider_key(env_key: str, env_file: Path | None = None) -> str:
    """One credential, from the environment or a `KEY=value` line in `env_file`.

    The file is read line by line and never sourced: it also holds a dozen
    unrelated tokens and sourcing it would hand every one to the child.
    Shared with `meta_filter.score_tool`, which runs inside the agent's shell
    where the harness has stripped the provider variables from the
    environment and the file is the only route left to the key. The value
    is returned, never logged.

    Raises:
        KeyError: The key is in neither place; the message names both.
    """

    key = os.environ.get(env_key, "").strip()
    if key:
        return key
    path = Path(env_file) if env_file else Path.home() / ".env"
    if path.is_file():
        prefix = f"{env_key}="
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(prefix):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    raise KeyError(f"{env_key} is unset and not present in {path}")


# Sent as the resume leg's only message. The first leg's prompt is already in
# the thread, so repeating it would double the input bill.
RESUME_NUDGE = "Continue from where you left off and finish the task."

# Env this client must strip from the child. Codex needs exactly one
# credential and an agentic call can run arbitrary shell, so every other
# credential-looking variable (`base.scrub_credentials`), and the other LLM
# vendors' settings, are reach it should not have. The client's own key
# (`env_key`) is re-set below.
_VENDOR_PREFIXES: tuple[str, ...] = (
    "ANTHROPIC_",
    "CLAUDE_CODE_",
    "OPENAI_",
    "GEMINI_",
    "GOOGLE_",
    "HUGGING_FACE",
    "HF_",
)

# Substrings OpenRouter puts in a failure body when the account cannot pay,
# as opposed to being misconfigured. Same containment reason as the Claude
# Code client's markers: retrying a cap is futile and the run should stop on
# the first one rather than burn iterations rediscovering it.
_USAGE_LIMIT_MARKERS: tuple[str, ...] = (
    "insufficient credits",
    "requires more credits",
    "credit balance is too low",
    "402 payment required",
    "quota exceeded",
    "insufficient_quota",
)


@dataclass
class CodexResponse(LLMResponse):
    """`LLMResponse` plus what only a metered route can report.

    Subclassed rather than added to `LLMResponse` so the shared protocol stays
    provider-neutral; callers that only know `LLMResponse` are unaffected.
    """

    cost_usd: float = 0.0
    cost_source: str = "list_price"
    cached_tokens: int = 0
    tool_calls: int = 0
    thread_id: str | None = None
    transcript_dir: str | None = None
    reads_outside_workspace: tuple[str, ...] = ()
    # Shell redirects that look like they left the workspace. Reported rather
    # than raised on: see `harness_guard`'s docstring.
    inferred_writes_outside_workspace: tuple[str, ...] = ()


@dataclass
class CodexCliClient:
    """LLMClient backed by the local `codex` CLI in `exec` (headless) mode."""

    # An OpenRouter model slug, not a Codex alias: the provider block below
    # routes every call through OpenRouter's /v1/responses endpoint.
    model: str = "openai/gpt-5.6-luna"
    # The agent's working root. Codex's `-C`, the sandbox boundary, and where
    # AGENTS.md is delivered. Defaults to a per-process temp dir so a static
    # call cannot see a project tree.
    scratch_dir: Path | None = None
    binary: str = "codex"
    # One generate call, seconds, matching the Claude Code client so a swap
    # does not silently change the window a proposer call gets.
    timeout: float = 1800.0
    # Empty means static: no filesystem tools, a read-only sandbox, and a
    # workspace holding only the prompt. Populated means agentic, and the
    # names are advisory (see the module docstring).
    allowed_tools: tuple[str, ...] = ()
    # Appended to the system prompt. Delivered via AGENTS.md, since Codex has
    # no flag for it.
    append_system_prompt: str | None = None
    # The boundary the post-hoc audit is measured against, set by
    # `with_workspace`. None falls back to `scratch_dir`, which is the same
    # directory for every caller that goes through `with_workspace`.
    guard_root: Path | None = None
    # Isolated CLI home, so the operator's own Codex login, MCP servers and
    # session history are untouched. None creates one temp dir per client,
    # deliberately *outside* the workspace: in the agentic case the workspace
    # is the run directory, and Codex's rollouts are hundreds of KB per call
    # that would end up mirrored into the run's backed-up artifacts.
    codex_home: Path | None = None
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    env_key: str = "OPENROUTER_API_KEY"
    # Where to look for `env_key` when it is absent from the environment.
    # Read line by line, never sourced: this file also holds a dozen
    # unrelated tokens and sourcing it would hand every one to the child.
    env_file: Path | None = None
    # Per-call artifacts (leg logs, tool_calls.json). None puts them in a
    # dot-directory under the workspace.
    transcript_dir: Path | None = None
    prices: dict[str, tuple[float, float, float, float]] | None = None
    # Two keys of the workspace-write sandbox, rendered into
    # `[sandbox_workspace_write]` and both default off, so a client built
    # without them gets the plain workspace-write sandbox. `network_access` is
    # opened for the proposer's criterion-scoring tool
    # (`meta_filter.score_tool`), which has to reach OpenRouter; with it the
    # Codex proposer as a whole can reach the network for that call.
    # `writable_roots` is kept as a capability but the tool does NOT use it:
    # it stages its cells inside the run directory and the optimizer merges
    # them, because the run's scorer cache also holds the hidden selection
    # rows' cells. Checked with `codex sandbox linux --full-auto` and a config
    # file carrying these keys: curl to openrouter.ai answers 200 and a write
    # under a listed root lands, where both are refused without them.
    network_access: bool = False
    writable_roots: tuple[Path, ...] = ()
    _calls: int = field(default=0, init=False)
    _home: Path | None = field(default=None, init=False)

    # ------------------------------------------------------------- interface

    @property
    def audit_path(self) -> Path | None:
        root = self.guard_root or self.scratch_dir
        return Path(root) / ".agent_audit.jsonl" if root else None

    @property
    def is_agentic(self) -> bool:
        return bool(self.allowed_tools)

    @property
    def sandbox_mode(self) -> str:
        return "workspace-write" if self.is_agentic else "read-only"

    def with_workspace(self, workspace: Path) -> CodexCliClient:
        """Copy of this client rooted at `workspace` (the agent's sandbox).

        The workspace is both Codex's `-C` and the audit boundary. Unlike the
        Claude Code client this cannot *enforce* the boundary — `-C` only
        anchors relative paths and the sandbox only stops writes — so the
        copy's job is to make the boundary knowable, not to hold it.
        """

        return replace(self, scratch_dir=Path(workspace), guard_root=Path(workspace))

    def __post_init__(self) -> None:
        if self.scratch_dir is None:
            self.scratch_dir = Path(tempfile.mkdtemp(prefix="codex-headless-"))
        else:
            self.scratch_dir = Path(self.scratch_dir)
            self.scratch_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- plumbing

    def _price_table(self) -> dict[str, tuple[float, float, float, float]]:
        return self.prices if self.prices is not None else LIST_PRICES

    def _api_key(self) -> str:
        """The provider credential, from the environment or `env_file`."""

        try:
            return read_provider_key(self.env_key, self.env_file)
        except KeyError as exc:
            raise RuntimeError(
                f"{exc.args[0]} — codex would fall back to a login this machine may not have"
            ) from None

    def child_env(self, scratch: Path | None = None) -> dict[str, str]:
        """The subprocess environment: one credential, an isolated CLI home.

        `scratch`, when given, becomes TMPDIR/TMP/TEMP. The workspace-write
        sandbox lets the agent write anywhere under /tmp, which two runs at
        once share (two agents both writing `/tmp/diagnosis.json`, or one
        leaving `/tmp/eval_policy.py` behind for any later agent to find). A
        per-call directory inside the workspace keeps
        scratch inside the audit boundary; the AGENTS.md note tells the agent
        to use it.
        """

        env = scrub_credentials(dict(os.environ), prefixes=_VENDOR_PREFIXES)
        env[self.env_key] = self._api_key()
        home = self._codex_home()
        env["CODEX_HOME"] = str(home)
        # Codex also reads XDG for caches and logs; point those at the same
        # isolated tree so nothing lands in the operator's ~/.config.
        for var, sub in (
            ("XDG_CONFIG_HOME", "config"),
            ("XDG_DATA_HOME", "data"),
            ("XDG_CACHE_HOME", "cache"),
            ("XDG_STATE_HOME", "state"),
        ):
            env[var] = str(home / sub)
        if scratch is not None:
            Path(scratch).mkdir(parents=True, exist_ok=True)
            for var in ("TMPDIR", "TMP", "TEMP"):
                env[var] = str(scratch)
        return env

    def _codex_home(self) -> Path:
        if self.codex_home is not None:
            return Path(self.codex_home)
        if self._home is None:
            # Not under /tmp: in workspace-write mode Codex 0.118 places its
            # sandbox helper binaries under CODEX_HOME and refuses to create
            # them beneath a temporary dir ("Refusing to create helper
            # binaries under temporary dir"), after which the exec dies with
            # ENOENT. The read-only smoke never needed the helpers, which is
            # why /tmp looked fine until the first agentic call.
            root = Path.home() / ".cache" / "meta_filter" / "codex-homes"
            root.mkdir(parents=True, exist_ok=True)
            self._home = Path(tempfile.mkdtemp(prefix="codex-home-", dir=root))
        return self._home

    def _next_call_index(self) -> int:
        """1 + the highest `callNNN` already under the transcript root."""

        root = self._transcript_root()
        matches = (re.match(r"call(\d+)$", p.name) for p in root.glob("call*"))
        found = [int(m.group(1)) for m in matches if m]
        return (max(found) if found else 0) + 1

    def _transcript_root(self) -> Path:
        if self.transcript_dir is not None:
            return Path(self.transcript_dir)
        root = self.guard_root or self.scratch_dir
        assert root is not None
        return Path(root) / ".codex_calls"

    def render_config(self) -> str:
        """`config.toml` wiring Codex to the OpenRouter model provider.

        `wire_api = "responses"` is not cosmetic: OpenRouter serves both the
        chat-completions and the responses shape, and Codex 0.118 speaks only
        the latter.
        """

        sandbox = self._sandbox_section()
        return (
            f'model_provider = "{self.provider}"\n'
            'approval_policy = "never"\n'
            f'sandbox_mode = "{self.sandbox_mode}"\n'
            "\n"
            # Codex 0.118 runs the native `apply_patch` tool inside its bundled
            # bubblewrap, which needs an unprivileged user namespace. Ubuntu
            # 24.04 denies that to binaries without an AppArmor profile
            # (`kernel.apparmor_restrict_unprivileged_userns = 1`), so every
            # file_change failed ("bwrap: loopback: Failed RTM_NEWADDR" /
            # "setting up uid map: Permission denied") while shell commands,
            # sandboxed by Landlock, kept working. The agent then loses its
            # edit tool and falls back to whole-file heredoc writes — which
            # the kept-lines check classifies as rewrites. Landlock for
            # everything fixes it and
            # keeps the write boundary (workspace + TMPDIR).
            "[features]\n"
            "use_legacy_landlock = true\n"
            "\n"
            f"{sandbox}[model_providers.{self.provider}]\n"
            f'name = "{self.provider}"\n'
            f'base_url = "{self.base_url}"\n'
            f'env_key = "{self.env_key}"\n'
            'wire_api = "responses"\n'
        )

    def _sandbox_section(self) -> str:
        """`[sandbox_workspace_write]`, only when something in it is non-default.

        Emitted only for an agentic client: a static call runs read-only and
        the section would be dead text there.
        """

        if not self.is_agentic or not (self.network_access or self.writable_roots):
            return ""
        lines = ["[sandbox_workspace_write]"]
        if self.network_access:
            lines.append("network_access = true")
        if self.writable_roots:
            roots = ", ".join(json.dumps(str(Path(r).resolve())) for r in self.writable_roots)
            lines.append(f"writable_roots = [{roots}]")
        return "\n".join(lines) + "\n\n"

    def _prepare_home(self) -> Path:
        home = self._codex_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.toml").write_text(self.render_config(), encoding="utf-8")
        # See the module docstring: present-but-keyless auth.json wins over
        # the provider's env_key and kills the call before it is sent.
        (home / "auth.json").unlink(missing_ok=True)
        return home

    def argv(self, workspace: Path, resume_thread: str | None = None) -> list[str]:
        args = [self.binary, "exec"]
        if resume_thread:
            args += ["resume", resume_thread]
        args += [
            "-m",
            self.model,
            "-s",
            self.sandbox_mode,
            "-c",
            'approval_policy="never"',
            "--skip-git-repo-check",
            "-C",
            str(Path(workspace).resolve()),
            "--json",
        ]
        if resume_thread:
            args.append(RESUME_NUDGE)
        return args

    # -------------------------------------------------------------- protocol

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> CodexResponse:
        _ = temperature, cache_system  # unsupported by the CLI

        # Number the call from what is on disk, not from an instance counter:
        # `with_workspace` hands the proposer a fresh copy per call, whose
        # counter starts at 0 again, so every call would overwrite call001
        # and only the last transcript would survive.
        self._calls = self._next_call_index()
        call_dir = self._transcript_root() / f"call{self._calls:03d}"
        call_dir.mkdir(parents=True, exist_ok=True)

        # A static call gets its own empty workspace. Denying tools by name is
        # not available here, so "answers from its prompt alone" has to be
        # made true by there being nothing else to read.
        if self.is_agentic:
            workspace = Path(self.scratch_dir)
        else:
            workspace = call_dir / "ws"
            workspace.mkdir(parents=True, exist_ok=True)

        prompt_file = call_dir / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        self._prepare_home()
        scratch = call_dir / "tmp"
        env = self.child_env(scratch)
        scratch_note = (
            f"Scratch files (patches, helper scripts, notes) go under "
            f"`{scratch}` — it is $TMPDIR for this call and lies inside the "
            f"workspace. Do not write to /tmp or anywhere else outside the "
            f"workspace; other runs share this machine and every such write "
            f"is logged against this call."
        )
        appended = "\n\n".join(p for p in (self.append_system_prompt, scratch_note) if p)

        with _agents_md(workspace, system, appended):
            legs = self._run_legs(workspace, call_dir, prompt_file, env)

        parsed = parse_legs(sorted(call_dir.glob("leg*.jsonl")))
        if not parsed["usage_seen"]:
            # A killed leg never emits `turn.completed`, so the stream has no
            # usage. The rollout does: it is written incrementally and
            # survives the SIGKILL.
            rollout = parse_rollouts(self._codex_home(), parsed["thread_ids"])
            if rollout["usage_seen"]:
                parsed["usage"] = rollout["usage"]
                parsed["usage_seen"] = True
            if not parsed["text"]:
                parsed["text"] = rollout["text"]
            if not parsed["calls"]:
                parsed["calls"] = rollout["calls"]

        (call_dir / "tool_calls.json").write_text(
            json.dumps(parsed["calls"], indent=2), encoding="utf-8"
        )
        (call_dir / "legs.json").write_text(json.dumps(legs, indent=2), encoding="utf-8")

        records = hg.audit_records(parsed["calls"], self._boundary_root(workspace))
        audit = self.audit_path
        if audit is not None:
            hg.append_audit(audit, records)
        (call_dir / "boundary_audit.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        reads = hg.reads_outside(records)
        if reads:
            logger.warning(
                "codex call %d read %d path(s) outside the workspace; first: %s",
                self._calls,
                len(reads),
                reads[0],
            )
        redirects = hg.inferred_writes_outside(records)
        if redirects:
            logger.warning(
                "codex call %d looks like it redirected output outside the "
                "workspace (%d path(s), first: %s) — not raising, the evidence "
                "is a command-line match",
                self._calls,
                len(redirects),
                redirects[0],
            )

        self._check_failure(legs, call_dir, parsed)

        escaped = hg.writes_outside(records)
        if escaped:
            raise hg.BoundaryViolation(
                f"codex exec wrote outside its workspace "
                f"{self._boundary_root(workspace)}: {', '.join(escaped[:5])}"
            )

        usage = parsed["usage"]
        cost, source = self._cost(usage, parsed["harness_cost_usd"])
        return CodexResponse(
            text=parsed["text"],
            prompt_tokens=usage["input"] + usage["cache_read"],
            completion_tokens=usage["output"],
            model=self.model,
            calls=1,
            cost_usd=cost,
            cost_source=source,
            cached_tokens=usage["cache_read"],
            tool_calls=len(parsed["calls"]),
            thread_id=(parsed["thread_ids"] or [None])[-1],
            transcript_dir=str(call_dir),
            reads_outside_workspace=tuple(reads),
            inferred_writes_outside_workspace=tuple(redirects),
        )

    def _boundary_root(self, workspace: Path) -> Path:
        return Path(self.guard_root) if self.guard_root else Path(workspace)

    def _run_legs(
        self, workspace: Path, call_dir: Path, prompt_file: Path, env: dict[str, str]
    ) -> list[dict]:
        """The first leg, plus one resume leg if it was killed at `timeout`."""

        legs = [
            _run_leg(
                self.argv(workspace),
                cwd=workspace,
                env=env,
                log=call_dir / "leg1.jsonl",
                timeout=self.timeout,
                stdin_file=prompt_file,
            )
        ]
        if not legs[0]["timed_out"]:
            return legs

        thread = (parse_legs([call_dir / "leg1.jsonl"])["thread_ids"] or [None])[-1]
        if thread is None:
            raise RuntimeError(
                f"codex exec timed out at {self.timeout:.0f}s before starting a "
                f"thread, so there is nothing to resume"
            )
        logger.warning(
            "codex exec timed out at %.0fs — resuming thread %s once",
            self.timeout,
            thread,
        )
        legs.append(
            _run_leg(
                self.argv(workspace, resume_thread=thread),
                cwd=workspace,
                env=env,
                log=call_dir / "leg2.jsonl",
                timeout=self.timeout,
                stdin_file=None,
            )
        )
        return legs

    def _check_failure(self, legs: list[dict], call_dir: Path, parsed: dict) -> None:
        last = legs[-1]
        tail = _log_tail(call_dir / str(last["log"]))
        if cap := _usage_limit_detail(tail):
            raise UsageLimitError(f"codex exec refused: {cap}")
        if last["timed_out"]:
            if parsed["text"]:
                logger.warning(
                    "codex exec timed out on both legs but produced text; "
                    "returning the partial answer"
                )
                return
            raise RuntimeError(f"codex exec timed out twice at {self.timeout:.0f}s with no output")
        if last["returncode"] != 0:
            raise RuntimeError(f"codex exec exit {last['returncode']}: {tail or '(no output)'}")
        if not parsed["text"] and not parsed["calls"]:
            raise RuntimeError(
                f"codex exec produced neither text nor tool calls: {tail or '(no output)'}"
            )

    def _cost(self, usage: dict[str, int], harness_cost: float | None) -> tuple[float, str]:
        if harness_cost:
            return round(float(harness_cost), 6), "harness"
        rate = self._price_table().get(self.model)
        if rate is None:
            logger.warning("no list price for %s; reporting cost 0.0", self.model)
            return 0.0, "unpriced"
        p_in, p_out, p_read, p_write = rate
        total = (
            usage["input"] * p_in
            + usage["output"] * p_out
            + usage["cache_read"] * p_read
            + usage["cache_write"] * p_write
        )
        return round(total, 6), "list_price"


# ------------------------------------------------------------------ AGENTS.md


class _agents_md:
    """Deliver the system prompt as AGENTS.md, then put the tree back.

    A context manager because the removal is the load-bearing half. Codex
    reads AGENTS.md from the working root on every invocation, so one left
    behind silently becomes the system prompt of the *next* call into the same
    workspace — and in the agentic case that workspace is the run directory,
    whose files are the run's evidence.
    """

    def __init__(self, workspace: Path, system: str | None, append: str | None) -> None:
        self.path = Path(workspace) / "AGENTS.md"
        self.body = "\n\n".join(part for part in (system, append) if part)
        self.previous: str | None = None
        self.existed = False

    def __enter__(self) -> Path | None:
        if not self.body:
            return None
        if self.path.exists():
            self.existed = True
            self.previous = self.path.read_text(encoding="utf-8")
        self.path.write_text(self.body, encoding="utf-8")
        return self.path

    def __exit__(self, *exc) -> None:
        if not self.body:
            return
        try:
            if self.existed and self.previous is not None:
                self.path.write_text(self.previous, encoding="utf-8")
            else:
                self.path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not clean up %s", self.path)


# -------------------------------------------------------------- subprocess


def _run_leg(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
    timeout: float,
    stdin_file: Path | None,
) -> dict:
    """One CLI invocation, killed at `timeout`, with its stream kept either way.

    The kill has to take the whole process group. Codex spawns shells for the
    model's commands, and a bare `kill()` leaves those children holding the
    pipe, so the read never ends and the timeout never returns.
    """

    started = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as out:
        stdin = stdin_file.open("rb") if stdin_file else subprocess.DEVNULL
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                env=env,
                stdin=stdin,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait(timeout=30)
        finally:
            if stdin_file and stdin is not subprocess.DEVNULL:
                stdin.close()
    return {
        "argv": [a if len(a) < 200 else f"<{len(a)} chars>" for a in argv],
        "timed_out": timed_out,
        "returncode": proc.returncode,
        "wall_seconds": round(time.time() - started, 1),
        "log": log.name,
    }


def _log_tail(path: Path, lines: int = 5) -> str:
    if not path.is_file():
        return ""
    body = path.read_text(encoding="utf-8", errors="replace").strip()
    return "\n".join(body.splitlines()[-lines:])[:2000]


def _usage_limit_detail(body: str) -> str:
    """The cap sentence from a failure body, or `""` if it is not a cap."""

    low = (body or "").lower()
    if not any(marker in low for marker in _USAGE_LIMIT_MARKERS):
        return ""
    for line in reversed((body or "").strip().splitlines()):
        if any(marker in line.lower() for marker in _USAGE_LIMIT_MARKERS):
            return line.strip()[:400]
    return (body or "").strip()[:400]


# ------------------------------------------------------------- log parsing


def _jsonl(path: Path):
    if not Path(path).is_file():
        return
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _empty_usage() -> dict[str, int]:
    return {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}


def _add_usage(usage: dict[str, int], payload: dict) -> None:
    """Fold one Codex usage report in, un-double-counting the cached slice.

    `input_tokens` is the whole prompt *including* what was served from
    cache, so the cached part is subtracted before the rest is priced at the
    full input rate. Skipping this overstates a cache-heavy call by roughly
    its hit rate, and a proposer call runs at 90%+.
    """

    cached = int(payload.get("cached_input_tokens", 0) or 0)
    usage["input"] += max(0, int(payload.get("input_tokens", 0) or 0) - cached)
    usage["cache_read"] += cached
    usage["output"] += int(payload.get("output_tokens", 0) or 0)


def parse_legs(logs: list[Path]) -> dict:
    """Text, usage, thread ids and tool calls from the `--json` streams.

    The stream is authoritative for a leg that finished: `turn.completed`
    carries the cumulative token usage, `file_change` items name every path
    written with its kind, and `command_execution` items carry the shell
    command verbatim. Both of the latter are what the boundary audit reads.
    """

    calls: list[dict] = []
    texts: list[str] = []
    thread_ids: list[str] = []
    usage = _empty_usage()
    usage_seen = False
    harness_cost = 0.0

    for log in logs:
        for rec in _jsonl(log):
            kind = rec.get("type")
            if kind == "thread.started" and rec.get("thread_id"):
                thread_ids.append(str(rec["thread_id"]))
                continue
            if kind == "turn.completed":
                turn_usage = rec.get("usage") or {}
                if turn_usage:
                    _add_usage(usage, turn_usage)
                    usage_seen = True
                if turn_usage.get("cost") or turn_usage.get("total_cost_usd"):
                    harness_cost += float(
                        turn_usage.get("cost") or turn_usage.get("total_cost_usd")
                    )
                continue
            if kind != "item.completed":
                continue
            item = rec.get("item") or {}
            item_type = str(item.get("type", ""))
            if item_type == "agent_message" and item.get("text"):
                texts.append(str(item["text"]))
            elif item_type == "command_execution":
                calls.append(
                    {
                        "tool": "command_execution",
                        "target": str(item.get("command", ""))[:4000],
                        "writes": [],
                        "shell": True,
                        "exit_code": item.get("exit_code"),
                    }
                )
            elif item_type == "file_change":
                changes = item.get("changes") or []
                calls.append(
                    {
                        "tool": "file_change",
                        "target": " ".join(str(c.get("path", "")) for c in changes)[:4000],
                        # `status: failed` means the sandbox refused the write,
                        # so it is a read-shaped attempt, not a landed write.
                        "writes": [
                            str(c.get("path", ""))
                            for c in changes
                            if item.get("status") != "failed"
                        ],
                        "status": item.get("status"),
                    }
                )

    return {
        "calls": calls,
        "text": "\n\n".join(texts),
        "usage": usage,
        "usage_seen": usage_seen,
        "harness_cost_usd": round(harness_cost, 6) if harness_cost else 0.0,
        "thread_ids": thread_ids,
    }


def parse_rollouts(codex_home: Path, thread_ids: list[str]) -> dict:
    """The same fields, recovered from Codex's own session rollouts.

    Only needed when a leg was killed: the `--json` stream then has no
    `turn.completed` and so no usage at all, while the rollout is appended as
    the turn runs and survives the SIGKILL.
    """

    calls: list[dict] = []
    texts: list[str] = []
    usage = _empty_usage()
    usage_seen = False
    wanted = {t for t in thread_ids if t}
    sessions = Path(codex_home) / "sessions"
    if not sessions.is_dir():
        return {
            "calls": calls,
            "text": "",
            "usage": usage,
            "usage_seen": False,
            "harness_cost_usd": 0.0,
            "thread_ids": thread_ids,
        }

    for path in sorted(sessions.rglob("rollout-*.jsonl")):
        if wanted and not any(t in path.name for t in wanted):
            continue
        last_total: dict | None = None
        for rec in _jsonl(path):
            payload = rec.get("payload") or {}
            kind = payload.get("type")
            if rec.get("type") == "response_item" and kind == "function_call":
                try:
                    args = json.loads(payload.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                target = (
                    args.get("cmd")
                    or args.get("command")
                    or args.get("path")
                    or args.get("file_path")
                    or args.get("input")
                    or ""
                )
                if isinstance(target, list):
                    target = " ".join(str(t) for t in target)
                writes = []
                if args.get("file_path") and payload.get("name") in (
                    "write_file",
                    "apply_patch",
                    "edit_file",
                ):
                    writes = [str(args["file_path"])]
                calls.append(
                    {
                        "tool": str(payload.get("name", "?")),
                        "target": f"{target} {args.get('workdir') or ''}".strip()[:4000],
                        "writes": writes,
                    }
                )
            elif (
                rec.get("type") == "response_item"
                and kind == "message"
                and payload.get("role") == "assistant"
            ):
                for block in payload.get("content") or []:
                    if isinstance(block, dict) and block.get("text"):
                        texts.append(str(block["text"]))
            elif rec.get("type") == "event_msg" and kind == "token_count":
                total = (payload.get("info") or {}).get("total_token_usage") or {}
                if total:
                    last_total = total
        if last_total:
            _add_usage(usage, last_total)
            usage_seen = True

    return {
        "calls": calls,
        "text": "\n\n".join(texts),
        "usage": usage,
        "usage_seen": usage_seen,
        "harness_cost_usd": 0.0,
        "thread_ids": thread_ids,
    }
