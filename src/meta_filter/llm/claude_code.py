"""Claude Code headless subprocess LLM client.

Instead of hitting `api.anthropic.com` directly, this client shells out
to the local `claude` CLI in non-interactive mode:

    claude -p --model <alias> --output-format json \\
        --system-prompt <system> \\
        --disable-slash-commands \\
        --exclude-dynamic-system-prompt-sections
    < prompt.txt

Why use this at all?

* Avoids storing a raw Anthropic API key when the user is already
  authenticated to Claude Code (OAuth). No key ever leaves the machine.
* Works against Claude Code's ToS-approved auth path — safer than
  reverse-engineering OAuth Bearer calls to the raw API.

Limitations:

* Sequential subprocess call per generation. Boot cost ~1-2s per call
  plus API latency. Threaded parallelism is possible but rate-limited by
  the user's Claude Code subscription.
* No temperature control exposed by the CLI; whatever the CLI defaults
  to is what you get. Deterministic seeded work is not supported.
* Runs in a *scratch cwd* so the current project's CLAUDE.md /
  agents / skills don't leak into the system prompt and inflate cost.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

from meta_filter.llm import pricing
from meta_filter.llm.agent_guard import ENV_AUDIT, ENV_ROOT, build_guard_settings
from meta_filter.llm.base import (
    LLMResponse,
    UsageLimitError,
    scrub_credentials,
    usage_limit_detail,
)

logger = logging.getLogger(__name__)


# Aliases the Claude Code CLI understands directly.
_ALIAS_SET = {"haiku", "sonnet", "opus"}

# Everything that can reach the filesystem. A static call answers from its
# prompt, so all of these are denied; an agentic call gets back only the ones
# it explicitly asks for.
_FILESYSTEM_TOOLS: tuple[str, ...] = (
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "NotebookEdit",
)
# Denied on every call: nothing here needs the network, and Task spawns
# sub-agents whose cost we cannot predict or attribute.
_REMOTE_TOOLS: tuple[str, ...] = ("Task", "WebFetch", "WebSearch")


OAUTH = "oauth"
API_KEY = "api_key"


def _auth_env(auth: str) -> dict[str, str]:
    """Build the child environment for the chosen authentication path.

    Under `oauth` the key is *dropped*. An operator who sources a shared
    `.env` for some unrelated credential (an HF token, say) would otherwise
    hand this subprocess a second identity, and the CLI silently authenticates
    as that one instead. It fails per call rather than at startup, so the run
    gets a long way in before anyone notices, paying for every call that
    exits 1 this way.

    Under `api_key` that same strip is the bug. A rented machine has no OAuth
    login to fall back on, so the key has to survive and CLAUDE_CODE_SIMPLE=1
    has to be set or the CLI will look for a login that is not there.
    """

    env = dict(os.environ)
    # Least privilege: the claude CLI needs none of the infrastructure
    # secrets, and an agentic call can run arbitrary python via Bash — with a
    # storage or hub token in its environment it could download the very files
    # verification isolation keeps off the machine. Strip every credential
    # but the CLI's own. Root can still read the parent's /proc environ, so
    # this raises the bar rather than perfecting it.
    scrub_credentials(
        env,
        prefixes=("HUGGING_FACE", "HF_", "OPENAI_", "OPENROUTER_", "GEMINI_", "GOOGLE_"),
        keep=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    )
    if auth == API_KEY:
        if not env.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "auth='api_key' but ANTHROPIC_API_KEY is unset — the CLI would "
                "fall back to an OAuth login this machine may not have"
            )
        env["CLAUDE_CODE_SIMPLE"] = "1"
        return env
    if auth != OAUTH:
        raise ValueError(f"unknown auth mode {auth!r}; expected {OAUTH!r} or {API_KEY!r}")
    env.pop("ANTHROPIC_API_KEY", None)
    return env


@dataclass
class ClaudeCodeHeadlessClient:
    """LLMClient backed by the local `claude` CLI in `-p` (print) mode."""

    model: str = "sonnet"
    # Runs in a scratch cwd so project CLAUDE.md / skills / agents don't
    # attach to every call. Defaults to a per-process temp dir.
    scratch_dir: Path | None = None
    # Claude CLI executable. Overridable for tests.
    binary: str = "claude"
    # Timeout for a single generate call, seconds. A proposer call emitting
    # 3 candidates (40k+ chars of JSON) has been observed past 15 min when
    # several tiers contend for the API. Meta-Harness budgets 2400s for the
    # comparable call; 30 min is our middle ground.
    timeout: float = 1800.0
    # Optional per-call max spend cap (USD). Guards against runaway
    # loops in the outer optimizer. None disables the cap.
    max_budget_usd: float | None = None
    # Agentic mode. Empty means the CLI answers from the prompt alone (the
    # static path). Populated, the model gets tools and can explore the
    # working directory before answering — Meta-Harness grants its proposer
    # Read/Glob/Grep/Agent/Write/Edit/Bash with no path allow-list, and the
    # sandbox is the cwd, not a permission table.
    allowed_tools: tuple[str, ...] = ()
    # Extra text appended to the system prompt. Agentic callers use this to
    # describe the directory layout the tools will see.
    append_system_prompt: str | None = None
    # Confine the call to this directory via a PreToolUse hook. Set by
    # `with_workspace`. None leaves the call unguarded, which is only safe
    # when it has no filesystem tools at all.
    guard_root: Path | None = None
    # Executables outside the guard root a Bash call may invoke — the
    # interpreter of the proposer's criterion-scoring tool and nothing else.
    # Empty (every run before the tool) renders the guard settings unchanged.
    guard_allow: tuple[str, ...] = ()
    # How the CLI authenticates. "oauth" uses the subscription login on this
    # machine and drops ANTHROPIC_API_KEY from the child environment so a
    # sourced `.env` cannot swap the identity underneath the call. "api_key"
    # is the inverse and exists for rented machines, which have no OAuth
    # login to use: it sets CLAUDE_CODE_SIMPLE=1 and passes the key through.
    # Read from META_FILTER_CLAUDE_AUTH when unset so a machine can select it
    # without every launch command growing a flag.
    auth: str = field(default_factory=lambda: os.environ.get("META_FILTER_CLAUDE_AUTH", "oauth"))
    _cumulative_calls: int = field(default=0, init=False)

    @property
    def audit_path(self) -> Path | None:
        return Path(self.guard_root) / ".agent_audit.jsonl" if self.guard_root else None

    @property
    def is_agentic(self) -> bool:
        return bool(self.allowed_tools)

    def with_workspace(self, workspace: Path) -> ClaudeCodeHeadlessClient:
        """Copy of this client rooted at `workspace` (the agent's sandbox).

        The workspace is also the guard boundary: `cwd` only anchors relative
        paths, so without the hook an absolute path walks straight out.
        """

        return replace(self, scratch_dir=Path(workspace), guard_root=Path(workspace))

    def __post_init__(self) -> None:
        if self.scratch_dir is None:
            self.scratch_dir = Path(tempfile.mkdtemp(prefix="claude-headless-"))
        else:
            self.scratch_dir = Path(self.scratch_dir)
            self.scratch_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- protocol

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> LLMResponse:
        _ = temperature, cache_system  # unsupported by the CLI

        args = [
            self.binary,
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--disable-slash-commands",
            "--exclude-dynamic-system-prompt-sections",
            "--dangerously-skip-permissions",
        ]
        if system:
            args += ["--system-prompt", system]
        # `--allowedTools` is a permission allow-list, and
        # `--dangerously-skip-permissions` already approves everything — so it
        # grants, it does not restrict. `--disallowedTools` is the only flag
        # that actually blocks. Deny whatever this call has no business using.
        if self.allowed_tools:
            args += ["--allowedTools", *self.allowed_tools]
        denied = tuple(t for t in _FILESYSTEM_TOOLS + _REMOTE_TOOLS if t not in self.allowed_tools)
        if denied:
            args += ["--disallowedTools", *denied]
        if self.append_system_prompt:
            args += ["--append-system-prompt", self.append_system_prompt]
        if self.max_budget_usd is not None:
            args += ["--max-budget-usd", str(self.max_budget_usd)]
        guard_env: dict[str, str] = {}
        if self.guard_root is not None:
            root = Path(self.guard_root).resolve()
            audit = self.audit_path
            assert audit is not None
            args += ["--settings", build_guard_settings(root, audit, self.guard_allow)]
            guard_env = {ENV_ROOT: str(root), ENV_AUDIT: str(audit)}

        child_env = _auth_env(self.auth)
        run_kwargs = dict(
            input=prompt,
            capture_output=True,
            text=True,
            cwd=str(self.scratch_dir),
            timeout=self.timeout,
            check=False,
            env={**child_env, **guard_env},
        )
        try:
            proc = subprocess.run(args, **run_kwargs)
        except subprocess.TimeoutExpired:
            logger.warning("claude headless timed out at %.0fs — retrying once", self.timeout)
            try:
                proc = subprocess.run(args, **run_kwargs)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"claude headless call timed out twice at {self.timeout}s"
                ) from exc

        # A spend cap or exhausted balance also surfaces as exit 1 with a
        # JSON body on stdout, and it is the opposite of the 529 below:
        # retrying is futile and every later call in the run will fail
        # identically. Raising a distinct type lets the optimizer stop on the
        # first one instead of spending iterations rediscovering it.
        if proc.returncode != 0 and (cap := _usage_limit_detail(proc.stdout)):
            raise UsageLimitError(f"claude headless refused: {cap}")

        # Transient server overload surfaces as exit 1 with a 529 body. Each
        # one would kill a whole proposer call, and the optimizer's only
        # containment is to skip the iteration. A short backoff-retry keeps a
        # multi-hour run from paying an iteration per blip; anything still
        # overloaded after two waits is a real outage and raises as before.
        overloaded = proc.returncode != 0 and (
            '"api_error_status":529' in (proc.stdout or "") or "Overloaded" in (proc.stdout or "")
        )
        for wait in (60, 120):
            if not overloaded:
                break
            logger.warning("claude headless hit 529 Overloaded — retrying in %ds", wait)
            time.sleep(wait)
            proc = subprocess.run(args, **run_kwargs)
            overloaded = proc.returncode != 0 and (
                '"api_error_status":529' in (proc.stdout or "")
                or "Overloaded" in (proc.stdout or "")
            )

        if proc.returncode != 0:
            # The CLI reports most failures as JSON on *stdout* and leaves
            # stderr empty, so a stderr-only message reads as
            # "claude headless exit 1:" and says nothing at all.
            detail = "\n".join(
                tail
                for stream in (proc.stderr, proc.stdout)
                if (tail := "\n".join((stream or "").strip().splitlines()[-5:]))
            )
            raise RuntimeError(f"claude headless exit {proc.returncode}: {detail or '(no output)'}")

        raw = proc.stdout.strip()
        if not raw:
            raise RuntimeError("claude headless returned empty stdout")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"claude headless produced non-JSON output: {raw[:200]}") from exc

        if payload.get("is_error"):
            raise RuntimeError(
                f"claude headless api error: {payload.get('api_error_status')} "
                f"result={payload.get('result', '')[:200]}"
            )

        text = str(payload.get("result", ""))
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
        prompt_tokens = input_tokens + cache_read + cache_write
        model_used = _extract_model_name(payload) or self.model
        cost_usd, cost_source = self._cost(
            payload, model_used, prompt_tokens, output_tokens, cache_read, cache_write
        )

        self._cumulative_calls += 1
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            model=model_used,
            calls=1,
            cost_usd=cost_usd,
            cost_source=cost_source,
            cached_tokens=cache_read,
        )

    def _cost(
        self,
        payload: dict,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read: int,
        cache_write: int,
    ) -> tuple[float, str]:
        """What the call cost, and whether that is measured or estimated.

        `total_cost_usd` is the CLI's own accounting, and it is only a bill
        when Anthropic is the one billing. Redirect the CLI at another gateway
        with `ANTHROPIC_BASE_URL` — which is how an Opus proposer runs on the
        same OpenRouter account as everything else — and the
        field keeps arriving, still computed from Anthropic's rate card, for a
        call Anthropic never charged for. Summing those into a run's spend
        double-counts the OpenRouter charge and hides the real one.
        """

        if _billed_by_anthropic():
            reported = payload.get("total_cost_usd")
            if reported is not None:
                return round(float(reported), 6), "harness"
        return pricing.cost_from_list_price(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )


def _billed_by_anthropic() -> bool:
    """True when the CLI's calls go to Anthropic itself.

    Unset means the CLI's own default, which is Anthropic. Anything else is
    a gateway, and the only safe assumption about a gateway is that it prices
    differently from the card the CLI is quoting.
    """

    base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    if not base:
        return True
    host = urlparse(base if "//" in base else f"//{base}").hostname or base
    return host.lower().endswith("anthropic.com")


def _extract_model_name(payload: dict) -> str | None:
    usage_by_model = payload.get("modelUsage") or {}
    if isinstance(usage_by_model, dict) and usage_by_model:
        return next(iter(usage_by_model))
    return None


def is_supported_alias(name: str) -> bool:
    return name in _ALIAS_SET


# Substrings the CLI puts in a failure body when the account is out of
# allowance rather than misconfigured. Matched on the message text, not on
# `"api_error_status":400`, because 400 also covers a malformed request —
# which *is* a configuration error and must keep its existing handling.
_USAGE_LIMIT_MARKERS = (
    "reached your specified API usage limits",
    "credit balance is too low",
    "usage limit reached",
)


def _usage_limit_detail(stdout: str | None) -> str:
    return usage_limit_detail(stdout, _USAGE_LIMIT_MARKERS)
