"""PreToolUse guard that confines an agentic call to one directory.

Prompt-level boundaries do not work. The agentic proposer was twice told
to stay inside its run directory and twice wrote its candidates into a
different run's tree, the second time crossing from one output root into
another entirely. `cwd` anchors relative paths, it does not fence absolute
ones, so nothing in the prompt or the directory layout can hold.

This module is the enforcement point. It runs as a `PreToolUse` hook
injected via `claude --settings`, receives the pending tool call on stdin,
and exits 2 to block anything reaching outside the boundary. Every decision
is appended to an audit log so the caller can tell afterwards whether the
agent tried.

Kept dependency-free (stdlib only) and invoked by absolute file path, so it
does not need the package on `sys.path` inside the agent's subprocess.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

ENV_ROOT = "META_FILTER_GUARD_ROOT"
ENV_AUDIT = "META_FILTER_GUARD_AUDIT"
# `os.pathsep`-joined absolute paths a Bash command may name although they
# lie outside the boundary. Exactly one use: the interpreter the proposer's
# criterion-scoring tool runs under (`meta_filter.score_tool`), which lives
# in the repository's virtualenv and would otherwise be blocked as an
# absolute escape. A token must EQUAL an allowed path; prefixes and
# directories are not honoured, so allowing the interpreter allows nothing
# else in the venv.
ENV_ALLOW = "META_FILTER_GUARD_ALLOW"

# Tools whose inputs name a path outright. These are fully checkable.
_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "NotebookEdit": ("notebook_path",),
    "Glob": ("path",),
    "Grep": ("path",),
}


def build_guard_settings(
    root: Path, audit_path: Path, allow: tuple[str, ...] | list[str] = ()
) -> str:
    """JSON for `claude --settings` wiring this module in as a PreToolUse hook.

    Auto-memory is disabled here rather than in the hook because it is not a
    tool call: Claude Code injects remembered context directly, so PreToolUse
    never sees it and the boundary below cannot cover it. Left on, one run's
    findings would carry into the next and the runs would stop being
    independent samples.

    `allow` names executables outside the boundary a Bash command may invoke
    (see `ENV_ALLOW`); empty renders the settings every run before the
    scoring tool used.
    """

    matcher = "|".join((*_PATH_FIELDS, "Bash"))
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))}"
    env = {
        ENV_ROOT: str(Path(root).resolve()),
        ENV_AUDIT: str(Path(audit_path).resolve()),
    }
    if allow:
        env[ENV_ALLOW] = os.pathsep.join(str(Path(a).resolve()) for a in allow)
    return json.dumps(
        {
            "autoMemoryEnabled": False,
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": matcher,
                        "hooks": [{"type": "command", "command": command}],
                    }
                ]
            },
            "env": env,
        }
    )


def is_contained(candidate: str, root: Path, cwd: Path) -> bool:
    """True if `candidate` resolves to `root` or something beneath it.

    Resolves symlinks, so a link planted inside the boundary cannot be used
    to step outside it.
    """

    try:
        resolved = Path(os.path.expanduser(candidate))
        if not resolved.is_absolute():
            resolved = cwd / resolved
        resolved = resolved.resolve()
    except (OSError, ValueError):
        return False
    return resolved == root or root in resolved.parents


def path_violations(tool_name: str, tool_input: dict, root: Path, cwd: Path) -> list[str]:
    out = []
    for field in _PATH_FIELDS.get(tool_name, ()):
        value = tool_input.get(field)
        if isinstance(value, str) and value and not is_contained(value, root, cwd):
            out.append(f"{field}={value}")
    return out


def bash_violations(
    command: str, root: Path, cwd: Path, allow: tuple[str, ...] | list[str] = ()
) -> list[str]:
    """Path-like tokens in a shell command that point outside `root`.

    A shell command is an arbitrary string, so unlike the tools above this
    can never be exhaustive — treat it as a strong filter, not a proof.
    Every ambiguous case resolves to "block": a false positive costs the
    agent one retry, a false negative corrupts another experiment. `allow`
    is the one exception: a token that resolves to exactly one of those
    paths (the scoring tool's interpreter) passes.
    """

    # Command substitution can hide any path at all inside a token that
    # looks harmless, so there is nothing useful left to inspect.
    if "$(" in command or "`" in command:
        return ["command substitution"]

    try:
        tokens = shlex.split(command)
    except ValueError:
        return ["unparseable command"]

    allowed = {str(Path(a).resolve()) for a in allow if a}
    out = []
    for token in tokens:
        candidate = token
        if candidate.startswith("-"):
            # `--include=/etc/x` hides a path behind a flag; a bare flag
            # like `-la` has none.
            _, sep, rhs = candidate.partition("=")
            if not sep:
                continue
            candidate = rhs
        if not candidate:
            continue
        if ".." not in candidate and "/" not in candidate:
            continue  # a bare word: subcommand, pattern, or filename in cwd
        if allowed and _resolved(candidate) in allowed:
            continue
        if not is_contained(candidate, root, cwd):
            out.append(token)
    return out


def _resolved(candidate: str) -> str:
    try:
        return str(Path(os.path.expanduser(candidate)).resolve())
    except (OSError, ValueError):
        return ""


def _audit(record: dict) -> None:
    path = os.environ.get(ENV_AUDIT)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass  # auditing must never break the run


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # not our event shape; do not interfere

    root_env = os.environ.get(ENV_ROOT)
    if not root_env:
        return 0  # guard not configured for this call

    root = Path(root_env).resolve()
    cwd = Path(event.get("cwd") or os.getcwd())
    tool_name = str(event.get("tool_name", ""))
    tool_input = event.get("tool_input") or {}

    if tool_name == "Bash":
        allow = tuple(p for p in os.environ.get(ENV_ALLOW, "").split(os.pathsep) if p)
        bad = bash_violations(str(tool_input.get("command", "")), root, cwd, allow)
    else:
        bad = path_violations(tool_name, tool_input, root, cwd)

    _audit(
        {
            "tool": tool_name,
            "decision": "deny" if bad else "allow",
            "violations": bad,
            "input": tool_input if bad else None,
        }
    )
    if not bad:
        return 0

    print(
        f"Blocked: {tool_name} refers to {', '.join(bad)}, which is outside "
        f"the working directory {root}. Everything you need is inside it. "
        f"Retry using a path under {root}, and never use `..`.",
        file=sys.stderr,
    )
    return 2  # exit 2 blocks the call and shows stderr to the model


if __name__ == "__main__":
    sys.exit(main())
