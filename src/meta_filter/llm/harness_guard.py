"""Post-hoc boundary audit for CLI harnesses that cannot be fenced pre-hoc.

`agent_guard` *blocks*: Claude Code exposes a `PreToolUse` hook, so a tool
call that names a path outside the workspace never runs. Codex has no such
hook. Its `workspace-write` sandbox confines writes to the working root and
leaves reads entirely open, and neither half is inspectable before the fact.

So this module records instead of blocking. It reads a harness's own tool log
after the call and appends one record per out-of-bounds path to the same
`.agent_audit.jsonl` the blocking guard writes, in the same shape, so a
caller that already reads that file (`agentic_proposer._guard_denials`) keeps
working unchanged.

The `decision` field carries the only distinction that matters afterwards:

* `deny`  — a *write* landed outside the boundary, on the harness's own
  word: it reported the file as changed. The sandbox was supposed to make
  this impossible, so it is a containment failure and the client raises on
  it. "deny" is retroactive here: nothing was stopped.
* `warn`  — a *read* reached outside the boundary, or a shell command looks
  like it redirected output there. Expected, not preventable, and worth
  having on the record because it is how one run's evidence leaks into
  another's proposal. A shell redirect is `warn` rather than `deny` because
  the evidence is a substring match on an arbitrary command line: raising on
  it would let one bad guess abort a paid multi-hour call.
* `allow` — the path is an ancestor of the boundary root, produced when a long
  command was truncated mid-path. Nothing was reached through it.

Both `warn` kinds are on the record and on the response, so a caller that
wants to be stricter than "raise on a reported write" has the evidence.

Kept free of any dependency beyond `agent_guard.is_contained` so both guards
agree on what "inside" means, symlinks included.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from meta_filter.llm.agent_guard import is_contained

DENY = "deny"
WARN = "warn"
ALLOW = "allow"

# Absolute-looking tokens. A leading `/` only starts an absolute path when
# nothing of the token precedes it, so the character before it must be a
# separator. `run_harness2.py` used a bare `(?<![\\w.-])` here and that reads
# `iterations/{006,007}/proposer/x.json` as the absolute path
# `/proposer/x.json`, because `}` is neither a word character nor a dot —
# a brace expansion inside a relative path becomes a phantom escape.
_ABS_CHARS = "\\w./@+-"
_SEPARATORS = "\\s'\"=(),\\[<>|&;:`$"

# Read-only system paths every shell command names in passing. `grep` resolves
# to /usr/bin/grep, a python subprocess reads /usr/lib, and flagging those
# buries the one path that matters under hundreds that do not.
BENIGN_PREFIXES: tuple[str, ...] = (
    "/dev/null",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/tty",
    "/dev/urandom",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/lib/",
    "/lib64/",
    "/opt/",
    "/proc/self",
    "/sys/",
    "/etc/localtime",
    "/etc/ssl",
    "/etc/resolv.conf",
)


_ABS = re.compile(rf"(?:^|(?<=[{_SEPARATORS}]))/[{_ABS_CHARS}]{{2,}}")

# `> path`, `>>path`, `| tee path`: the three ways a shell command names a
# destination it is about to write. Not exhaustive and not meant to be — see
# the module docstring on why a hit here warns instead of raising.
_REDIRECT = re.compile(rf"(?:>>?|\btee\b(?:\s+-a)?)\s*['\"]?(/[{_ABS_CHARS}]{{2,}})")


def absolute_paths(blob: str) -> list[str]:
    """Distinct absolute path tokens in a free-form string, sorted."""

    return sorted(set(_ABS.findall(blob or "")))


def shell_write_targets(command: str) -> list[str]:
    """Absolute paths a shell command appears to redirect output into."""

    return sorted(set(_REDIRECT.findall(command or "")))


def classify(path: str, root: Path) -> str | None:
    """`None` if `path` is inside `root` or benign, else `"outside"`/`"prefix"`.

    A `prefix` hit names an ancestor of `root`. That happens when the harness
    truncated a long command mid-path, or when the model emitted the workspace
    path with a stray space in it — the token is a fragment of the boundary,
    not a step beyond it.
    """

    root = Path(root).resolve()
    root_s = str(root)
    if path.startswith(root_s) or path.startswith(BENIGN_PREFIXES):
        return None
    if is_contained(path, root, root):
        return None
    return "prefix" if root_s.startswith(path.rstrip("/")) else "outside"


def audit_records(calls: Iterable[dict], root: Path) -> list[dict]:
    """One audit record per out-of-bounds path named by a tool call.

    Each call is `{"tool", "target", "writes", "shell"}`: `target` is the
    free-form blob (a shell command, a file argument) mined for path tokens,
    `writes` is what the harness itself reported as modified, and `shell`
    marks a target that is a command line, whose redirects are inspected too.

    A reported write outside `root` is a `deny`. A redirect that looks like it
    left `root` is a `warn` carrying `write_inferred`. Everything else outside
    is a `warn` read.
    """

    root = Path(root).resolve()
    out: list[dict] = []
    for call in calls:
        tool = str(call.get("tool", "?"))
        target = str(call.get("target", ""))
        writes = {str(p) for p in call.get("writes") or ()}
        inferred = set(shell_write_targets(target)) if call.get("shell") else set()
        inferred -= writes
        seen: set[str] = set()
        for path in sorted(writes) + sorted(inferred) + absolute_paths(target):
            if path in seen:
                continue
            seen.add(path)
            kind = classify(path, root)
            if kind is None:
                continue
            if kind == "prefix":
                decision, verb = ALLOW, "read"
            elif path in writes:
                decision, verb = DENY, "write"
            elif path in inferred:
                decision, verb = WARN, "redirect"
            else:
                decision, verb = WARN, "read"
            out.append(
                {
                    "tool": tool,
                    "decision": decision,
                    "path": path,
                    "kind": kind,
                    "enforced": False,
                    "write_inferred": path in inferred,
                    "violations": [f"{verb}={path}"],
                    "input": {"target": target[:400]} if decision != ALLOW else None,
                }
            )
    return out


def writes_outside(records: Iterable[dict]) -> list[str]:
    """Distinct paths a call wrote outside the boundary. Empty means contained."""

    return sorted({r["path"] for r in records if r.get("decision") == DENY})


def reads_outside(records: Iterable[dict]) -> list[str]:
    """Distinct paths read outside the boundary. Recorded, never preventable."""

    return sorted(
        {r["path"] for r in records if r.get("decision") == WARN and not r.get("write_inferred")}
    )


def inferred_writes_outside(records: Iterable[dict]) -> list[str]:
    """Paths a shell command looks like it wrote outside the boundary.

    Separate from `writes_outside` because the evidence is weaker: a redirect
    spotted in a command string, not a change the harness reported. Surfaced
    loudly and never raised on.
    """

    return sorted({r["path"] for r in records if r.get("write_inferred")})


def append_audit(audit_path: Path, records: Iterable[dict]) -> None:
    """Append records as JSONL. Auditing must never break the run."""

    records = list(records)
    if not records:
        return
    try:
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
    except OSError:
        pass


class BoundaryViolation(RuntimeError):
    """A harness call wrote outside the workspace it was confined to.

    Distinct from a generic failure because the run's artifacts can no longer
    be trusted to belong to this run: the agentic proposer has twice written
    candidates into a *different* run's tree, and a silent one of those is
    worse than a crash.
    """
