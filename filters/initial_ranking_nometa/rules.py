# Deterministic rules for the metadata-free composite initial filter: the
# Nemotron 3 Ultra p.17 transcript heuristics, and nothing else.
#
# The non-blank seed of the metadata-free cells (`configs/metadata_free/`),
# shared by the DSv4Pro and the Nemotron pool. Every rule reads the commands
# the agent issued and the diff it submitted; no rule reads any other
# metadata key, so a rule fires the same way on either pool.
#
# Contract:
#     def rule_<name>(trajectory) -> dict | None      # None when it declines
#     fired = {"name": str, "verdict": "ACCEPT|REJECT|REVIEW", "evidence": [str]}
#
# `trajectory` is duck-typed: `.metadata` (a mapping holding `submission`,
# the agent's own diff) and `.turns` (turns expose `role` / `action` /
# `content`). Each assistant turn's `content` is the raw shell command, which
# is what the Nemotron-style detection regexes below match against.
#
# LAYERS (each evidence line says which one it belongs to)
#   T    transcript + submitted diff only
#
# Rules:
#   rule_nemotron_*          T    the seven p.17 signals, all REJECT
#
# Signal 1 (`rule_nemotron_no_valid_submission`) reads the submitted diff
# alone: a trajectory that ships no git diff has not ended in a patch.

from __future__ import annotations

import re

# --- Thresholds ---------------------------------------------------------------

EDIT_TEST_LOOP_MAX_RETURNS = 8
EDIT_TEST_LOOP_MAX_REVERTS = 3
LOST_EXPLORATION_MIN_ACTIONS = 20
LOST_EXPLORATION_EXPLORE_SHARE = 0.85
LOST_EXPLORATION_MAX_EDITS = 2
MALFORMED_RATE_MAX = 0.10

NOTE_T = "[layer T] reads only the commands the agent issued and the diff it submitted"

# --- Command canonicalizer ---------------------------------------------------

_WRAPPER_HEADS = {"bash", "sh", "zsh", "env", "sudo", "time", "nohup", "xargs"}


def _canonicalize_action(cmd: str, *, depth: int = 0) -> str:
    """Turn a raw shell command into a canonical action token."""

    if not cmd or depth > 3:
        return "shell"
    cmd_stripped = cmd.strip()
    if not cmd_stripped:
        return "shell"

    while re.match(r"^[A-Z_][A-Z_0-9]*=\S+\s+", cmd_stripped):
        cmd_stripped = cmd_stripped.split(None, 1)[1] if " " in cmd_stripped else cmd_stripped

    tokens = cmd_stripped.split()
    if not tokens:
        return "shell"
    head = tokens[0].lower().lstrip("./")

    if head in {"cd", "pushd", "popd", "chdir", "pwd", "true"} and depth < 3:
        for sep in (" && ", "&&", " ; ", ";"):
            if sep in cmd_stripped:
                rest = cmd_stripped.split(sep, 1)[1].strip()
                if rest:
                    return _canonicalize_action(rest, depth=depth + 1)

    if head in _WRAPPER_HEADS and len(tokens) >= 2:
        m = re.search(r"""-c\s+(?:"([^"]+)"|'([^']+)'|(\S+))""", cmd_stripped)
        if m:
            inner = next(g for g in m.groups() if g)
            return _canonicalize_action(inner, depth=depth + 1)
        if head in {"env", "sudo", "time", "nohup", "xargs"}:
            return _canonicalize_action(" ".join(tokens[1:]), depth=depth + 1)
        return head

    if head in {"python", "python3", "python2", "py"}:
        if len(tokens) >= 3 and tokens[1] == "-m":
            return tokens[2].split(".")[0].lower()
        if len(tokens) >= 2 and tokens[1] == "-c":
            return "python_inline"
        if len(tokens) >= 3 and tokens[1] == "setup.py" and tokens[2] == "test":
            return "pytest"
        return "python"

    if head == "git" and len(tokens) >= 2:
        return f"git_{tokens[1].lower()}"

    m = re.match(r"([A-Za-z_][\w\-\.]*)", head)
    return m.group(1).lower() if m else "shell"


DISALLOWED_GIT_SUBCOMMANDS = {
    "push",
    "pull",
    "fetch",
    "clone",
    "cherry-pick",
    "reflog",
    "fsck",
    "remote",
    "ls-remote",
}

_EDIT_RE = re.compile(
    r"(sed\s+(-[a-zA-Z]+\s+)*-i"
    r"|cat\s*>{1,2}\s*\S"
    r"|tee\s+(-a\s+)?\S"
    r"|\bgit\s+apply\b"
    r"|\bpatch\s+-p\d"
    r"|apply_patch"
    r"|(echo|printf)\s+.{0,400}?>{1,2}\s*\S"
    r"|open\(\s*[^)]{0,200}['\"][wa]\+?['\"]"
    r"|\.write_text\(|\.write\("
    r")",
    re.DOTALL,
)
_HEREDOC_WRITE_RE = re.compile(r"(cat|tee)\s*(>{1,2}\s*\S+\s*)?<<|>{1,2}\s*\S+\s*<<")

_GIT_DISALLOWED_RE = re.compile(
    r"\bgit\s+(?:-[^\s]+\s+)*(push|pull|fetch|clone|cherry-pick|reflog|fsck|remote|ls-remote)\b"
)
_GIT_REVERT_RE = re.compile(
    r"\bgit\s+(checkout\s+(\S+\s+)?--\s|restore\s|stash\b(?!\s+list)|reset\s+--hard)"
)

_TEST_TOKENS = {"pytest", "py.test", "nosetests", "nose2", "nose", "tox", "unittest"}
_PYRUN_TOKENS = {"python", "python_inline"}

_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\n")

_DEBUG_LINE_PATTERNS = {
    "print(": re.compile(r"\bprint\("),
    "pdb": re.compile(r"\bimport\s+pdb\b|\bpdb\.set_trace\b"),
    "breakpoint()": re.compile(r"\bbreakpoint\(\)"),
}


# --- Trajectory access helpers ----------------------------------------------


def _meta(trajectory, key, default=None):
    md = getattr(trajectory, "metadata", None) or {}
    return md.get(key, default) if hasattr(md, "get") else default


def _field(turn, name):
    return turn.get(name) if isinstance(turn, dict) else getattr(turn, name, None)


def _text(value):
    if isinstance(value, list | tuple):
        return " ".join(_text(v) for v in value)
    return value if isinstance(value, str) else ""


def _commands(trajectory):
    """(commands, n_malformed, n_assistant) over the assistant turns."""

    cmds: list[str] = []
    malformed = 0
    n_assistant = 0
    for turn in list(getattr(trajectory, "turns", None) or []):
        if _field(turn, "role") != "assistant":
            continue
        n_assistant += 1
        text = _text(_field(turn, "content"))
        if text.strip():
            cmds.append(text)
        else:
            malformed += 1
    return cmds, malformed, n_assistant


# --- Per-command classification ----------------------------------------------


def _segment_tokens(cmd: str) -> list[str]:
    tokens = []
    for seg in _SEGMENT_SPLIT_RE.split(cmd):
        seg = seg.strip()
        if seg:
            tokens.append(_canonicalize_action(seg))
    return tokens


def classify_command(cmd: str) -> dict:
    """What one shell command does: edit / test / pyrun / git flags."""

    flags = {
        "edit": False,
        "test": False,
        "pyrun": False,
        "git_disallowed": False,
        "git_revert": False,
    }
    if not cmd or not cmd.strip():
        return flags

    tokens = _segment_tokens(cmd)
    flags["test"] = any(t in _TEST_TOKENS for t in tokens)
    flags["pyrun"] = any(t in _PYRUN_TOKENS for t in tokens)
    flags["edit"] = bool(_EDIT_RE.search(cmd) or _HEREDOC_WRITE_RE.search(cmd))
    flags["git_disallowed"] = bool(_GIT_DISALLOWED_RE.search(cmd)) or any(
        t.startswith("git_") and t.removeprefix("git_") in DISALLOWED_GIT_SUBCOMMANDS
        for t in tokens
    )
    flags["git_revert"] = bool(_GIT_REVERT_RE.search(cmd))
    return flags


def _primary(flags: dict) -> str:
    if flags["edit"]:
        return "E"
    if flags["test"]:
        return "T"
    if flags["pyrun"]:
        return "P"
    return "R"


def find_debug_artifacts(submission: str) -> list[str]:
    """Debug patterns on ADDED lines of the final patch, per the paper."""

    found: set[str] = set()
    for line in submission.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for name, pattern in _DEBUG_LINE_PATTERNS.items():
            if pattern.search(line):
                found.add(name)
    return sorted(found)


def _analyze(trajectory) -> dict:
    """One pass computing every count the seven Nemotron signals read."""

    cmds, n_malformed, n_assistant = _commands(trajectory)
    flags_seq = [classify_command(c) for c in cmds]
    seq = "".join(_primary(f) for f in flags_seq)
    n_actions = len(flags_seq)
    core = [c for c in seq if c in "ET"]
    return {
        "n_actions": n_actions,
        "n_edits": sum(f["edit"] for f in flags_seq),
        "n_tests": sum(f["test"] for f in flags_seq),
        "n_reverts": sum(f["git_revert"] for f in flags_seq),
        "n_test_edit_returns": sum(1 for a, b in zip(core, core[1:]) if a == "T" and b == "E"),
        "git_disallowed": any(f["git_disallowed"] for f in flags_seq),
        "n_malformed": n_malformed,
        "n_assistant": n_assistant,
        "malformed_rate": n_malformed / n_assistant if n_assistant else 1.0,
        "explore_share": seq.count("R") / n_actions if n_actions else 1.0,
    }


def _fire(name, verdict, evidence):
    return {"name": name, "verdict": verdict, "evidence": evidence}


# --- The seven Nemotron signals (layer T), one rule each ---------------------


def rule_nemotron_no_valid_submission(trajectory) -> dict | None:
    """Signal 1: the trajectory must end with a submitted git diff."""

    submission = _text(_meta(trajectory, "submission"))
    if submission.strip().startswith("diff --git"):
        return None
    return _fire(
        "rule_nemotron_no_valid_submission",
        "REJECT",
        [
            f"a {len(submission)}-byte submission that is not a git diff: the "
            "trajectory does not end in a patch",
            NOTE_T,
        ],
    )


def rule_nemotron_disallowed_git(trajectory) -> dict | None:
    """Signal 2: disallowed git operations anywhere in the trajectory."""

    if not _analyze(trajectory)["git_disallowed"]:
        return None
    return _fire(
        "rule_nemotron_disallowed_git",
        "REJECT",
        [
            "a disallowed git operation (push/pull/fetch/clone/cherry-pick/"
            "reflog/fsck/remote/ls-remote) appears in the command stream",
            NOTE_T,
        ],
    )


def rule_nemotron_edit_test_loop(trajectory) -> dict | None:
    """Signal 3: thrashing between edits and test runs without converging."""

    stats = _analyze(trajectory)
    returns, reverts = stats["n_test_edit_returns"], stats["n_reverts"]
    if returns < EDIT_TEST_LOOP_MAX_RETURNS and reverts < EDIT_TEST_LOOP_MAX_REVERTS:
        return None
    return _fire(
        "rule_nemotron_edit_test_loop",
        "REJECT",
        [
            f"{returns} return(s) to editing after a test run (limit "
            f"{EDIT_TEST_LOOP_MAX_RETURNS}) and {reverts} self-revert(s) (limit "
            f"{EDIT_TEST_LOOP_MAX_REVERTS}): the edit-test cycle never converges",
            NOTE_T,
        ],
    )


def rule_nemotron_lost_in_exploration(trajectory) -> dict | None:
    """Signal 4: mostly reads/searches the repo but rarely (or never) edits."""

    stats = _analyze(trajectory)
    if stats["n_edits"] == 0:
        return _fire(
            "rule_nemotron_lost_in_exploration",
            "REJECT",
            [
                f"{stats['n_actions']} command(s) and not one of them edits a "
                "file: the agent never moved from exploration to a fix",
                NOTE_T,
            ],
        )
    if (
        stats["n_actions"] >= LOST_EXPLORATION_MIN_ACTIONS
        and stats["explore_share"] >= LOST_EXPLORATION_EXPLORE_SHARE
        and stats["n_edits"] <= LOST_EXPLORATION_MAX_EDITS
    ):
        return _fire(
            "rule_nemotron_lost_in_exploration",
            "REJECT",
            [
                f"{stats['explore_share']:.0%} of {stats['n_actions']} commands "
                f"are read/search with only {stats['n_edits']} edit(s) "
                f"(limits: >= {LOST_EXPLORATION_EXPLORE_SHARE:.0%} explore over "
                f">= {LOST_EXPLORATION_MIN_ACTIONS} actions, "
                f"<= {LOST_EXPLORATION_MAX_EDITS} edits)",
                NOTE_T,
            ],
        )
    return None


def rule_nemotron_tool_call_hygiene(trajectory) -> dict | None:
    """Signal 5: high rate of malformed (command-less) assistant turns."""

    stats = _analyze(trajectory)
    if stats["malformed_rate"] <= MALFORMED_RATE_MAX:
        return None
    return _fire(
        "rule_nemotron_tool_call_hygiene",
        "REJECT",
        [
            f"{stats['n_malformed']} of {stats['n_assistant']} assistant turn(s) "
            f"carry no runnable command ({stats['malformed_rate']:.0%} malformed, "
            f"limit {MALFORMED_RATE_MAX:.0%})",
            NOTE_T,
        ],
    )


def rule_nemotron_debug_artifacts(trajectory) -> dict | None:
    """Signal 6: debug artifacts on added lines of the final patch."""

    artifacts = find_debug_artifacts(_text(_meta(trajectory, "submission")))
    if not artifacts:
        return None
    return _fire(
        "rule_nemotron_debug_artifacts",
        "REJECT",
        [
            "the submitted patch adds debug artifact(s) "
            f"{', '.join(artifacts)}: leftover instrumentation, not a fix",
            NOTE_T,
        ],
    )


def rule_nemotron_no_verification(trajectory) -> dict | None:
    """Signal 7: the agent edits but never invokes a test runner."""

    stats = _analyze(trajectory)
    if not (stats["n_edits"] >= 1 and stats["n_tests"] == 0):
        return None
    return _fire(
        "rule_nemotron_no_verification",
        "REJECT",
        [
            f"{stats['n_edits']} edit command(s) and zero test-runner "
            "invocations (pytest/unittest/tox/nose): the change was never "
            "verified",
            NOTE_T,
        ],
    )
