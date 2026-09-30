"""agent_guard — path containment and Bash filtering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_filter.llm.agent_guard import (
    ENV_AUDIT,
    ENV_ROOT,
    bash_violations,
    build_guard_settings,
    is_contained,
    main,
    path_violations,
)


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "run"
    (r / "iterations").mkdir(parents=True)
    (tmp_path / "other_run").mkdir()
    return r


def test_is_contained_accepts_inside_and_the_root_itself(root):
    assert is_contained(str(root), root, root)
    assert is_contained("iterations", root, root)
    assert is_contained(str(root / "iterations" / "000"), root, root)


def test_is_contained_rejects_traversal_and_siblings(root):
    assert not is_contained("..", root, root)
    assert not is_contained("../other_run", root, root)
    assert not is_contained(str(root.parent / "other_run"), root, root)
    assert not is_contained("/etc/passwd", root, root)


def test_is_contained_follows_symlinks_out(root):
    # A link planted inside the boundary must not become a way through it.
    (root / "escape").symlink_to(root.parent / "other_run")
    assert not is_contained("escape", root, root)


def test_path_violations_flags_only_out_of_bounds(root):
    assert path_violations("Write", {"file_path": "candidates/a.py"}, root, root) == []
    bad = path_violations("Write", {"file_path": "../other_run/a.py"}, root, root)
    assert bad and "other_run" in bad[0]


def test_bash_allows_in_bounds_exploration(root):
    for cmd in ("ls -la", "grep -rn 'foo' iterations/", "find . -name '*.json'"):
        assert bash_violations(cmd, root, root) == [], cmd


def test_bash_blocks_traversal_and_absolute_escape(root):
    assert bash_violations("ls ../other_run", root, root)
    assert bash_violations(f"cp x {root.parent}/other_run/y", root, root)
    assert bash_violations("cat /etc/passwd", root, root)


def test_bash_blocks_what_it_cannot_read(root):
    # Ambiguity resolves to a block: one wasted retry beats a corrupted run.
    assert bash_violations("echo $(cat ../other_run/x)", root, root)
    assert bash_violations("ls 'unterminated", root, root)


def test_bash_inspects_paths_hidden_behind_flags(root):
    assert bash_violations("grep -r --include=/etc/x foo .", root, root)
    assert bash_violations("ls -la", root, root) == []


def test_main_denies_and_audits(root, monkeypatch, capsys):
    audit = root / "audit.jsonl"
    monkeypatch.setenv(ENV_ROOT, str(root))
    monkeypatch.setenv(ENV_AUDIT, str(audit))
    event = {
        "cwd": str(root),
        "tool_name": "Write",
        "tool_input": {"file_path": "../other_run/policy.py"},
    }
    monkeypatch.setattr("sys.stdin", _Stdin(json.dumps(event)))

    assert main() == 2
    assert "outside the working directory" in capsys.readouterr().err
    rec = json.loads(audit.read_text().strip())
    assert rec["decision"] == "deny"
    assert rec["tool"] == "Write"


def test_main_allows_in_bounds(root, monkeypatch):
    audit = root / "audit.jsonl"
    monkeypatch.setenv(ENV_ROOT, str(root))
    monkeypatch.setenv(ENV_AUDIT, str(audit))
    event = {
        "cwd": str(root),
        "tool_name": "Read",
        "tool_input": {"file_path": "iterations/000/x.json"},
    }
    monkeypatch.setattr("sys.stdin", _Stdin(json.dumps(event)))

    assert main() == 0
    assert json.loads(audit.read_text().strip())["decision"] == "allow"


def test_build_guard_settings_wires_a_pretooluse_hook(root):
    settings = json.loads(build_guard_settings(root, root / "audit.jsonl"))
    entry = settings["hooks"]["PreToolUse"][0]
    for tool in ("Read", "Write", "Edit", "Glob", "Grep", "Bash"):
        assert tool in entry["matcher"]
    assert entry["hooks"][0]["command"].endswith("agent_guard.py'") or entry["hooks"][0][
        "command"
    ].endswith("agent_guard.py")
    assert settings["env"][ENV_ROOT] == str(Path(root).resolve())


def test_build_guard_settings_disables_auto_memory(root):
    """Auto-memory bypasses PreToolUse entirely, so it must be off in settings."""

    settings = json.loads(build_guard_settings(root, root / "audit.jsonl"))
    assert settings["autoMemoryEnabled"] is False


class _Stdin:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> str:
        return self._payload
