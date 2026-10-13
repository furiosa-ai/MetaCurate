"""Load and materialize hybrid filter policies.

A filter policy is a directory containing three files:

    filter/
        rubric.md    (text)
        rules.py     (top-level `rule_<name>(trajectory) -> dict|None` callables)
        policy.py    (aggregation logic exposing `decide(...)`, plus threshold
                      constants: REVIEW_LOWER, REVIEW_UPPER, REJECT_MARGIN)

Filters are treated as immutable versioned directories — the loader always
snapshots them into a run's `iterations/<k>/candidates/<id>/` directory so
history is fully replayable.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import shutil
import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path

# Candidate filters are loaded concurrently (optimizer evaluates candidates in
# a thread pool), and the policy import below temporarily rebinds the global
# name `rules`. Serialize that window so candidates can't see each other's.
_IMPORT_LOCK = threading.Lock()


class FilterValidationError(RuntimeError):
    """Raised when a candidate filter directory is not runnable."""


@dataclass
class LoadedFilter:
    """A concrete filter, imported and ready to run."""

    directory: Path
    version: str
    rubric_md: str
    rules_module: types.ModuleType
    policy_module: types.ModuleType
    rule_functions: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.directory.name

    def snapshot(self, target: str | Path) -> Path:
        """Copy the filter directory to `target` and return the new path."""

        target = Path(target)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.directory, target)
        return target


def _hash_dir(directory: Path) -> str:
    """Content hash across the three canonical files."""

    h = hashlib.sha256()
    for fname in ("rubric.md", "rules.py", "policy.py"):
        path = directory / fname
        if path.exists():
            h.update(fname.encode())
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    return h.hexdigest()[:12]


def _validate_python_syntax(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        raise FilterValidationError(f"{path}: syntax error at {exc.lineno}: {exc.msg}") from exc


def _import_module(path: Path, module_name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise FilterValidationError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure a fresh module identity per load (avoid cross-candidate leakage).
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_filter(directory: str | Path) -> LoadedFilter:
    """Validate and import a filter directory."""

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FilterValidationError(f"filter directory not found: {directory}")

    for required in ("rubric.md", "rules.py", "policy.py"):
        if not (directory / required).exists():
            raise FilterValidationError(f"missing {required} in {directory}")

    _validate_python_syntax(directory / "rules.py")
    _validate_python_syntax(directory / "policy.py")

    version = _hash_dir(directory)
    tag = f"meta_filter_ephemeral_{version}"

    with _IMPORT_LOCK:
        rules_module = _import_module(directory / "rules.py", f"{tag}_rules")
        # Proposals routinely write `from rules import _helper` in policy.py.
        # Bind the sibling module under that plain name for the policy import.
        previous_rules = sys.modules.get("rules")
        sys.modules["rules"] = rules_module
        try:
            policy_module = _import_module(directory / "policy.py", f"{tag}_policy")
        finally:
            if previous_rules is None:
                sys.modules.pop("rules", None)
            else:
                sys.modules["rules"] = previous_rules

    if not hasattr(policy_module, "decide"):
        raise FilterValidationError(f"policy.py must expose `decide(...)` ({directory})")

    rule_functions = [
        getattr(rules_module, name)
        for name in dir(rules_module)
        if name.startswith("rule_") and callable(getattr(rules_module, name))
    ]

    rubric_md = (directory / "rubric.md").read_text(encoding="utf-8")

    return LoadedFilter(
        directory=directory,
        version=version,
        rubric_md=rubric_md,
        rules_module=rules_module,
        policy_module=policy_module,
        rule_functions=rule_functions,
    )


def materialize_candidate(
    target_dir: str | Path,
    *,
    rubric_md: str,
    rules_py: str,
    policy_py: str,
) -> Path:
    """Write the three canonical files to `target_dir` and return the path."""

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "rubric.md").write_text(rubric_md, encoding="utf-8")
    (target / "rules.py").write_text(rules_py, encoding="utf-8")
    (target / "policy.py").write_text(policy_py, encoding="utf-8")
    return target
