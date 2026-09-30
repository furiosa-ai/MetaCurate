"""Filter complexity measurement.

Complexity is deliberately observable so the optimizer can be pressured to
simplify hybrid filters that grow both a large rubric AND a large rule set.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _count_python_loc(path: Path) -> int:
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    loc = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        loc += 1
    return loc


def _count_rule_functions(path: Path) -> int:
    """Count top-level callables in `rules.py` whose name starts with `rule_`."""

    if not path.exists():
        return 0
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return 0
    return sum(
        1
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("rule_")
    )


def _count_rubric_criteria(path: Path) -> tuple[int, int]:
    """Return (criterion_count, rubric_text_length_chars)."""

    if not path.exists():
        return (0, 0)
    text = path.read_text(encoding="utf-8")
    criteria = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("criterion:") or stripped.startswith("## Criterion"):
            criteria += 1
    return criteria, len(text)


# What a complexity point costs, per signal. Configurable because the
# defaults are themselves a hypothesis: `criteria` and `rule_count` at 1.0
# make deleting a rubric axis the cheapest available gain, and `rules_loc` at
# 0.05 rewards compression rather than simplicity (a revision that shrinks
# the statement count with every published value bit-identical). Overriding
# these is how that gets measured instead of argued.
DEFAULT_COMPLEXITY_COEFFICIENTS: dict[str, float] = {
    "rubric_criteria": 1.0,
    "rule_count": 1.0,
    "rules_loc": 0.05,
    "policy_loc": 0.02,
    "rubric_chars": 1 / 500.0,
}


def measure_filter_complexity(
    filter_dir: str | Path,
    coefficients: dict[str, float] | None = None,
) -> tuple[dict[str, int], float]:
    """Return a complexity breakdown and a single weighted scalar score.

    The scalar score is a simple sum of components; the breakdown is preserved
    so the retrospective analyzer and reports can present it directly.
    """

    filter_dir = Path(filter_dir)
    rubric_path = filter_dir / "rubric.md"
    rules_path = filter_dir / "rules.py"
    policy_path = filter_dir / "policy.py"

    criteria, rubric_chars = _count_rubric_criteria(rubric_path)
    rules_loc = _count_python_loc(rules_path)
    policy_loc = _count_python_loc(policy_path)
    rule_count = _count_rule_functions(rules_path)

    breakdown = {
        "rubric_criteria": criteria,
        "rubric_chars": rubric_chars,
        "rules_loc": rules_loc,
        "policy_loc": policy_loc,
        "rule_count": rule_count,
    }
    # Character counts dominate raw LOC, so the default damps them to keep the
    # scalar in a comparable range with rule/criterion counts.
    coef = {**DEFAULT_COMPLEXITY_COEFFICIENTS, **(coefficients or {})}
    score = sum(coef.get(name, 0.0) * value for name, value in breakdown.items())
    return breakdown, score
