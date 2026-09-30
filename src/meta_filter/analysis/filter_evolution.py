"""Post-hoc analysis of filter evolution across a run.

This is intentionally read-only and never runs inside the optimization loop.
Its labels (for example, "rubric criterion deleted while new rule added") are
observations, not instructions to the proposer.

The module is safe to import in isolation — it only depends on `complexity`
which is a pure function of filesystem contents.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from meta_filter.complexity import measure_filter_complexity


@dataclass
class IterationSnapshot:
    iteration: int
    filter_dir: Path
    rubric_criteria: list[str]
    rule_names: list[str]
    complexity: dict[str, int]
    complexity_score: float


@dataclass
class Observation:
    iteration: int
    kind: str
    detail: str


def _extract_criterion_titles(rubric_md: str) -> list[str]:
    titles: list[str] = []
    for line in rubric_md.splitlines():
        stripped = line.strip()
        m = re.match(r"^(?:##\s*Criterion:|Criterion:)\s*(.+?)\s*$", stripped)
        if m:
            titles.append(m.group(1).strip())
    return titles


def _extract_rule_names(rules_py: str) -> list[str]:
    try:
        tree = ast.parse(rules_py)
    except SyntaxError:
        return []
    return sorted(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("rule_")
    )


def _load_snapshot(iteration: int, filter_dir: Path) -> IterationSnapshot:
    rubric_md = (
        (filter_dir / "rubric.md").read_text(encoding="utf-8")
        if (filter_dir / "rubric.md").exists()
        else ""
    )
    rules_py = (
        (filter_dir / "rules.py").read_text(encoding="utf-8")
        if (filter_dir / "rules.py").exists()
        else ""
    )
    breakdown, score = measure_filter_complexity(filter_dir)
    return IterationSnapshot(
        iteration=iteration,
        filter_dir=filter_dir,
        rubric_criteria=_extract_criterion_titles(rubric_md),
        rule_names=_extract_rule_names(rules_py),
        complexity=breakdown,
        complexity_score=score,
    )


def _iteration_selected_filter(iter_dir: Path) -> Path | None:
    selection = iter_dir / "selection.json"
    if not selection.exists():
        return None
    data = json.loads(selection.read_text(encoding="utf-8"))
    winner = data.get("selected_candidate")
    if not winner:
        return None
    candidate = iter_dir / "candidates" / winner
    return candidate if candidate.exists() else None


def analyze_run(run_dir: str | Path) -> dict:
    """Return a structured analysis report (also renders as text in __str__)."""

    run_dir = Path(run_dir)
    iterations_root = run_dir / "iterations"
    if not iterations_root.exists():
        raise FileNotFoundError(f"no iterations directory at {iterations_root}")

    iter_dirs = sorted(iterations_root.glob("[0-9][0-9][0-9]"))
    snapshots: list[IterationSnapshot] = []

    # Include the pre-optimization initial filter as snapshot -1 so the
    # analyzer can attribute the very first structural change instead of
    # starting mid-run.
    if iter_dirs:
        current_filter_dir = iter_dirs[0] / "current_filter"
        if current_filter_dir.exists():
            snapshots.append(_load_snapshot(-1, current_filter_dir))

    for iter_dir in iter_dirs:
        selected = _iteration_selected_filter(iter_dir)
        base = selected if selected is not None else iter_dir / "current_filter"
        if not base.exists():
            continue
        snapshots.append(_load_snapshot(int(iter_dir.name), base))

    observations: list[Observation] = []

    for prev, curr in zip(snapshots, snapshots[1:]):
        added_rules = set(curr.rule_names) - set(prev.rule_names)
        removed_rules = set(prev.rule_names) - set(curr.rule_names)
        added_crit = set(curr.rubric_criteria) - set(prev.rubric_criteria)
        removed_crit = set(prev.rubric_criteria) - set(curr.rubric_criteria)

        for name in sorted(added_rules):
            observations.append(
                Observation(curr.iteration, "rule_added", f"new deterministic rule: {name}")
            )
        for name in sorted(removed_rules):
            observations.append(
                Observation(curr.iteration, "rule_removed", f"removed deterministic rule: {name}")
            )
        for name in sorted(added_crit):
            observations.append(
                Observation(curr.iteration, "rubric_criterion_added", f"new criterion: {name}")
            )
        for name in sorted(removed_crit):
            observations.append(
                Observation(
                    curr.iteration,
                    "rubric_criterion_removed",
                    f"criterion dropped: {name}",
                )
            )
        if added_rules and removed_crit:
            observations.append(
                Observation(
                    curr.iteration,
                    "rubric_shrank_after_rule_introduced",
                    f"criteria dropped={sorted(removed_crit)}, rules added={sorted(added_rules)}",
                )
            )
        if removed_rules and added_crit:
            observations.append(
                Observation(
                    curr.iteration,
                    "rule_replaced_by_semantic_criterion",
                    f"rules dropped={sorted(removed_rules)}, criteria added={sorted(added_crit)}",
                )
            )

        if curr.complexity_score < prev.complexity_score:
            observations.append(
                Observation(
                    curr.iteration,
                    "complexity_decreased",
                    f"{prev.complexity_score:.2f} → {curr.complexity_score:.2f}",
                )
            )
        elif curr.complexity_score > prev.complexity_score:
            observations.append(
                Observation(
                    curr.iteration,
                    "complexity_increased",
                    f"{prev.complexity_score:.2f} → {curr.complexity_score:.2f}",
                )
            )

    report = {
        "run_dir": str(run_dir),
        "snapshots": [
            {
                "iteration": s.iteration,
                "rubric_criteria": s.rubric_criteria,
                "rule_names": s.rule_names,
                "complexity": s.complexity,
                "complexity_score": s.complexity_score,
            }
            for s in snapshots
        ],
        "observations": [asdict(o) for o in observations],
    }
    return report
