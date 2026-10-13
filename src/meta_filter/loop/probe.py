"""Fixed probe set — constant data the loop's drift is measured on.

Every iteration's SELECTED filter is run on the same fixed probe trajectories
(the run's ``--probe`` file), so accept-rate
drift and decision flips are observable on data that never changes underneath
the filter. The probe set carries no gold labels or adjudication, and its
rows are visible to the proposer and critic — unlike the hidden heldout.

It must NEVER enter selection or the objective: this module only records
what the selected filter did and renders the trend for prompts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meta_filter.core.filter_loader import LoadedFilter
from meta_filter.core.schemas import Decision, Trajectory
from meta_filter.core.solver import Solver
from meta_filter.visibility.trajectory import load_trajectories

PROBE_SUBDIR = "probe"
PROBE_DECISIONS_FILE = "probe_decisions.jsonl"


@dataclass
class ProbeReport:
    """What the selected filter did on the probe set, one iteration."""

    decisions: dict[str, str]
    accept_rate: float
    accepts: int = 0
    total: int = 0
    # The filter's continuous scores on the probe rows.
    scores: dict[str, float] = field(default_factory=dict)


class ProbeSet:
    """The fixed probe trajectories, loaded from the jsonl the make script wrote."""

    def __init__(self, trajectories: list[Trajectory]):
        self.trajectories = trajectories

    @classmethod
    def from_path(cls, path: str | Path) -> ProbeSet:
        return cls(load_trajectories(path))

    def __len__(self) -> int:
        return len(self.trajectories)

    def evaluate(self, loaded_filter: LoadedFilter, solver: Solver) -> ProbeReport:
        """Run the filter on the probe trajectories and summarize the outcome."""

        raw = solver.run(loaded_filter, self.trajectories)
        decisions = {d.trajectory_id: str(d.decision) for d in raw}
        scores = {d.trajectory_id: float(d.rank_score) for d in raw if d.rank_score is not None}

        accepts = sum(
            1 for t in self.trajectories if decisions.get(t.trajectory_id) == Decision.ACCEPT
        )
        total = len(self.trajectories)
        return ProbeReport(
            decisions=decisions,
            accept_rate=accepts / total if total else 0.0,
            accepts=accepts,
            total=total,
            scores=scores,
        )


def fixed_validation_pairs(
    trajectory_ids: list[str], *, n_pairs: int = 120, seed: int = 4242
) -> list[tuple[str, str]]:
    """Deterministic unordered pairs from the probe ids.

    Validation in the proper sense: a FIXED set of pairs on FIXED data, so
    the loop's own objective (pairwise concordance, adjudicated by the loop's
    own pair judge — never any external gold) can be tracked across
    iterations and compared across runs that share the probe set and judge.
    The pairs derive only from the sorted id list and the seed, so every run
    regenerates the identical list without storing it.
    """

    ids = sorted(set(trajectory_ids))
    universe = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]
    if len(universe) <= n_pairs:
        return universe
    rng = random.Random(seed)
    return sorted(rng.sample(universe, n_pairs))


def flip_rate(prev_decisions: dict[str, str] | None, new_decisions: dict[str, str]) -> float:
    """Fraction of common trajectory ids whose decision changed.

    Disjoint (or missing) id sets have nothing in common to compare, so the
    flip rate is 0.0 — "no observed flips", not "everything flipped".
    """

    if not prev_decisions:
        return 0.0
    common = prev_decisions.keys() & new_decisions.keys()
    if not common:
        return 0.0
    flips = sum(1 for tid in common if prev_decisions[tid] != new_decisions[tid])
    return flips / len(common)


def _decisions_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / PROBE_SUBDIR / PROBE_DECISIONS_FILE


def append_report(
    run_dir: str | Path,
    iteration: int,
    report: ProbeReport,
    prev: dict[str, str] | None,
    validation_concordance: float | None = None,
) -> dict[str, Any]:
    """Persist one iteration's probe row and return it.

    `prev` is the previous iteration's decisions dict (``latest_decisions``),
    used only to compute the flip rate — the file itself is append-only.
    """

    row: dict[str, Any] = {
        "iteration": iteration,
        "accept_rate": report.accept_rate,
        "flip_rate_vs_prev": flip_rate(prev, report.decisions),
        "accepts": report.accepts,
        "total": report.total,
        "decisions": report.decisions,
    }
    if validation_concordance is not None:
        row["validation_concordance"] = round(validation_concordance, 4)
    path = _decisions_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def load_reports(run_dir: str | Path) -> list[dict[str, Any]]:
    """All persisted probe rows for a run, oldest first."""

    path = _decisions_path(run_dir)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_decisions(run_dir: str | Path) -> dict[str, str] | None:
    """The most recent iteration's decisions dict, or None before iteration 1."""

    rows = load_reports(run_dir)
    if not rows:
        return None
    return dict(rows[-1].get("decisions", {}))


def render_for_prompt(rows: list[dict[str, Any]], max_chars: int = 1200) -> str:
    """Compact trend table of the last few probe rows for prompt inclusion.

    Shows iteration, accept rate, flip rate vs the previous iteration, and the
    accept count — never the raw decisions, which would blow the budget and
    invite overfitting to individual rows.
    """

    if not rows:
        return ""

    has_validation = any(r.get("validation_concordance") is not None for r in rows)
    header = (
        "Fixed probe set (constant data, no labels) — selected filter per iteration:\n"
        "iter | accept_rate | flip_vs_prev | accepts"
        + (" | val_concordance" if has_validation else "")
    )

    def line(row: dict[str, Any]) -> str:
        total = row.get("total")
        composition = f"{row.get('accepts', 0)}/{total}" if total else "-"
        base = (
            f"{row.get('iteration', '?'):>4} | "
            f"{row.get('accept_rate', 0.0):11.3f} | "
            f"{row.get('flip_rate_vs_prev', 0.0):12.3f} | "
            f"{composition}"
        )
        if has_validation:
            vc = row.get("validation_concordance")
            base += f" | {vc:.3f}" if vc is not None else " | -"
        return base

    # Newest rows are the most informative: keep taking rows from the end
    # while the whole block still fits in max_chars.
    lines: list[str] = []
    for row in reversed(rows):
        candidate = [line(row), *lines]
        if len("\n".join([header, *candidate])) > max_chars:
            break
        lines = candidate
    if not lines:
        return header[:max_chars]
    return "\n".join([header, *lines])
