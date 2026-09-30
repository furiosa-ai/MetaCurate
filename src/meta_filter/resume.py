"""Pick a killed run back up where it stopped.

A 20-iteration run is most of a day, and one can die partway through (the
OOM killer, a lost machine). Losing the completed iterations to start over is
not just slow: the agentic proposer
is not seed-deterministic, so a restart does not reproduce them either. What
was already paid for has to carry forward.

Everything the loop needs is already on disk, written per iteration:

    summary.csv                     the quality trajectory
    iterations/NNN/selection.json   which candidate won and why
    iterations/NNN/proposer/response.json   the diagnosis behind it
    iterations/NNN/current_filter/  the filter that iteration started from

The last of those is the important one. `save_current_filter(k)` runs at the
*top* of iteration k, so a run killed during iteration k leaves
`iterations/k/current_filter` holding exactly the filter that iteration was
working from — the winner of k-1. Resuming means starting at k with that
filter, not rebuilding it from the diffs.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ResumeState:
    """Everything `Optimizer.run` accumulates that must survive a restart."""

    start_iteration: int
    filter_dir: Path
    past_diagnoses: list[dict] = field(default_factory=list)
    quality_trajectory: list[dict] = field(default_factory=list)
    past_winners: list[dict] = field(default_factory=list)


def completed_iterations(run_dir: Path) -> list[int]:
    """Iteration numbers that reached the point of writing a summary row.

    `summary.csv` is appended near the end of an iteration, after selection,
    so a row is the signal that the iteration produced a winner rather than
    merely started.
    """

    summary = Path(run_dir) / "summary.csv"
    if not summary.exists():
        return []

    done: list[int] = []
    with summary.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                done.append(int(row["iteration"]))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(set(done))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def rebuild(run_dir: Path) -> ResumeState | None:
    """Reconstruct the loop's carried state, or None if there is nothing to resume.

    The three history lists feed the proposer's self-continuity prompt. Losing
    them would leave a resumed run proposing as if iteration 0, blind to its
    own past diagnoses — which is worse than obviously worse: it looks like it
    is working.
    """

    run_dir = Path(run_dir)
    done = completed_iterations(run_dir)
    if not done:
        return None

    start = done[-1] + 1
    filter_dir = run_dir / "iterations" / f"{start:03d}" / "current_filter"
    if not filter_dir.is_dir():
        # The run died before the next iteration snapshotted its filter, so the
        # last winner's own candidate directory is the right starting point.
        # NOT `iterations/{done[-1]}/current_filter`: that snapshot is taken at
        # the *top* of an iteration, so it holds the filter the iteration
        # started from — resuming from it would silently discard the winning
        # edits of the very iteration whose summary row proves it completed.
        last_dir = run_dir / "iterations" / f"{done[-1]:03d}"
        winner = _read_json(last_dir / "selection.json").get("selected_candidate", "")
        filter_dir = last_dir / "candidates" / winner
        if not winner or not filter_dir.is_dir():
            logger.warning(
                "no winner directory to resume from under %s (selection said %r)",
                last_dir,
                winner,
            )
            return None

    state = ResumeState(start_iteration=start, filter_dir=filter_dir)

    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                k = int(row["iteration"])
            except (KeyError, TypeError, ValueError):
                continue
            state.quality_trajectory.append(
                {
                    "iteration": k,
                    "score": _as_float(row.get("score")),
                    "rubric_quality": _as_float(row.get("rubric_quality")),
                    "rule_quality": _as_float(row.get("rule_quality")),
                    "critic_agreement": _as_float(row.get("critic_agreement")),
                    "review_rate": _as_float(row.get("review_rate")),
                }
            )

    for k in done:
        it_dir = run_dir / "iterations" / f"{k:03d}"
        selection = _read_json(it_dir / "selection.json")
        response = _read_json(it_dir / "proposer" / "response.json")
        candidates = response.get("candidates") or []

        state.past_diagnoses.append(
            {
                "iteration": k,
                "diagnosis": list(response.get("diagnosis") or []),
                "candidate_notes": [(c or {}).get("notes", "") for c in candidates],
            }
        )
        winner = selection.get("selected_candidate", "")
        state.past_winners.append(
            {
                "iteration": k,
                "winner": winner,
                "reason": selection.get("reason", ""),
                "winner_notes": next(
                    (
                        (c or {}).get("notes", "")
                        for c in candidates
                        if (c or {}).get("id") == winner
                    ),
                    "",
                ),
            }
        )

    return state


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
