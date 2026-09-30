"""Measure how a candidate relates to the filter it was supposed to improve.

The proposer is told to copy `current_filter/` and edit the copy, one
hypothesis per candidate. Instructions alone do not hold: candidates can share
only 20-50% of their lines with the baseline — rewrites wearing the name of
edits. Then nothing accumulates across generations, and because the scorer
cache is keyed on the rubric text, every candidate pays full price for every
trajectory.

This module turns that into something observable. It does not reject
candidates: a rewrite can still be the right answer, and the proposer is the
one holding the evidence. It reports the shape so the loop can log it, the
proposer can be shown its own past behaviour, and a run that quietly drifts
back to rewriting is visible rather than inferred.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

FILTER_FILES = ("rubric.md", "rules.py", "policy.py")

# Above this share of new lines, "edit" is the wrong word for what happened.
# Rewrites run 0.50-0.78; a candidate that adds one rule and the policy
# branch reading it lands well under 0.20.
REWRITE_THRESHOLD = 0.35

# Below this share of the parent's own lines surviving in a file, likewise —
# and this is the half novelty cannot see. Novelty counts what arrived; a
# candidate that deletes two thirds of the rubric and adds nothing has a
# novelty of zero and is still a replacement, and would read as an edit.
# Per file, not pooled: rules.py is several times the rubric's size in lines,
# so a pooled ratio lets an untouched rules.py pay for a gutted rubric.
KEPT_LINES_THRESHOLD = 0.5


@dataclass(frozen=True)
class ProposalShape:
    """How much of a candidate is new relative to its baseline."""

    candidate: str
    added: int
    removed: int
    baseline_lines: int
    files_touched: tuple[str, ...]
    # (file, share of that file's parent lines still present), for the files
    # the parent actually had content in. Empty when there is no parent.
    kept_shares: tuple[tuple[str, float], ...] = ()

    @property
    def novelty(self) -> float:
        """Added lines as a share of the baseline's size."""

        if self.baseline_lines <= 0:
            return 1.0
        return self.added / self.baseline_lines

    @property
    def min_kept_share(self) -> float:
        """The worst-kept file. 1.0 when there was no parent to keep."""

        return min((share for _, share in self.kept_shares), default=1.0)

    @property
    def is_rewrite(self) -> bool:
        return self.novelty >= REWRITE_THRESHOLD or self.min_kept_share < KEPT_LINES_THRESHOLD

    def describe(self) -> str:
        worst = min(self.kept_shares, key=lambda kv: kv[1], default=None)
        kept = f", keeps {worst[1]:.0%} of {worst[0]}" if worst else ""
        return (
            f"{self.candidate}: +{self.added}/-{self.removed} lines over "
            f"{self.baseline_lines} ({self.novelty:.0%} new{kept}) across "
            f"{', '.join(self.files_touched) or 'nothing'}"
        )


def measure(baseline_dir: Path, candidate_dir: Path) -> ProposalShape:
    """Diff a candidate against the filter it started from."""

    added = removed = baseline_lines = 0
    touched: list[str] = []
    kept: list[tuple[str, float]] = []

    for name in FILTER_FILES:
        before = _read_lines(Path(baseline_dir) / name)
        after = _read_lines(Path(candidate_dir) / name)
        baseline_lines += len(before)

        # As a set of stripped, non-empty bodies: moving a section or
        # reindenting it keeps the line, and only deleting it does not.
        parent_bodies = _bodies(before)
        if parent_bodies:
            survivors = parent_bodies & _bodies(after)
            kept.append((name, len(survivors) / len(parent_bodies)))

        file_added = file_removed = 0
        for line in difflib.unified_diff(before, after, n=0, lineterm=""):
            if line.startswith("+") and not line.startswith("+++"):
                file_added += 1
            elif line.startswith("-") and not line.startswith("---"):
                file_removed += 1

        if file_added or file_removed:
            touched.append(name)
        added += file_added
        removed += file_removed

    return ProposalShape(
        candidate=Path(candidate_dir).name,
        added=added,
        removed=removed,
        baseline_lines=baseline_lines,
        files_touched=tuple(touched),
        kept_shares=tuple(kept),
    )


def report(baseline_dir: Path, candidate_dirs: list[Path]) -> list[ProposalShape]:
    """Measure every candidate and log the ones that rewrote rather than edited."""

    shapes = [measure(baseline_dir, d) for d in candidate_dirs]
    rewrites = [s for s in shapes if s.is_rewrite]
    for shape in shapes:
        logger.debug("proposal shape: %s", shape.describe())
    if rewrites:
        logger.warning(
            "%d of %d candidates rewrote rather than edited the baseline: %s",
            len(rewrites),
            len(shapes),
            "; ".join(s.describe() for s in rewrites),
        )
    return shapes


def _bodies(lines: list[str]) -> set[str]:
    """The distinct non-blank content of `lines`, indentation removed."""

    return {stripped for line in lines if (stripped := line.strip())}


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
