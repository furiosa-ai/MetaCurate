"""Detect a rubric criterion that cannot be scored, at the iteration it appears.

Two measurements already see such a criterion every iteration:
`criterion_lift` records how far the ordering moves when a criterion is
ablated, and an unscoreable one sits far below its siblings, e.g.

    Gold-Patch Relation        0.0085   <- unscoreable
    Query/PR Quality           0.0109   <- unscoreable
    Task Grounding             0.0205
    Error Recovery             0.0272
    Edit Justification         0.0302
    Verification Completeness  0.0429

and `criterion_stats` records how concentrated each criterion's scores are:
an unscoreable one piles onto one anchor. Both reach the proposer's prompt.
What neither says is the *cause*: a rubric that tells the scorer to read
`metadata.<field>` while the scorer's view never renders that field makes the
criterion constant, and the natural conclusion is "removal is a rank no-op"
rather than "the field never arrives". This module warns on the symptom and,
when it can, names that cause.

## Two independent detectors

**Degenerate** -- the criterion's scores pile onto one value, at or above
`criterion_stats.DEAD_MODAL_SHARE`. Cause-agnostic: a criterion that returns
a constant orders nothing whatever the reason. The threshold of 0.90
separates the cases cleanly on the distributions behind the lifts above:
0.995 and 0.945 for the two unscoreable axes against 0.635, 0.530, 0.455 and
0.380 for the four that worked.

**Unrendered field** -- low rank lift *and* the criterion's own section reads
a `metadata.<field>` its role does not render. This one names the cause, so it
is worth having even though the degenerate check would usually catch the same
criterion.

Either fires. Both may fire for one criterion, and then one finding carries
both reasons.

## Why the unrendered-field detector needs two conditions

**Low lift alone is not enough.** A criterion can be legitimately low-lift by
being redundant with another one. Warning on lift alone cries wolf, and a
warning nobody trusts is worth less than none.

**A field reference alone is not enough either.** A rubric header may name
`metadata.gold_patch` and `metadata.resolved` precisely to say they are
*absent* on this pool. Mentioning is not requiring, so the scan is scoped to
a criterion's own section and never sees the header.

## Why the lift threshold is relative

Absolute thresholds mis-fire in both directions on the numbers above: 0.005
misses both dead axes (0.0085, 0.0109), and 0.02 flags Task Grounding at
0.0205, which was working. Half the median splits them -- the median of those
six is 0.02385, half is 0.0119 -- and it keeps working on a pool where lift
scales differently, which a fixed number does not. Below three criteria there
is no useful median, so the absolute fallback applies there and only there.

A known gap, recorded rather than papered over: the field detector only sees
the `metadata.<field>` form. A criterion that asks for the same fields in
prose ("the problem statement", "the gold patch") and leaves the `metadata.`
spelling to the rubric header stays silent here, and the degenerate detector
is what catches it.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Iterable, Mapping

from meta_filter.diagnostics.criterion_stats import DEAD_MODAL_SHARE
from meta_filter.visibility.view_policy import METADATA_REF_RE, declared_fields, fields_for

logger = logging.getLogger(__name__)

# Fraction of the median lift below which a criterion counts as not ordering
# anything. See the module docstring for how it was chosen.
DEAD_LIFT_FRACTION = 0.5
# Used only when there are too few criteria for a median to mean anything.
DEAD_LIFT_ABSOLUTE = 0.012
MIN_CRITERIA_FOR_MEDIAN = 3


def _lift_threshold(lifts: list[float]) -> tuple[float, str]:
    if len(lifts) >= MIN_CRITERIA_FOR_MEDIAN:
        median = statistics.median(lifts)
        return DEAD_LIFT_FRACTION * median, f"{DEAD_LIFT_FRACTION:g}x median {median:.4f}"
    return DEAD_LIFT_ABSOLUTE, f"absolute (only {len(lifts)} criteria, no useful median)"


def check(
    rubric_md: str,
    criteria: Iterable[str],
    criterion_rank_shift: Mapping[str, float],
    *,
    criterion_stats: Mapping[str, object] | None = None,
    role: str = "scorer",
) -> list[dict]:
    """Criteria that cannot be scored, with whatever cause is legible.

    `criterion_stats` is the per-candidate diagnostic block from
    `meta_filter.diagnostics.criterion_stats.criterion_stats` (it may be None when too
    few rows carried rubric scores).

    Returns one record per finding, most-degenerate first. Empty is the normal
    result and means nothing needs saying.
    """

    # Imported here, not at module scope, so there is exactly one section
    # splitter in the project: a second copy would drift from the one the
    # cache keys are built on, which is the class of bug this file exists for.
    from meta_filter.core.llm_scorer import _split_rubric_sections

    criteria = list(criteria)
    if not criteria:
        return []

    per_criterion: Mapping[str, Mapping] = {}
    if criterion_stats:
        maybe = criterion_stats.get("per_criterion")
        if isinstance(maybe, Mapping):
            per_criterion = maybe

    scored = [c for c in criteria if c in criterion_rank_shift]
    lifts = [float(criterion_rank_shift[c]) for c in scored]
    threshold, basis = _lift_threshold(lifts) if lifts else (0.0, "no lift measured")

    sections = _split_rubric_sections(rubric_md, criteria)
    findings: list[dict] = []
    for criterion in criteria:
        reasons: list[str] = []
        record: dict = {"criterion": criterion, "role": role}

        stats = per_criterion.get(criterion)
        if isinstance(stats, Mapping) and stats.get("dead"):
            reasons.append("degenerate")
            record["modal_share"] = round(float(stats.get("modal_share", 0.0)), 3)
            record["modal_value"] = stats.get("modal_value")
            record["modal_threshold"] = DEAD_MODAL_SHARE

        if criterion in criterion_rank_shift:
            lift = float(criterion_rank_shift[criterion])
            record["rank_lift"] = round(lift, 4)
            if lift < threshold:
                section = sections.get(criterion) or ""
                referenced = frozenset(METADATA_REF_RE.findall(section))
                missing = sorted(referenced - fields_for(role, declared_fields(section)))
                if missing:
                    reasons.append("unrendered_field")
                    record["missing_fields"] = missing
                    record["threshold"] = round(threshold, 4)
                    record["threshold_basis"] = basis

        if reasons:
            record["reasons"] = reasons
            findings.append(record)

    findings.sort(key=lambda f: (-f.get("modal_share", 0.0), f.get("rank_lift", 0.0)))
    return findings


def render(findings: Iterable[dict]) -> str:
    """One log line per finding, naming the remedy."""

    lines = []
    for f in findings:
        criterion = f["criterion"]
        parts = []
        if "degenerate" in f.get("reasons", []):
            parts.append(
                f"{f['modal_share']:.0%} of its scores sit at "
                f"{f['modal_value']:+g} (>= {f['modal_threshold']:.0%})"
            )
        if "unrendered_field" in f.get("reasons", []):
            fields = ", ".join(f["missing_fields"])
            parts.append(
                f"rank lift {f['rank_lift']:.4f} < {f['threshold']:.4f} "
                f"({f['threshold_basis']}) and its section reads metadata.{fields}, "
                f"which the {f['role']} does not render"
            )
            remedy = (
                f"Either declare it in the section (<!-- reads: {fields} -->, subject "
                f"to the role ceiling in view_policy) or rewrite the anchor to read "
                f"the transcript."
            )
        else:
            remedy = (
                "Check what evidence the anchors ask for and whether the "
                f"{f['role']} is given it; a constant criterion orders nothing "
                "and still costs a scoring call."
            )
        lines.append(f"DEAD AXIS: criterion {criterion!r} — " + "; ".join(parts) + ". " + remedy)
    return "\n".join(lines)


def log(findings: Iterable[dict], iteration: int | None = None) -> None:
    """Warn about each finding. Silent when there is nothing to say."""

    findings = list(findings)
    if not findings:
        return
    prefix = "" if iteration is None else f"iter {iteration}: "
    for line in render(findings).splitlines():
        logger.warning("%s%s", prefix, line)
