"""The dead-axis check: loud on an unanswerable criterion, silent otherwise.

A criterion that reads a field its role never receives cannot be scored, so it
returns a constant and orders nothing. The evidence is on disk every
iteration — `criterion_lift.jsonl` has such an axis far below its siblings and
the criterion-health block has it at a near-total modal share.

The tests below pin what decides whether anyone will trust the warning: each
of the two detectors fires on the real thing, and neither speaks up on the
three ways of resembling the real thing without being it — low lift from
redundancy, a field reference that orders fine, and a header that names a
field only to say it is absent.
"""

from __future__ import annotations

from pathlib import Path

from meta_filter.core.filter_loader import load_filter
from meta_filter.diagnostics.criterion_stats import DEAD_MODAL_SHARE
from meta_filter.diagnostics.dead_axis import DEAD_LIFT_ABSOLUTE, check, render

ROOT = Path(__file__).resolve().parents[1]

# Per-criterion rank lift of the six-criterion fixture filter, two axes dead.
LIFTS = {
    "Gold-Patch Relation": 0.0085,
    "Query/PR Quality": 0.0109,
    "Task Grounding": 0.0205,
    "Error Recovery": 0.0272,
    "Edit Justification": 0.0302,
    "Verification Completeness": 0.0429,
}

RUBRIC_UNDECLARED = """\
# Rubric

## Criterion: Reference Axis
Compare the diff against `metadata.problem_statement`.

## Criterion: Transcript Axis
Read the transcript.

## Criterion: Recovery Axis
Read the transcript.
"""


def test_fires_on_a_criterion_that_reads_a_field_the_scorer_never_gets() -> None:
    findings = check(
        RUBRIC_UNDECLARED,
        ["Reference Axis", "Transcript Axis", "Recovery Axis"],
        {"Reference Axis": 0.004, "Transcript Axis": 0.030, "Recovery Axis": 0.028},
    )
    assert [f["criterion"] for f in findings] == ["Reference Axis"]
    assert findings[0]["missing_fields"] == ["problem_statement"]
    assert "declare it in the section" in render(findings)


def test_silent_once_the_criterion_declares_the_field() -> None:
    declared = RUBRIC_UNDECLARED.replace(
        "## Criterion: Reference Axis\n",
        "## Criterion: Reference Axis\n<!-- reads: problem_statement -->\n",
    )
    assert (
        check(
            declared,
            ["Reference Axis", "Transcript Axis", "Recovery Axis"],
            {"Reference Axis": 0.004, "Transcript Axis": 0.030, "Recovery Axis": 0.028},
        )
        == []
    )


def test_silent_on_low_lift_alone() -> None:
    """Redundancy makes a criterion low-lift legitimately; that is not this."""

    assert (
        check(
            RUBRIC_UNDECLARED.replace("`metadata.problem_statement`", "the submitted diff"),
            ["Reference Axis", "Transcript Axis", "Recovery Axis"],
            {"Reference Axis": 0.0001, "Transcript Axis": 0.030, "Recovery Axis": 0.028},
        )
        == []
    )


def test_silent_on_a_field_reference_that_orders_fine() -> None:
    assert (
        check(
            RUBRIC_UNDECLARED,
            ["Reference Axis", "Transcript Axis", "Recovery Axis"],
            {"Reference Axis": 0.031, "Transcript Axis": 0.030, "Recovery Axis": 0.028},
        )
        == []
    )


def test_header_mentions_are_not_criterion_reads() -> None:
    """A header may name a field to say it is ABSENT; a document-scoped scan
    would flag every criterion below it."""

    rubric = (
        "# Rubric\n\n`metadata.problem_statement` is not available on this pool.\n\n"
        "## Criterion: Transcript Axis\nRead the transcript.\n\n"
        "## Criterion: Recovery Axis\nRead the transcript.\n"
    )
    criteria = ["Transcript Axis", "Recovery Axis"]
    assert check(rubric, criteria, {c: 0.0 for c in criteria}) == []


def test_the_measured_modal_shares_catch_both_dead_axes() -> None:
    """0.995 and 0.945 against 0.635 / 0.530 / 0.455 / 0.380 — the repo's own
    DEAD_MODAL_SHARE of 0.90 separates them with room to spare."""

    loaded = load_filter(ROOT / "tests" / "fixtures" / "filters" / "six_criteria")
    stats = {
        "per_criterion": {
            "Gold-Patch Relation": {"modal_share": 0.995, "modal_value": 0.0, "dead": True},
            "Query/PR Quality": {"modal_share": 0.945, "modal_value": 0.0, "dead": True},
            "Task Grounding": {"modal_share": 0.635, "modal_value": 2.0, "dead": False},
            "Edit Justification": {"modal_share": 0.530, "modal_value": 2.0, "dead": False},
            "Error Recovery": {"modal_share": 0.455, "modal_value": 1.0, "dead": False},
            "Verification Completeness": {"modal_share": 0.380, "modal_value": 2.0, "dead": False},
        }
    }
    findings = check(
        loaded.rubric_md,
        loaded.policy_module.RUBRIC_CRITERIA,
        LIFTS,
        criterion_stats=stats,
    )
    assert {f["criterion"] for f in findings} == {"Gold-Patch Relation", "Query/PR Quality"}
    query = next(f for f in findings if f["criterion"] == "Query/PR Quality")
    assert query["reasons"] == ["degenerate"]
    assert "94%" in render([query]) or "95%" in render([query])
    assert query["modal_threshold"] == DEAD_MODAL_SHARE


def test_absolute_fallback_when_there_is_no_useful_median() -> None:
    """A one-criterion filter has no useful median."""

    single = "## Criterion: Reference Axis\nCompare against `metadata.problem_statement`.\n"
    assert check(single, ["Reference Axis"], {"Reference Axis": DEAD_LIFT_ABSOLUTE / 2})
    assert check(single, ["Reference Axis"], {"Reference Axis": DEAD_LIFT_ABSOLUTE * 2}) == []


def test_criteria_without_a_lift_measurement_are_skipped() -> None:
    assert check(RUBRIC_UNDECLARED, ["Reference Axis"], {}) == []
