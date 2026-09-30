"""The criterion-health block carries the whole |rho| matrix and says WHY a
dead axis is dead.

A filter can carry two axes at 100 % `+0` for iterations with "DEAD AXIS"
printed beside each and nothing about the cause, and redundancy reduced to
one pair hides the rest. These tests pin the matrix (every live pair, `!` at
the threshold), the two dead-axis reasons, and the character budget the
matrix is paid for out of.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from meta_filter.criterion_stats import (
    DEAD_REASON_ABSENT,
    DEAD_REASON_ONE_LEVEL,
    HEALTH_BLOCK_BUDGET,
    criterion_stats,
    dead_axes,
    dead_axis_reasons,
    redundant_pairs,
    render_corr_matrix,
    render_criterion_health,
    short_names,
)

RUBRIC = """# Rubric

Doubt is 0.

## Criterion: Task Grounding

Does the agent read the task?

## Criterion: Gold-Patch Relation
<!-- reads: gold_patch -->

Compare the agent's diff with `metadata.gold_patch`.

## Criterion: Query/PR Quality

Is the problem statement specific and the gold patch coherent?

## Criterion: Error Recovery

Does the agent recover after a failing command?
"""
CRITERIA = ["Task Grounding", "Gold-Patch Relation", "Query/PR Quality", "Error Recovery"]


def _decisions(columns: dict[str, list[float]]):
    n = len(next(iter(columns.values())))
    return [
        SimpleNamespace(rubric_scores={c: float(v[i]) for c, v in columns.items()})
        for i in range(n)
    ]


def _spread(seed: int, n: int = 50) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in rng.integers(-2, 3, size=n)]


def _stats():
    """Six axes: two independent, one a copy of the first, one dead and
    constant (not live), one dead with a single stray row (live by std, dead
    by share), and one more independent axis."""

    live = _spread(1)
    return criterion_stats(
        _decisions(
            {
                "Task Grounding": live,
                "Error Recovery": _spread(2),
                "Edit Justification": _spread(3),
                "Verification Completeness": list(live),
                "Gold-Patch Relation": [0.0] * 50,
                "Query/PR Quality": [0.0] * 49 + [1.0],
            }
        )
    )


def _rows(n: int = 12, **metadata):
    return [SimpleNamespace(metadata=dict(metadata)) for _ in range(n)]


def test_the_stats_record_the_full_matrix_over_the_live_axes():
    stats = _stats()
    # Live is by standard deviation: the constant axis is out, the 49-zeros
    # one is in (and dead by share — both facts are true of it).
    assert stats["live_criteria"] == [
        "Edit Justification",
        "Error Recovery",
        "Query/PR Quality",
        "Task Grounding",
        "Verification Completeness",
    ]
    matrix = stats["abs_corr"]
    assert len(matrix) == 5 and all(len(row) == 5 for row in matrix)
    i, j = 3, 4  # Task Grounding vs its copy
    assert matrix[i][j] == matrix[j][i] > 0.99
    assert redundant_pairs(stats) == {frozenset({"Task Grounding", "Verification Completeness"})}
    assert dead_axes(stats) == ["Gold-Patch Relation", "Query/PR Quality"]


def test_the_matrix_renders_every_pair_and_marks_the_threshold():
    lines = render_corr_matrix(_stats(), threshold=0.7)
    text = "\n".join(lines)
    assert (
        "legend: EJ=Edit Justification, ER=Error Recovery, QPQ=Query/PR Quality, "
        "TG=Task Grounding, VC=Verification Completeness"
    ) in text
    # Two decimals; the redundant pair is marked in both cells, the rest not.
    assert text.count("1.00!") == 2
    row_tg = next(line for line in lines if line.strip().startswith("TG"))
    assert "!" in row_tg
    row_er = next(line for line in lines if line.strip().startswith("ER"))
    assert "!" not in row_er
    assert "-" in row_er  # the diagonal


def test_short_names_are_initials_and_collisions_get_a_digit():
    assert short_names(["Task Grounding", "Query/PR Quality", "Test Grounding"]) == {
        "Task Grounding": "TG",
        "Query/PR Quality": "QPQ",
        "Test Grounding": "TG2",
    }


def test_a_dead_axis_whose_input_is_absent_on_every_row_says_so():
    reasons = dead_axis_reasons(
        RUBRIC,
        CRITERIA,
        ["Gold-Patch Relation", "Query/PR Quality", "Error Recovery"],
        _rows(resolved=True, exit_status="Submitted"),
    )
    # Declared field, `metadata.` reference, and the prose alias all count.
    assert reasons["Gold-Patch Relation"] == {
        "reason": DEAD_REASON_ABSENT,
        "fields": ["gold_patch"],
    }
    assert reasons["Query/PR Quality"]["reason"] == DEAD_REASON_ABSENT
    assert reasons["Query/PR Quality"]["fields"] == ["gold_patch", "problem_statement"]
    # A criterion that asks for no field cannot be dead for want of one.
    assert reasons["Error Recovery"] == {"reason": DEAD_REASON_ONE_LEVEL, "fields": []}


def test_an_input_present_on_one_row_is_not_absent():
    rows = _rows(gold_patch="")
    rows.append(SimpleNamespace(metadata={"gold_patch": "diff --git a b"}))
    reasons = dead_axis_reasons(RUBRIC, CRITERIA, ["Gold-Patch Relation"], rows)
    assert reasons["Gold-Patch Relation"]["reason"] == DEAD_REASON_ONE_LEVEL


def test_the_block_prints_the_reason_beside_the_dead_axis():
    text = render_criterion_health(
        _stats(),
        threshold=0.7,
        dead_reasons={
            "Gold-Patch Relation": {"reason": DEAD_REASON_ABSENT, "fields": ["gold_patch"]},
            "Query/PR Quality": {"reason": DEAD_REASON_ONE_LEVEL, "fields": []},
        },
    )
    assert "Gold-Patch Relation: 100% of scores sit at +0 — DEAD AXIS: its input" in text
    assert "(metadata.gold_patch) is absent on every row" in text
    assert "Query/PR Quality: 98% of scores sit at +0 — DEAD AXIS: scorer gives one level" in text
    assert "anchors do not separate these trajectories" in text
    assert "|rho| between every pair of live axes" in text
    assert "most correlated pair: Task Grounding vs Verification Completeness" in text


def test_a_missing_reason_falls_back_to_the_one_level_wording():
    text = render_criterion_health(_stats(), threshold=0.7)
    assert text.count("DEAD AXIS: scorer gives one level") == 2


def test_the_block_stays_within_its_budget_by_dropping_live_levels_first():
    tight = 1900
    full = render_criterion_health(_stats(), threshold=0.7, budget=10_000)
    assert len(full) > tight, "the fixture has to overflow for the trim to show"
    trimmed = render_criterion_health(_stats(), threshold=0.7, budget=tight)
    assert len(trimmed) <= tight
    # At the default ceiling the same rubric renders whole.
    assert len(render_criterion_health(_stats(), threshold=0.7)) <= HEALTH_BLOCK_BUDGET
    # The matrix, its legend and every dead axis's reason survive the trim.
    assert "legend:" in trimmed
    assert trimmed.count("DEAD AXIS") == 2
    assert "1.00!" in trimmed


def test_a_tiny_budget_drops_the_legend_last_and_never_the_matrix():
    text = render_criterion_health(_stats(), threshold=0.7, budget=100)
    assert "score levels:" not in text
    assert "legend:" not in text
    assert "1.00!" in text
    assert text.count("DEAD AXIS") == 2
