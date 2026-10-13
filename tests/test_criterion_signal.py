"""`criterion_lift.single_criterion_concordance` and the criterion-signal table."""

from __future__ import annotations

from meta_filter.core.schemas import Decision, FilterDecision
from meta_filter.diagnostics.criterion_lift import (
    CRITERION_SIGNAL_BUDGET,
    render_criterion_signal,
    single_criterion_concordance,
)

# Six rows; the judge's order is r0 > r1 > ... > r5 (pairs below).
PAIRS = [("r0", "r1"), ("r1", "r2"), ("r2", "r3"), ("r3", "r4"), ("r4", "r5"), ("r0", "r5")]


def _decisions(columns: dict[str, list[float | None]]) -> list[FilterDecision]:
    out = []
    for i in range(6):
        scores = {c: v[i] for c, v in columns.items() if v[i] is not None}
        out.append(
            FilterDecision(trajectory_id=f"r{i}", decision=Decision.ACCEPT, rubric_scores=scores)
        )
    return out


def test_each_criterion_is_scored_alone_against_the_judged_pairs():
    alone = single_criterion_concordance(
        _decisions(
            {
                "Perfect": [5, 4, 3, 2, 1, 0],
                "Inverted": [0, 1, 2, 3, 4, 5],
                "Flat": [1, 1, 1, 1, 1, 1],
                "Partial": [5, 4, None, None, 1, 0],
            }
        ),
        PAIRS,
    )
    assert alone["Perfect"] == {"concordance": 1.0, "n_pairs": 6}
    assert alone["Inverted"] == {"concordance": 0.0, "n_pairs": 6}
    # Every pair an own tie: half credit, the "orders nothing" reading.
    assert alone["Flat"] == {"concordance": 0.5, "n_pairs": 6}
    # Rows the criterion never scored drop their pairs rather than costing them.
    assert alone["Partial"] == {"concordance": 1.0, "n_pairs": 3}


def test_judge_ties_leave_the_exam_and_no_pairs_means_no_table():
    alone = single_criterion_concordance(
        _decisions({"Perfect": [5, 4, 3, 2, 1, 0]}), PAIRS, ties=[("r0", "r1")]
    )
    assert alone["Perfect"]["n_pairs"] == 5
    assert single_criterion_concordance(_decisions({"Perfect": [5, 4, 3, 2, 1, 0]}), []) == {}


def test_the_table_is_sorted_by_alone_concordance_and_marks_dead_axes():
    alone = {
        "Flat": {"concordance": 0.5, "n_pairs": 6},
        "Perfect": {"concordance": 1.0, "n_pairs": 6},
        "Inverted": {"concordance": 0.0, "n_pairs": 6},
    }
    lift = {"Perfect": 0.2103, "Flat": 0.0}
    stats = {"per_criterion": {"Flat": {"dead": True}, "Perfect": {"dead": False}}}
    text = render_criterion_signal(alone, lift, stats)
    lines = text.splitlines()
    assert lines[0].startswith("criterion signal (last winner, over up to 6 judged train pairs)")
    assert "criterion | alone concordance | rank lift when ablated | dead?" in lines[0]
    assert lines[1] == "  Perfect | 1.000 | 0.2103 | live"
    assert lines[2] == "  Flat | 0.500 | 0.0000 | DEAD"
    assert lines[3] == "  Inverted | 0.000 | n/a | live"
    assert "alone ~0.5" in lines[-1]
    assert render_criterion_signal({}, lift, stats) == ""


def test_the_table_is_capped_and_folds_the_tail():
    alone = {
        f"Criterion With A Long Name {i:02d}": {"concordance": 0.9 - i / 100, "n_pairs": 40}
        for i in range(30)
    }
    text = render_criterion_signal(alone, {}, None)
    assert len(text) <= CRITERION_SIGNAL_BUDGET
    assert "... +" in text and "more" in text
    assert "Criterion With A Long Name 00 | 0.900" in text
