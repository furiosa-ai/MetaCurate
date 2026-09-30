"""Picking a killed run back up.

Two 20-iteration runs have now been killed partway through by the OOM killer.
The agentic proposer is not seed-deterministic, so restarting does not
reproduce the iterations already paid for — they have to be carried, and so
does the history the proposer reasons from.
"""

from __future__ import annotations

import json

import pytest

from meta_filter.resume import completed_iterations, rebuild

SUMMARY_HEADER = (
    "iteration,selected,is_baseline,loss,score,critic_agreement,rubric_quality,"
    "rule_quality,review_rate,complexity_score\n"
)


@pytest.fixture()
def killed_run(tmp_path):
    """A run that wrote three summary rows, then died inside iteration 3."""

    run = tmp_path / "run_test"
    rows = [
        "0,iter000_a,False,300.0,0.70,0.86,0.30,0.50,0.0,29.6\n",
        "1,iter001_b,False,200.0,0.85,0.95,0.45,0.42,0.0,22.7\n",
        "2,iter002_c,False,150.0,0.97,0.95,0.42,1.00,0.0,20.2\n",
    ]
    run.mkdir()
    (run / "summary.csv").write_text(SUMMARY_HEADER + "".join(rows), encoding="utf-8")

    for k, winner in enumerate(["iter000_a", "iter001_b", "iter002_c"]):
        it = run / "iterations" / f"{k:03d}"
        (it / "current_filter").mkdir(parents=True)
        (it / "proposer").mkdir()
        (it / "selection.json").write_text(
            json.dumps({"selected_candidate": winner, "reason": f"won round {k}"}),
            encoding="utf-8",
        )
        (it / "proposer" / "response.json").write_text(
            json.dumps(
                {
                    "diagnosis": [f"finding {k}"],
                    "candidates": [{"id": winner, "notes": f"note {k}"}],
                }
            ),
            encoding="utf-8",
        )

    # Every completed iteration's winner exists as a candidate directory,
    # exactly as the optimizer leaves it.
    for k, winner in enumerate(["iter000_a", "iter001_b", "iter002_c"]):
        (run / "iterations" / f"{k:03d}" / "candidates" / winner).mkdir(parents=True)

    # Iteration 3 started — `save_current_filter` runs at the top of the loop —
    # and then the process was killed.
    (run / "iterations" / "003" / "current_filter").mkdir(parents=True)
    return run


def test_a_summary_row_is_what_marks_an_iteration_done(killed_run):
    """It is appended after selection, so a row means a winner exists — unlike
    an `iterations/NNN/` directory, which appears the moment one starts."""

    assert completed_iterations(killed_run) == [0, 1, 2]


def test_resume_starts_at_the_iteration_that_was_interrupted(killed_run):
    state = rebuild(killed_run)

    assert state.start_iteration == 3
    # The killed iteration already snapshotted the filter it was working from,
    # which is the winner of iteration 2 — no need to rebuild it from diffs.
    assert state.filter_dir == killed_run / "iterations" / "003" / "current_filter"


def test_the_proposers_own_history_is_carried_forward(killed_run):
    """Without it a resumed run proposes as if it were at iteration 0, blind to
    its past diagnoses — which reads as working rather than as failing."""

    state = rebuild(killed_run)

    assert [d["diagnosis"] for d in state.past_diagnoses] == [
        ["finding 0"],
        ["finding 1"],
        ["finding 2"],
    ]
    assert [w["winner"] for w in state.past_winners] == ["iter000_a", "iter001_b", "iter002_c"]
    assert state.past_winners[-1]["winner_notes"] == "note 2"
    assert [q["iteration"] for q in state.quality_trajectory] == [0, 1, 2]
    assert state.quality_trajectory[-1]["rule_quality"] == 1.0


def test_a_run_killed_before_the_next_snapshot_falls_back_to_the_last_winner(
    killed_run,
):
    """`save_current_filter` may not have run yet when the process died.

    The fallback has to be the winner's own candidate directory, NOT
    `iterations/002/current_filter` — that snapshot is written at the *top* of
    iteration 2, so it holds the filter iteration 2 started from. Resuming
    from it would silently discard the winning edits the summary row proves
    iteration 2 made.
    """

    import shutil

    shutil.rmtree(killed_run / "iterations" / "003")

    state = rebuild(killed_run)
    assert state.start_iteration == 3
    assert state.filter_dir == (killed_run / "iterations" / "002" / "candidates" / "iter002_c")


def test_a_run_with_no_finished_iteration_has_nothing_to_resume(tmp_path):
    run = tmp_path / "fresh"
    run.mkdir()
    assert rebuild(run) is None

    run_started = tmp_path / "started"
    (run_started / "iterations" / "000").mkdir(parents=True)
    assert rebuild(run_started) is None
