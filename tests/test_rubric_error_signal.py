"""Rubric-level error signal for the proposer.

Two readouts that name a CRITERION rather than a pair or a score:

* per misordered pair, the per-criterion difference between the judge's
  pick and the row the filter put above it, with the axis that most opposes
  the judge flagged — the subtraction the proposer was left to do by eye
  over six criteria and five pairs, and did not;
* at promotion, the share of selection rows whose score moved between the
  winning draw and the fresh re-draw, per criterion — the one measurement
  of scorer consistency per axis the loop ever takes, written to
  `iterations/<k>/criterion_stability.json` and shown next to the criterion
  health block.
"""

from __future__ import annotations

import json

from meta_filter.core.schemas import Decision, FilterDecision, Trajectory
from meta_filter.diagnostics.criterion_stats import criterion_stability, render_criterion_stability
from meta_filter.loop.optimizer import _criterion_breakdown, _pair_misorder_examples
from meta_filter.propose.proposer import _format_pair_misorders
from tests.selection_gate_fixture import TRUTH_POLICY
from tests.test_promotion_rescore_is_canonical import TRUTH_POLICY_RESTATED, _llm_optimizer


def _decision(tid: str, scores: dict[str, float]) -> FilterDecision:
    return FilterDecision(
        trajectory_id=tid, decision=Decision.REVIEW, confidence=0.5, rubric_scores=scores
    )


# --- 3a: per-criterion breakdown of a misordered pair -----------------------------


def test_the_breakdown_flags_the_criterion_that_most_opposes_the_judge():
    preferred = _decision("hi", {"Grounding": 2.0, "Verification": -2.0, "Recovery": 1.0})
    other = _decision("lo", {"Grounding": 1.0, "Verification": 2.0, "Recovery": 1.0})
    line = _criterion_breakdown(preferred, other)
    assert line.startswith("per-criterion (judge's pick minus the row ranked above it): ")
    # Most opposing first, flagged; supporting axes after.
    assert line.index("Verification -4 (OPPOSES the judge most)") < line.index("Recovery +0")
    assert "Grounding +1" in line
    assert line.count("OPPOSES") == 1


def test_no_flag_when_every_criterion_agrees_with_the_judge():
    line = _criterion_breakdown(_decision("a", {"X": 2.0}), _decision("b", {"X": 1.0}))
    assert "OPPOSES" not in line and "X +1" in line
    assert _criterion_breakdown(None, _decision("b", {"X": 1.0})) == ""
    assert _criterion_breakdown(_decision("a", {"X": 1.0}), _decision("b", {"Y": 1.0})) == ""


def test_a_wide_rubric_keeps_the_two_most_opposing_axes_within_the_budget():
    names = [f"Criterion Number {i:02d}" for i in range(12)]
    preferred = _decision("hi", {n: -2.0 if i < 5 else 1.0 for i, n in enumerate(names)})
    other = _decision("lo", {n: 2.0 if i < 5 else 0.0 for i, n in enumerate(names)})
    line = _criterion_breakdown(preferred, other)
    assert len(line) <= 220 + len("; 10 other criteria in rubric_scores")
    assert "OPPOSES the judge most" in line
    assert line.count("Criterion Number") == 2
    assert line.endswith("; 10 other criteria in rubric_scores")


def test_misorder_records_carry_the_breakdown_and_the_prompt_renders_it():
    traj = {tid: Trajectory(trajectory_id=tid, task=f"task {tid}") for tid in ("hi", "lo")}
    decisions = {
        "hi": _decision("hi", {"Grounding": 2.0, "Verification": -2.0}),
        "lo": _decision("lo", {"Grounding": 1.0, "Verification": 2.0}),
    }
    fns, fps = _pair_misorder_examples(
        scores={"hi": 10.0, "lo": 20.0},
        resolved_pairs=[("hi", "lo")],
        traj_by_id=traj,
        decisions_by_id=decisions,
        iteration=3,
        filter_version="v",
        rationale_for=lambda winner, loser: "hi verified its fix",
    )
    assert len(fns) == 1 and len(fps) == 1
    for record in (*fns, *fps):
        assert "Verification -4 (OPPOSES the judge most)" in record.criterion_breakdown
    block = _format_pair_misorders(fns, fps, iteration=3)
    lines = block.splitlines()
    judge = next(i for i, line in enumerate(lines) if "the judge's rationale" in line)
    assert lines[judge + 1].startswith(
        "- per-criterion (judge's pick minus the row ranked above it)"
    )
    assert "Verification -4 (OPPOSES the judge most)" in lines[judge + 1]
    # Round-trips through the critic_outputs JSON shape.
    assert json.loads(fns[0].model_dump_json())["criterion_breakdown"] == fns[0].criterion_breakdown


# --- 3b: criterion stability between the two draws -----------------------------------


def test_stability_counts_moved_rows_per_criterion_over_shared_rows_only():
    winning = {
        "r1": {"A": 1.0, "B": 0.0},
        "r2": {"A": 1.0, "B": 2.0},
        "r3": {"A": -1.0, "B": 1.0},
        "only_in_winning": {"A": 0.0, "B": 0.0},
    }
    fresh = {
        "r1": {"A": 1.0, "B": 1.0},
        "r2": {"A": 2.0, "B": 2.0},
        "r3": {"A": -1.0, "B": -1.0},
    }
    out = criterion_stability(winning, fresh)
    assert out["n_rows"] == 3
    assert out["per_criterion"]["A"] == {
        "n": 3,
        "moved": 1,
        "share_moved": 0.3333,
        "mean_abs_delta": 0.3333,
    }
    assert out["per_criterion"]["B"] == {
        "n": 3,
        "moved": 2,
        "share_moved": 0.6667,
        "mean_abs_delta": 1.0,
    }
    assert criterion_stability({}, fresh) == {"n_rows": 0, "per_criterion": {}}


def test_the_stability_block_names_the_unstable_axis_and_is_empty_when_unmeasured():
    out = criterion_stability(
        {f"r{i}": {"Stable": 1.0, "Loose": float(i % 2)} for i in range(10)},
        {f"r{i}": {"Stable": 1.0, "Loose": float((i + 1) % 2)} for i in range(10)},
    )
    block = render_criterion_stability(out, candidate="iter002_00_x")
    lines = block.splitlines()
    assert lines[0].startswith("criterion stability of iter002_00_x: the promoted filter was")
    assert "scored twice on 10 hidden rows" in lines[0]
    assert lines[1].startswith("  * Loose: 100% of rows moved")
    assert "anchors are read inconsistently" in lines[1]
    assert lines[2].startswith("  * Stable: 0% of rows moved") and "inconsistently" not in lines[2]
    assert render_criterion_stability(None) == ""
    assert render_criterion_stability({"n_rows": 0, "per_criterion": {}}) == ""


def test_a_promotion_writes_criterion_stability_and_the_next_prompt_shows_it(tmp_path):
    """The jittered scorer moves every cell between draws, so the promoted
    filter's one criterion reads 100% moved; the file lands beside the
    iteration's other artifacts and the block reaches iteration 1's prompt."""

    optimizer, run, client = _llm_optimizer(
        tmp_path,
        script={0: [("truth", TRUTH_POLICY)], 1: [("restated", TRUTH_POLICY_RESTATED)]},
        iterations=2,
        run_id="stability",
    )
    run_dir = optimizer.run()

    selection = json.loads((run_dir / "iterations/000/selection.json").read_text())
    assert selection["gate"]["decision"] == "accepted"
    payload = json.loads((run_dir / "iterations/000/criterion_stability.json").read_text())
    assert payload["iteration"] == 0 and payload["candidate"] == "iter000_00_truth"
    assert payload["n_rows"] == len(run.selection_ids)
    assert set(payload["per_criterion"]) == {"Truth"}, "only the criteria the policy scores"
    assert payload["per_criterion"]["Truth"]["share_moved"] == 1.0
    assert not (run_dir / "iterations/001/criterion_stability.json").exists(), "no promotion"

    prompts = optimizer.proposer.prompts
    assert "criterion stability of iter000_00_truth" not in prompts[0]
    assert "criterion stability of iter000_00_truth" in prompts[1]
    assert "100% of rows moved" in prompts[1]
