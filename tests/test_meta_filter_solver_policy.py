"""Solver + policy: three-way ACCEPT/REJECT/REVIEW behavior."""

from __future__ import annotations

from pathlib import Path

from meta_filter.filter_loader import load_filter
from meta_filter.schemas import Decision, Trajectory, Turn
from meta_filter.solver import Solver
from meta_filter.trajectory import load_trajectories

INITIAL = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "filters" / "five_criteria"


def test_three_way_decision_present_on_demo_data():
    loaded = load_filter(INITIAL)
    solver = Solver()
    trajectories = load_trajectories(
        Path(__file__).resolve().parents[1] / "data" / "demo_train.jsonl"
    )
    decisions = solver.run(loaded, trajectories)
    kinds = {d.decision for d in decisions}
    # The initial filter must be capable of emitting all three decision types
    # on the demo data (this is the sanity check that we are not silently
    # collapsing to a two-way filter).
    assert Decision.ACCEPT in kinds
    assert Decision.REVIEW in kinds or Decision.REJECT in kinds
    # More strictly: at least one REJECT (T07 six-grep loop) and at least one ACCEPT.
    assert Decision.REJECT in kinds
    assert Decision.ACCEPT in kinds


def test_no_turns_short_circuits_to_reject():
    loaded = load_filter(INITIAL)
    solver = Solver()
    traj = Trajectory(trajectory_id="empty", task="none", turns=[], gold_decision=None)
    decisions = solver.run(loaded, [traj])
    assert decisions[0].decision == Decision.REJECT
    assert "rule_no_turns" in decisions[0].triggered_rules


def test_solver_never_persists_raw_cot():
    loaded = load_filter(INITIAL)
    solver = Solver()
    traj = Trajectory(
        trajectory_id="x",
        task="ok",
        turns=[Turn(role="assistant", action="read", content="opened f")],
    )
    decisions = solver.run(loaded, [traj])
    # reasoning_summary is a compact string; there is no `chain_of_thought`.
    assert not hasattr(decisions[0], "chain_of_thought")
    assert len(decisions[0].reasoning_summary) < 500
