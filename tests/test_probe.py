"""Fixed probe set: stratified build, flip rate, persistence, prompt rendering."""

from __future__ import annotations

from meta_filter.probe import (
    ProbeSet,
    append_report,
    flip_rate,
    latest_decisions,
    load_reports,
    render_for_prompt,
)
from meta_filter.schemas import Decision, FilterDecision, Trajectory, Turn
from meta_filter.trajectory import dump_trajectories


def _traj(instance: str, attempt: int = 0, resolved: bool = False, host: str = "host-local"):
    return Trajectory(
        trajectory_id=f"{instance}::att{attempt}.{host}",
        task=f"fix {instance}",
        turns=[Turn(role="assistant", action="read", content="opened a file")],
        metadata={
            "instance_id": instance,
            "host": host,
            "attempt_idx": attempt,
            "resolved": resolved,
        },
    )


def _pool() -> list[Trajectory]:
    """Six instances, two attempts each: 2 all-solved, 2 partial, 2 none-solved."""

    pool = []
    for i in range(2):
        pool += [_traj(f"easy-{i}", 0, True), _traj(f"easy-{i}", 1, True)]
    for i in range(2):
        pool += [_traj(f"mid-{i}", 0, True), _traj(f"mid-{i}", 1, False)]
    for i in range(2):
        pool += [_traj(f"hard-{i}", 0, False), _traj(f"hard-{i}", 1, False)]
    return pool


class FakeSolver:
    """Returns canned decisions, ignoring the filter — probe only records them."""

    def __init__(self, decisions: dict[str, Decision]):
        self.decisions = decisions

    def run(self, loaded_filter, trajectories):
        return [
            FilterDecision(
                trajectory_id=t.trajectory_id,
                decision=self.decisions.get(t.trajectory_id, Decision.REVIEW),
            )
            for t in trajectories
        ]


# ---------------------------------------------------------------------------
# Flip rate
# ---------------------------------------------------------------------------


def test_flip_rate_counts_changed_common_ids():
    prev = {"a": "ACCEPT", "b": "REJECT", "c": "ACCEPT"}
    new = {"a": "ACCEPT", "b": "ACCEPT", "c": "REJECT"}
    assert flip_rate(prev, new) == 2 / 3


def test_flip_rate_on_disjoint_or_missing_sets_is_zero():
    assert flip_rate(None, {"a": "ACCEPT"}) == 0.0
    assert flip_rate({}, {"a": "ACCEPT"}) == 0.0
    assert flip_rate({"x": "ACCEPT"}, {"y": "REJECT"}) == 0.0


# ---------------------------------------------------------------------------
# Evaluation + persistence round-trip
# ---------------------------------------------------------------------------


def _probe_set(tmp_path) -> ProbeSet:
    selected = _pool()[::2]
    path = tmp_path / "probe_set.jsonl"
    dump_trajectories(selected, path)
    return ProbeSet.from_path(path)


def test_evaluate_reports_accept_rate_and_count(tmp_path):
    probe = _probe_set(tmp_path)
    accept_ids = [t.trajectory_id for t in probe.trajectories[:2]]
    solver = FakeSolver(dict.fromkeys(accept_ids, Decision.ACCEPT))

    report = probe.evaluate(loaded_filter=None, solver=solver)

    assert report.accept_rate == 2 / 6
    assert set(report.decisions) == {t.trajectory_id for t in probe.trajectories}
    assert (report.accepts, report.total) == (2, 6)


def test_report_persistence_round_trip(tmp_path):
    probe = _probe_set(tmp_path)
    run_dir = tmp_path / "run"
    ids = [t.trajectory_id for t in probe.trajectories]

    first = probe.evaluate(None, FakeSolver(dict.fromkeys(ids, Decision.ACCEPT)))
    row1 = append_report(run_dir, 1, first, prev=latest_decisions(run_dir))
    assert row1["flip_rate_vs_prev"] == 0.0

    # Half the probe flips to REJECT in the next iteration.
    flipped = dict.fromkeys(ids, Decision.ACCEPT) | dict.fromkeys(ids[:3], Decision.REJECT)
    second = probe.evaluate(None, FakeSolver(flipped))
    row2 = append_report(run_dir, 2, second, prev=latest_decisions(run_dir))

    assert row2["flip_rate_vs_prev"] == 0.5
    assert latest_decisions(run_dir) == second.decisions
    rows = load_reports(run_dir)
    assert [r["iteration"] for r in rows] == [1, 2]
    assert rows[1]["decisions"] == second.decisions


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _rows(n: int) -> list[dict]:
    return [
        {
            "iteration": i,
            "accept_rate": 0.5,
            "flip_rate_vs_prev": 0.1,
            "accepts": 3,
            "total": 6,
            "decisions": {"t1": "ACCEPT"},
        }
        for i in range(1, n + 1)
    ]


def test_render_for_prompt_is_a_trend_table_not_raw_decisions():
    text = render_for_prompt(_rows(3), max_chars=2000)
    assert "accept_rate" in text
    assert "| 3/6" in text
    assert "t1" not in text  # raw decisions never leak into prompts
    for i in (1, 2, 3):
        assert f"\n{i:>4} |" in text


def test_render_for_prompt_truncates_to_budget_keeping_newest():
    rows = _rows(50)
    text = render_for_prompt(rows, max_chars=300)
    assert len(text) <= 300
    assert "\n  50 |" in text  # newest row survives truncation
    assert "\n   1 |" not in text  # oldest is dropped first
    assert render_for_prompt([], max_chars=300) == ""


def test_fixed_validation_pairs_deterministic_and_capped() -> None:
    from meta_filter.probe import fixed_validation_pairs

    ids = [f"t{i}" for i in range(30)]
    a = fixed_validation_pairs(ids, n_pairs=50, seed=4242)
    b = fixed_validation_pairs(list(reversed(ids)), n_pairs=50, seed=4242)
    assert a == b  # order of input ids does not matter
    assert len(a) == 50
    assert all(x < y for x, y in a)  # canonical unordered form


def test_fixed_validation_pairs_small_universe_returns_all() -> None:
    from meta_filter.probe import fixed_validation_pairs

    pairs = fixed_validation_pairs(["a", "b", "c"], n_pairs=100)
    assert pairs == [("a", "b"), ("a", "c"), ("b", "c")]
