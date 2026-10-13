"""Per-item rank lift: how far the ordering moves when one item is ablated."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

from meta_filter.core.schemas import FilterDecision, Trajectory
from meta_filter.diagnostics.criterion_lift import measure_ranking, render_ranking


def _traj(tid: str) -> Trajectory:
    return Trajectory(trajectory_id=tid, metadata={})


def _loaded(decide, rules=()):
    policy = ModuleType("policy")
    policy.decide = decide
    return SimpleNamespace(policy_module=policy, rule_functions=list(rules))


def _decision(tid: str, score: float, scores: dict) -> FilterDecision:
    return FilterDecision(
        trajectory_id=tid, decision="REVIEW", rank_score=score, rubric_scores=scores
    )


def _weighted(traj, scores, triggered):
    score = 2.0 * scores.get("verification", 0.0) + scores.get("grounding", 0.0)
    if any(t.get("name") == "rule_veto" for t in triggered):
        score -= 10.0
    return {"decision": "REVIEW", "score": score}


ROWS = {
    "a": {"verification": 2.0, "grounding": -2.0},
    "b": {"verification": 0.0, "grounding": 2.0},
    "c": {"verification": 1.0, "grounding": 0.0},
}


def _decisions(loaded):
    return [
        _decision(tid, _weighted(_traj(tid), scores, [])["score"], scores)
        for tid, scores in ROWS.items()
    ]


def test_a_criterion_that_orders_rows_shifts_the_ranking():
    loaded = _loaded(_weighted)
    report = measure_ranking(loaded, [_traj(t) for t in ROWS], _decisions(loaded))

    assert report["rows_ranked"] == 3
    assert report["criterion_rank_shift"]["verification"] > 0
    assert report["criterion_rank_shift"]["grounding"] > 0


def test_an_unread_criterion_shifts_nothing():
    def verification_only(traj, scores, triggered):
        return {"decision": "REVIEW", "score": scores.get("verification", 0.0)}

    loaded = _loaded(verification_only)
    decisions = [_decision(tid, scores["verification"], scores) for tid, scores in ROWS.items()]
    report = measure_ranking(loaded, [_traj(t) for t in ROWS], decisions)

    assert report["criterion_rank_shift"]["grounding"] == 0
    assert report["criterion_rank_shift"]["verification"] > 0


def test_rows_without_a_rank_score_are_skipped():
    loaded = _loaded(_weighted)
    unscored = [FilterDecision(trajectory_id="a", decision="REVIEW", rubric_scores=ROWS["a"])]
    report = measure_ranking(loaded, [_traj("a")], unscored)
    assert report["rows_ranked"] == 0
    assert "no rows carried a rank score" in render_ranking(report)


def test_a_raising_policy_never_breaks_the_measurement():
    def boom(traj, scores, triggered):
        raise KeyError("verification")

    loaded = _loaded(boom)
    decisions = [_decision("a", 1.0, {"v": 1.0}), _decision("b", 2.0, {"v": 0.0})]
    report = measure_ranking(loaded, [_traj("a"), _traj("b")], decisions)
    assert report["rows_ranked"] == 2


def test_the_rendered_line_names_every_criterion():
    loaded = _loaded(_weighted)
    line = render_ranking(measure_ranking(loaded, [_traj(t) for t in ROWS], _decisions(loaded)))
    assert "per criterion" in line
    assert "verification" in line and "grounding" in line
