"""Filter-vs-critic agreement metric tests."""

from __future__ import annotations

from meta_filter.core.schemas import Decision, FilterDecision, Trajectory
from meta_filter.gate.metrics import compute_metrics, scalar_loss


def _traj(id_: str) -> Trajectory:
    return Trajectory(trajectory_id=id_, task="t", turns=[])


VERDICTS = {
    "g1": Decision.ACCEPT,
    "g2": Decision.ACCEPT,
    "b1": Decision.REJECT,
}


def _dec(id_: str, decision: Decision, confidence: float = 0.5) -> FilterDecision:
    return FilterDecision(
        trajectory_id=id_,
        decision=decision,
        confidence=confidence,
    )


def test_all_correct_gives_perfect_metrics():
    trajectories = [
        _traj("g1"),
        _traj("g2"),
        _traj("b1"),
    ]
    decisions = [
        _dec("g1", Decision.ACCEPT),
        _dec("g2", Decision.ACCEPT),
        _dec("b1", Decision.REJECT),
    ]
    m = compute_metrics(trajectories, decisions, VERDICTS)
    assert m.filter_accept_critic_reject == 0
    assert m.filter_reject_critic_accept == 0
    assert m.critic_reject_recall == 1.0
    assert m.critic_accept_retention == 1.0
    assert m.review_rate == 0.0


def test_fn_and_fp_are_symmetric():
    trajectories = [
        _traj("g1"),
        _traj("b1"),
    ]
    decisions = [
        _dec("g1", Decision.REJECT),  # FP
        _dec("b1", Decision.ACCEPT),  # FN
    ]
    m = compute_metrics(trajectories, decisions, VERDICTS)
    assert m.filter_accept_critic_reject == 1
    assert m.filter_reject_critic_accept == 1
    assert m.critic_reject_recall == 0.0
    assert m.critic_accept_retention == 0.0
    assert m.disputed_accept_rate == 1.0
    assert m.disputed_reject_rate == 1.0


def test_review_rate_counted_but_not_as_error():
    trajectories = [
        _traj("g1"),
        _traj("b1"),
    ]
    decisions = [
        _dec("g1", Decision.REVIEW),
        _dec("b1", Decision.REVIEW),
    ]
    m = compute_metrics(trajectories, decisions, VERDICTS)
    assert m.filter_accept_critic_reject == 0
    assert m.filter_reject_critic_accept == 0
    assert m.review_rate == 1.0
    assert m.reviewed == 2


def test_scalar_loss_penalizes_fp_and_fn_together():
    trajectories = [
        _traj("g1"),
        _traj("b1"),
    ]
    decisions = [
        _dec("g1", Decision.REJECT),  # FP
        _dec("b1", Decision.ACCEPT),  # FN
    ]
    m = compute_metrics(trajectories, decisions, VERDICTS)
    weights = {
        "lambda_fn": 5.0,
        "lambda_fp": 4.0,
        "lambda_review": 0.5,
        "lambda_cost": 0.1,
        "lambda_complexity": 0.05,
    }
    loss = scalar_loss(m, weights)
    # 5*1 + 4*1 = 9
    assert loss == 9.0
