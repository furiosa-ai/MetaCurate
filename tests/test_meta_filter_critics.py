"""FN/FP/boundary critics."""

from __future__ import annotations

from meta_filter.core.schemas import Decision, FilterDecision, Trajectory
from meta_filter.critics import (
    BoundaryCritic,
    PrecisionCritic,
    RecallCritic,
    StaticVerdictBackend,
)


def _traj(id_: str) -> Trajectory:
    return Trajectory(trajectory_id=id_, task="t", turns=[])


BACKEND = StaticVerdictBackend(
    {
        "g1": Decision.ACCEPT,
        "g2": Decision.ACCEPT,
        "b1": Decision.REJECT,
        "b2": Decision.REJECT,
    }
)


def test_recall_critic_finds_only_accepts_of_bad():
    trajectories = [
        _traj("g1"),
        _traj("b1"),
        _traj("b2"),
    ]
    decisions = [
        FilterDecision(trajectory_id="g1", decision=Decision.ACCEPT),
        FilterDecision(trajectory_id="b1", decision=Decision.ACCEPT),  # FN
        FilterDecision(trajectory_id="b2", decision=Decision.REVIEW),  # not FN — deferred
    ]
    critic = RecallCritic(backend=BACKEND)
    fn = critic.find_false_negatives(
        trajectories,
        decisions,
        iteration=0,
        filter_version="vX",
    )
    assert [ex.trajectory_id for ex in fn] == ["b1"]
    assert fn[0].failure_type == "false_negative"
    assert fn[0].raw_trajectory is not None


def test_precision_critic_finds_only_rejects_of_good():
    trajectories = [
        _traj("g1"),
        _traj("g2"),
        _traj("b1"),
    ]
    decisions = [
        FilterDecision(trajectory_id="g1", decision=Decision.REJECT),  # FP
        FilterDecision(trajectory_id="g2", decision=Decision.REVIEW),  # not FP
        FilterDecision(trajectory_id="b1", decision=Decision.REJECT),  # correct
    ]
    critic = PrecisionCritic(backend=BACKEND)
    fp = critic.find_false_positives(
        trajectories,
        decisions,
        iteration=0,
        filter_version="vX",
    )
    assert [ex.trajectory_id for ex in fp] == ["g1"]
    assert fp[0].failure_type == "false_positive"


def test_boundary_critic_flags_low_confidence_and_near_threshold():
    trajectories = [_traj("g1"), _traj("g2")]
    decisions = [
        FilterDecision(
            trajectory_id="g1",
            decision=Decision.ACCEPT,
            confidence=0.1,
        ),
        FilterDecision(
            trajectory_id="g2",
            decision=Decision.ACCEPT,
            confidence=0.9,
            rubric_scores={"a": 2.0, "b": -2.0},
        ),
    ]
    critic = BoundaryCritic(confidence_threshold=0.55, score_margin=1.0)
    boundary = critic.find_fragile_cases(
        trajectories,
        decisions,
        iteration=0,
        filter_version="vX",
    )
    ids = {ex.trajectory_id for ex in boundary}
    assert "g1" in ids  # low confidence
    assert "g2" in ids  # competing signals
