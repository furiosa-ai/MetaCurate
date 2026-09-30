"""The objective has to keep a gradient in every term it charges for.

A term can go dead early in a run: `rule_quality` pinned at its ceiling,
`review_rate` at its floor, and both `cost` and `complexity` saturated
against budgets the run has already blown past. Then only the remaining
terms move, and the loop optimizes those and lets the rest collapse.

Four of the terms carry weight 0 — a term with no gradient and a term with
no weight are the same charge, and one of the two is honest about it. What is left is charged, so
what is left has to keep moving: these tests hold that line for `complexity`
and pin the zeros as decisions rather than accidents.
"""

from __future__ import annotations

import pytest

from meta_filter.critics.rule_critic import RuleStats, _aggregate_quality
from meta_filter.metrics import DEFAULT_SCORE_WEIGHTS, compute_score
from meta_filter.schemas import FilterMetrics


def _metrics(**over):
    """A FilterMetrics with every required field filled; override what matters."""

    base = dict(
        total=200,
        critic_accept_total=100,
        critic_reject_total=100,
        accepted=100,
        rejected=100,
        reviewed=0,
        agree_accept=88,
        agree_reject=88,
        filter_accept_critic_reject=12,
        filter_reject_critic_accept=12,
        critic_reject_recall=0.88,
        critic_accept_retention=0.88,
        disputed_accept_rate=0.1,
        disputed_reject_rate=0.1,
        review_rate=0.0,
        critic_agreement=0.88,
        rubric_quality=0.45,
        rule_quality=0.8,
        estimated_filter_cost=515.0,
        complexity_score=41.5,
    )
    return FilterMetrics(**{**base, **over})


def _stat(name, fires, precision, redundancy=0.0):
    return RuleStats(
        name=name,
        fires=fires,
        truth_agreed=int(fires * precision),
        precision=precision,
        recall=0.0,
        redundancy=redundancy,
    )


def test_deleting_a_good_rule_is_now_a_loss_not_a_gain():
    """Three working rules, then the 95%- and 81%-precision ones deleted to
    keep one perfect rule: a plain mean pays 0.6358 -> 1.0 for doing so."""

    kept = [
        _stat("resolved_and_verified", 82, 1.00, 0.012),
        _stat("p2p_regression", 22, 0.9545, 0.111),
        _stat("unresolved_thrashing", 48, 0.8125, 0.111),
    ]
    dropped = [_stat("resolved_and_verified", 82, 1.00, 0.0)]

    # Fire sets overlap, so the union is smaller than the sum: ~140 of 200.
    kept_q = _aggregate_quality(kept, covered=140, labelled=200)
    dropped_q = _aggregate_quality(dropped, covered=82, labelled=200)

    assert kept_q > dropped_q, "deleting two working rules must cost, not pay"
    # The margin has to survive the 0.20 score weight to steer a proposer.
    assert (kept_q - dropped_q) * 0.20 > 0.02


def test_precision_alone_would_still_reward_deletion():
    """Without the coverage factor, fire-count weighting is not enough — this
    pins why the coverage term exists rather than just the weighting."""

    kept = [
        _stat("resolved_and_verified", 82, 1.00, 0.012),
        _stat("p2p_regression", 22, 0.9545, 0.111),
        _stat("unresolved_thrashing", 48, 0.8125, 0.111),
    ]
    dropped = [_stat("resolved_and_verified", 82, 1.00, 0.0)]

    # labelled=0 disables the coverage factor.
    assert _aggregate_quality(dropped) > _aggregate_quality(kept)


def test_a_rule_that_decides_most_of_the_pool_dominates_a_rare_one():
    """Weighting is by fire count, so a rule that touches one trajectory
    cannot drag the aggregate the way it did under a plain mean."""

    workhorse_plus_dud = [
        _stat("workhorse", 190, 0.95),
        _stat("rare_dud", 1, 0.0),
    ]
    # Under a mean this is (0.95 + 0.0)/2 = 0.475 — the one-shot dud halves it.
    assert _aggregate_quality(workhorse_plus_dud) > 0.9


def test_rules_that_never_fire_neither_help_nor_hurt():
    with_dead_rules = [
        _stat("live", 50, 0.9),
        _stat("dead_a", 0, 0.0),
        _stat("dead_b", 0, 0.0),
    ]
    assert _aggregate_quality(with_dead_rules) == pytest.approx(0.9)


def test_fewer_than_two_rules_stays_neutral():
    """Redundancy is a between-rule quantity, so a single-rule filter is not
    comparable to a multi-rule one."""

    assert _aggregate_quality([]) == 0.5


@pytest.mark.parametrize(
    "cost,complexity",
    [(515.0, 41.5), (570.8, 36.5)],  # sizes observed on grown filters
)
def test_complexity_still_has_gradient_at_observed_filter_sizes(cost, complexity):
    """A saturated normalizer is a constant charge with no signal — the run
    pays it and cannot reduce it, so it steers nothing.

    Complexity carries this on its own now: it is the only size control left
    in the objective, so a budget it had already blown past would leave
    nothing at all pushing back on filter growth.
    """

    base = _metrics(estimated_filter_cost=cost, complexity_score=complexity)
    simpler = _metrics(estimated_filter_cost=cost, complexity_score=complexity * 0.8)

    assert compute_score(simpler) > compute_score(base), "complexity term is saturated"


@pytest.mark.parametrize("cost", [515.0, 570.8])
def test_cost_is_deliberately_free_now(cost):
    """`estimated_filter_cost` is not charged, and that is a decision.

    It measures cache state rather than deployment cost — it freezes once
    every later candidate re-reads the same cached cells — and redefined as
    deployment cost it duplicates
    `complexity`, whose breakdown already carries rubric_criteria,
    rubric_chars, rule_count, rules_loc and policy_loc.
    """

    base = _metrics(estimated_filter_cost=cost)
    cheaper = _metrics(estimated_filter_cost=cost * 0.8)

    assert compute_score(cheaper) == compute_score(base)
    assert DEFAULT_SCORE_WEIGHTS["lambda_cost"] == 0.0
    # Re-enabling it must still work: the term is zero-weighted, not deleted.
    assert compute_score(cheaper, weights={"lambda_cost": 0.20}) > compute_score(
        base, weights={"lambda_cost": 0.20}
    )


def test_only_concordance_and_complexity_are_charged_by_default():
    """The objective, stated as an equality.

    The redundancy penalty is applied in the optimizer rather than here, and
    only reported, because it reads per-criterion score correlations rather
    than anything on `FilterMetrics`.
    """

    live = {k: v for k, v in DEFAULT_SCORE_WEIGHTS.items() if k.startswith("lambda_") and v}
    assert live == {"lambda_complexity": 0.15}

    m = _metrics(critic_agreement=0.88, complexity_score=60.0)
    assert abs(compute_score(m) - (0.88 - 0.15 * 0.5)) < 1e-9


def test_the_budgets_are_above_every_filter_observed_so_far():
    assert DEFAULT_SCORE_WEIGHTS["cost_budget"] > 570.8
    assert DEFAULT_SCORE_WEIGHTS["complexity_budget"] > 41.5
