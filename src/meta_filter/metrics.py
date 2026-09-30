"""Filter-vs-critic agreement metrics.

Symmetric in both disagreement directions so the critics and the selector
treat "filter accepted what the critic rejected" and its mirror as equally
first-class signals.

The verdict map is an explicit argument rather than a field read off each
trajectory. That is deliberate: this experiment has no ground truth, and an
attribute named `gold_decision` sitting on the data invited exactly the
confusion of treating a critic's opinion as a label.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from meta_filter.schemas import Decision, FilterDecision, FilterMetrics, Trajectory


def _rate(numer: int, denom: int) -> float:
    return float(numer) / denom if denom > 0 else 0.0


def compute_metrics(
    trajectories: Iterable[Trajectory],
    decisions: Iterable[FilterDecision],
    verdicts: Mapping[str, Decision | None],
    complexity_score: float = 0.0,
    complexity_breakdown: dict[str, int] | None = None,
    regression_must_accept_violations: int = 0,
    regression_must_reject_violations: int = 0,
    estimated_filter_cost: float = 0.0,
) -> FilterMetrics:
    """Compare a candidate's decisions against the critic's verdicts.

    Args:
        trajectories: The pool the decisions were produced on.
        decisions: One `FilterDecision` per trajectory the filter judged.
        verdicts: Critic verdict per trajectory id. A missing id or a `None`
            value means the critic did not adjudicate that trajectory; it
            counts toward `total` but toward no agreement counter.
        complexity_score: Aggregate filter-complexity penalty.
        complexity_breakdown: Per-signal complexity counts.
        regression_must_accept_violations: Replay violations on must-accept memory.
        regression_must_reject_violations: Replay violations on must-reject memory.
        estimated_filter_cost: Estimated LLM cost of running this filter.

    Returns:
        A populated `FilterMetrics`.
    """

    traj_map = {t.trajectory_id: t for t in trajectories}
    decision_list = list(decisions)

    critic_accept_total = sum(1 for tid in traj_map if verdicts.get(tid) == Decision.ACCEPT)
    critic_reject_total = sum(1 for tid in traj_map if verdicts.get(tid) == Decision.REJECT)

    accepted = rejected = reviewed = 0
    agree_accept = agree_reject = 0
    filter_accept_critic_reject = filter_reject_critic_accept = 0
    llm_calls = llm_tokens = 0

    for d in decision_list:
        if d.trajectory_id not in traj_map:
            continue
        llm_calls += d.llm_calls
        llm_tokens += d.llm_tokens
        verdict = verdicts.get(d.trajectory_id)

        if d.decision == Decision.ACCEPT:
            accepted += 1
            if verdict == Decision.ACCEPT:
                agree_accept += 1
            elif verdict == Decision.REJECT:
                filter_accept_critic_reject += 1
        elif d.decision == Decision.REJECT:
            rejected += 1
            if verdict == Decision.REJECT:
                agree_reject += 1
            elif verdict == Decision.ACCEPT:
                filter_reject_critic_accept += 1
        elif d.decision == Decision.REVIEW:
            reviewed += 1

    adjudicated = critic_accept_total + critic_reject_total
    return FilterMetrics(
        total=len(decision_list),
        critic_accept_total=critic_accept_total,
        critic_reject_total=critic_reject_total,
        accepted=accepted,
        rejected=rejected,
        reviewed=reviewed,
        agree_accept=agree_accept,
        agree_reject=agree_reject,
        filter_accept_critic_reject=filter_accept_critic_reject,
        filter_reject_critic_accept=filter_reject_critic_accept,
        critic_reject_recall=_rate(agree_reject, critic_reject_total),
        critic_accept_retention=_rate(agree_accept, critic_accept_total),
        disputed_accept_rate=_rate(filter_accept_critic_reject, accepted),
        disputed_reject_rate=_rate(filter_reject_critic_accept, rejected),
        review_rate=_rate(reviewed, len(decision_list)),
        critic_agreement=_rate(agree_accept + agree_reject, adjudicated),
        llm_calls=llm_calls,
        llm_tokens=llm_tokens,
        estimated_filter_cost=estimated_filter_cost,
        regression_must_accept_violations=regression_must_accept_violations,
        regression_must_reject_violations=regression_must_reject_violations,
        complexity=complexity_breakdown or {},
        complexity_score=complexity_score,
    )


def scalar_loss(
    metrics: FilterMetrics,
    weights: dict[str, float],
) -> float:
    """Weighted scalar objective. Lower is better. Retained for v1
    compatibility during the loss→score transition."""

    return (
        weights.get("lambda_fn", 1.0) * metrics.filter_accept_critic_reject
        + weights.get("lambda_fp", 1.0) * metrics.filter_reject_critic_accept
        + weights.get("lambda_review", 0.0) * metrics.reviewed
        + weights.get("lambda_cost", 0.0) * metrics.estimated_filter_cost
        + weights.get("lambda_complexity", 0.0) * metrics.complexity_score
    )


# Score weights. Four of the six terms are 0, leaving
#
#   score = pair_concordance
#         - 0.15 * min(1, complexity / 120)
#
# (the redundancy penalty is reported beside it, not charged; see
# `criterion_stats.charge_redundancy`).
#
# Why each of the four went to zero, so none of them comes back by accident:
#
# * `rubric_quality` / `rule_quality` are LLM critics grading prose. They cost
#   a call per iteration and score a text, not a behaviour, so a candidate can
#   buy them with rhetoric. They stay as DIAGNOSTICS — the critics still run —
#   but nothing selects on them.
# * `review_rate` is a gate-mode quantity. The ranking contract is a score for
#   every row, so there is no defer-to-human class left to charge for.
# * `cost` measured `estimated_filter_cost`, which measures CACHE STATE: it
#   freezes once every later candidate re-reads the same cached cells.
#   Redefined as deployment cost it duplicates
#   `complexity`, whose breakdown already carries rubric_criteria,
#   rubric_chars, rule_count, rules_loc and policy_loc.
#
# `complexity` therefore stays, and is the ONLY size control in the
# objective. Keep the lesson on its coefficients: deleting an axis must not be
# the cheapest way to gain score.
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "lambda_rubric_quality": 0.0,
    "lambda_rule_quality": 0.0,
    "lambda_review": 0.0,
    "lambda_cost": 0.0,
    "lambda_complexity": 0.15,
    # Normalizers, not weights: `min(1, x / budget)` saturates at the budget,
    # and a saturated term is a constant, not a gradient. Budgets of 200 / 30
    # are exceeded by a filter's first grown iteration (cost 515, complexity
    # 41), after which the run pays a flat penalty it can never reduce, so
    # neither term steers anything while still depressing every score. These
    # are sized above the largest filter observed (cost 571, complexity 41)
    # with room to grow into.
    "cost_budget": 800.0,
    "complexity_budget": 120.0,
}


def compute_score(
    metrics: FilterMetrics,
    weights: dict[str, float] | None = None,
) -> float:
    """The selection objective. Higher is better; every term is in [0, 1].

    score = critic_agreement            (the pair concordance)
          + λ_R   · rubric_quality
          + λ_Q   · rule_quality
          − λ_rev · review_rate
          − λ_c   · clamp(cost / cost_budget)
          − λ_cx  · clamp(complexity / complexity_budget)

    Four of the five weights default to 0 — see `DEFAULT_SCORE_WEIGHTS` for
    what each one measures and why it carries no weight. The terms are still
    computed and still summed, so a run can set one of them non-zero without
    a code change; `build_proposer_system`
    renders whichever terms are live so the proposer never sees a formula the
    selector is not applying.

    The redundancy penalty is NOT here. It is applied in the optimizer,
    because it reads per-criterion score correlations rather than anything on
    `FilterMetrics`.
    """

    w = {**DEFAULT_SCORE_WEIGHTS, **(weights or {})}
    cost_budget = max(1e-6, float(w.get("cost_budget", 200.0)))
    complexity_budget = max(1e-6, float(w.get("complexity_budget", 30.0)))

    cost_pen = min(1.0, metrics.estimated_filter_cost / cost_budget)
    complexity_pen = min(1.0, metrics.complexity_score / complexity_budget)

    return (
        metrics.critic_agreement
        + w["lambda_rubric_quality"] * metrics.rubric_quality
        + w["lambda_rule_quality"] * metrics.rule_quality
        - w["lambda_review"] * metrics.review_rate
        - w["lambda_cost"] * cost_pen
        - w["lambda_complexity"] * complexity_pen
    )
