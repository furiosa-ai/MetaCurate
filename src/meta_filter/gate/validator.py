"""Validator — evaluates a candidate against training pool, regression memory, and held-out set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from meta_filter.core.filter_loader import LoadedFilter
from meta_filter.core.schemas import (
    CandidateEvaluation,
    CounterExample,
    Decision,
    FilterDecision,
    FilterMetrics,
    Trajectory,
)
from meta_filter.core.solver import HeuristicRubricScorer, Solver
from meta_filter.critics import (
    BoundaryCritic,
    CriticBackend,
    PrecisionCritic,
    RecallCritic,
    StaticVerdictBackend,
)
from meta_filter.gate.complexity import measure_filter_complexity
from meta_filter.gate.memory import RegressionMemory
from meta_filter.gate.metrics import compute_metrics, scalar_loss


@dataclass
class ValidationBundle:
    """Everything the optimizer needs after validating one candidate."""

    evaluation: CandidateEvaluation
    train_decisions: list[FilterDecision]
    false_negatives: list[CounterExample]
    false_positives: list[CounterExample]
    boundary: list[CounterExample]


@dataclass
class Validator:
    """Runs the solver + critics on train/heldout + regression memory.

    Without a `critic_backend` nothing is adjudicated: the critics find no
    disagreements and the agreement metrics stay at zero. That is the honest
    default — this experiment has no labels of its own, so a candidate can
    only be scored against a second opinion someone explicitly supplied.
    """

    loss_weights: dict[str, float]
    boundary_confidence: float = 0.55
    boundary_margin: float = 1.0
    critic_backend: CriticBackend | None = None

    def evaluate(
        self,
        loaded_filter: LoadedFilter,
        *,
        candidate_id: str,
        is_baseline: bool,
        train_trajectories: Sequence[Trajectory],
        heldout_trajectories: Sequence[Trajectory],
        memory: RegressionMemory,
        solver: Solver | None = None,
        iteration: int,
        precomputed_train_decisions: list[FilterDecision] | None = None,
        train_verdicts: Mapping[str, Decision | None] | None = None,
        heldout_verdicts: Mapping[str, Decision | None] | None = None,
    ) -> ValidationBundle:
        solver = solver or Solver(rubric_scorer=HeuristicRubricScorer())

        if precomputed_train_decisions is None:
            train_decisions = solver.run(loaded_filter, train_trajectories)
        else:
            train_decisions = precomputed_train_decisions

        # Resolve the second opinion exactly once, then hand the critics a
        # frozen copy of it. Verdicts the optimizer already collected win over
        # a fresh backend call, so the critics and the metrics can never
        # disagree about what was said — and an LLM-backed judge is never
        # billed twice for the same trajectory.
        if train_verdicts is None:
            train_verdicts = _collect_verdicts(self.critic_backend, train_trajectories)
        backend = StaticVerdictBackend(
            {tid: v for tid, v in train_verdicts.items() if v is not None}
        )
        recall_critic = RecallCritic(backend=backend)
        precision_critic = PrecisionCritic(backend=backend)
        boundary_critic = BoundaryCritic(
            confidence_threshold=self.boundary_confidence,
            score_margin=self.boundary_margin,
        )
        fn = recall_critic.find_false_negatives(
            train_trajectories,
            train_decisions,
            iteration=iteration,
            filter_version=loaded_filter.version,
        )
        fp = precision_critic.find_false_positives(
            train_trajectories,
            train_decisions,
            iteration=iteration,
            filter_version=loaded_filter.version,
        )
        boundary = boundary_critic.find_fragile_cases(
            train_trajectories,
            train_decisions,
            iteration=iteration,
            filter_version=loaded_filter.version,
        )

        # Regression replay: re-run the stored counterexamples and count
        # violations symmetrically.
        must_accept_violations = _replay(
            solver=solver,
            loaded_filter=loaded_filter,
            counter_examples=memory.must_accept(),
            expected=Decision.ACCEPT,
        )
        must_reject_violations = _replay(
            solver=solver,
            loaded_filter=loaded_filter,
            counter_examples=memory.must_reject(),
            expected=Decision.REJECT,
        )

        breakdown, complexity_score = measure_filter_complexity(loaded_filter.directory)

        # Filter cost. If the rubric scorer is LLM-backed each solver call is
        # billed once per criterion; the mock heuristic scorer costs 0.
        est_cost = sum(d.llm_tokens for d in train_decisions) / 1000.0

        metrics = compute_metrics(
            train_trajectories,
            train_decisions,
            train_verdicts,
            complexity_score=complexity_score,
            complexity_breakdown=breakdown,
            regression_must_accept_violations=must_accept_violations,
            regression_must_reject_violations=must_reject_violations,
            estimated_filter_cost=est_cost,
        )
        metrics.loss = scalar_loss(metrics, self.loss_weights)

        # Optional held-out evaluation.
        heldout_metrics: FilterMetrics | None = None
        if heldout_trajectories:
            heldout_decisions = solver.run(loaded_filter, heldout_trajectories)
            # Held-out verdicts are never fetched implicitly: adjudicating the
            # held-out pool is a deliberate spend the caller opts into.
            heldout_metrics = compute_metrics(
                heldout_trajectories,
                heldout_decisions,
                heldout_verdicts or {},
                complexity_score=complexity_score,
                complexity_breakdown=breakdown,
                estimated_filter_cost=est_cost,
            )
            heldout_metrics.loss = scalar_loss(heldout_metrics, self.loss_weights)

        evaluation = CandidateEvaluation(
            candidate_id=candidate_id,
            is_baseline=is_baseline,
            metrics=metrics,
            false_negatives=[c.trajectory_id for c in fn],
            false_positives=[c.trajectory_id for c in fp],
            boundary_cases=[c.trajectory_id for c in boundary],
            heldout_metrics=heldout_metrics,
        )
        return ValidationBundle(
            evaluation=evaluation,
            train_decisions=train_decisions,
            false_negatives=fn,
            false_positives=fp,
            boundary=boundary,
        )


def _collect_verdicts(
    backend: CriticBackend | None,
    trajectories: Sequence[Trajectory],
) -> dict[str, Decision | None]:
    if backend is None:
        return {}
    return {t.trajectory_id: backend.critic_verdict(t) for t in trajectories}


def _replay(
    solver: Solver,
    loaded_filter: LoadedFilter,
    counter_examples,
    expected: Decision,
) -> int:
    """Count how many stored regression cases the candidate would violate.

    A must_accept example is violated iff the filter now REJECTs it.
    A must_reject example is violated iff the filter now ACCEPTs it.
    REVIEW is never counted as a regression — deferring is safe.

    We replay against the raw trajectory snapshot captured at counterexample
    creation time so the check is stable regardless of the current train pool.
    """

    if not counter_examples:
        return 0
    trajectories = [ce.raw_trajectory for ce in counter_examples if ce.raw_trajectory is not None]
    if not trajectories:
        return 0
    decisions = solver.run(loaded_filter, trajectories)
    violations = 0
    for d in decisions:
        if expected == Decision.ACCEPT and d.decision == Decision.REJECT:
            violations += 1
        elif expected == Decision.REJECT and d.decision == Decision.ACCEPT:
            violations += 1
    return violations
