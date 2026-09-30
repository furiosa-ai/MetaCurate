"""Typed schemas shared across the meta-filter pipeline.

These are intentionally light-weight pydantic models so that filters, critics
and the experience store all speak the same shape without dragging in a
database. Every schema is JSON-serializable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Decision(StrEnum):
    """Three-way filtering outcome."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


class Turn(BaseModel):
    """A single turn inside a trajectory."""

    model_config = ConfigDict(extra="allow")

    role: str
    action: str | None = None
    content: str = ""


class Trajectory(BaseModel):
    """One trajectory record.

    There is deliberately no truth field here. This experiment has no ground
    truth for "is this good SFT training data" — verdicts come from a
    `CriticBackend` and are passed around explicitly, so no code path can
    mistake a critic's opinion for a label that shipped with the data.
    """

    model_config = ConfigDict(extra="allow")

    trajectory_id: str
    task: str = ""
    turns: list[Turn] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FilterDecision(BaseModel):
    """Solver output for a single trajectory."""

    trajectory_id: str
    decision: Decision
    confidence: float = 0.0
    # The policy's continuous quality score for this row (higher = better
    # SFT data). None when the policy did not emit one.
    rank_score: float | None = None
    # The scorer could not produce a score for this row at all — an
    # unreadable answer after its retries, or a refusal. Distinct from a
    # missing `rank_score`, which a filter may also produce by choice: this
    # says nothing about the filter, so the row leaves the ranking exam for
    # every candidate rather than costing this one its pairs.
    scorer_failed: bool = False
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    triggered_rules: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    # Cost accounting for this single decision.
    llm_calls: int = 0
    llm_tokens: int = 0


class CounterExample(BaseModel):
    """A persistent regression case discovered by a critic.

    `raw_trajectory` snapshots the trajectory as it existed at capture time so
    the regression replay is fully self-contained (memory can be reloaded even
    if the original training pool is gone).
    """

    trajectory_id: str
    source_iteration: int
    filter_version: str
    # What the critic backend said. Not ground truth — the critic is a second
    # opinion that can itself be wrong.
    critic_decision: Decision
    predicted_decision: Decision
    failure_type: Literal[
        "false_negative",
        "false_positive",
        "boundary",
    ]
    failure_subtype: str = ""
    severity: int = 1
    confidence: float = 0.0
    evidence_turns: list[int] = Field(default_factory=list)
    description: str = ""
    why_filter_failed: str = ""
    filter_change_that_fixed_it: str | None = None
    raw_trajectory: Trajectory | None = None
    solver_decision: FilterDecision | None = None
    # Per-criterion difference between the two rows of a misordered pair
    # (judge's pick minus the row the filter put above it), with the
    # criterion that most opposes the judge's order flagged. One line,
    # pre-rendered by `optimizer._criterion_breakdown`, so the proposer can
    # see WHICH axis inverted the pair rather than two score dicts it has to
    # subtract itself.
    criterion_breakdown: str = ""
    # Where this pair came from and what has already been done with it — the
    # iteration that bought the verdict, whether it came from the disputed or
    # the consensus-audit stratum, and the earlier prompts that showed it.
    # Pre-rendered by `optimizer._pair_history_note`, one line. Empty when
    # nothing is known.
    pair_history: str = ""


class FilterMetrics(BaseModel):
    """How a candidate filter compares against the critic backend.

    Every field below is a *filter vs critic* comparison, not a measure of
    correctness. The critic is a second opinion, so a disagreement means the
    two differed — not that the filter was wrong. Field names say which side
    is which so no reader has to remember that.
    """

    total: int
    critic_accept_total: int
    critic_reject_total: int

    accepted: int
    rejected: int
    reviewed: int

    agree_accept: int
    agree_reject: int
    filter_accept_critic_reject: int
    filter_reject_critic_accept: int

    critic_reject_recall: float
    critic_accept_retention: float
    disputed_accept_rate: float
    disputed_reject_rate: float
    review_rate: float
    # The agreement term of the selection score. The optimizer overwrites it
    # with the candidate's pair concordance (zero for a degenerate scorer);
    # `compute_metrics` fills it with the per-row agreement rate.
    critic_agreement: float = 0.0

    llm_calls: int = 0
    llm_tokens: int = 0
    estimated_filter_cost: float = 0.0

    # Regression replay.
    regression_must_accept_violations: int = 0
    regression_must_reject_violations: int = 0

    # Filter complexity snapshot.
    complexity: dict[str, int] = Field(default_factory=dict)
    complexity_score: float = 0.0

    # Weighted scalar loss (lower is better), reported beside the score;
    # `validator.loss_weights` sets its weights.
    loss: float = 0.0

    # The rubric and rule critics' quality readings. 0.5 = neutral (no critic
    # run yet); RubricCritic and RuleCritic overwrite them.
    rubric_quality: float = 0.5
    rule_quality: float = 0.5

    # The selection objective (higher is better), from `compute_score`.
    score: float = 0.0


class CandidateEvaluation(BaseModel):
    """Everything we learned about a single candidate in a given iteration."""

    candidate_id: str
    is_baseline: bool
    metrics: FilterMetrics
    false_negatives: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)
    boundary_cases: list[str] = Field(default_factory=list)
    heldout_metrics: FilterMetrics | None = None
    invalid_reason: str | None = None


class IterationRecord(BaseModel):
    """Per-iteration summary persisted to disk."""

    iteration: int
    current_filter_version: str
    selected_candidate: str
    selected_reason: str
    evaluations: list[CandidateEvaluation]
    new_false_negatives: list[str] = Field(default_factory=list)
    new_false_positives: list[str] = Field(default_factory=list)
    new_boundary: list[str] = Field(default_factory=list)
    diff: str = ""

    # Rows the scorer could not read an answer for after its retries. They
    # carry no score, so they are excluded from pair selection and from
    # concordance; recorded here because an iteration graded on a smaller
    # pool than it was given should say so rather than be inferred from a
    # pair count.
    scorer_failed_rows: int = 0
    scorer_failed_row_ids: list[str] = Field(default_factory=list)

    # Candidates the solver pass dropped after the gate had already passed
    # them: candidate id -> why. Normally empty, because the precheck fires
    # `invalid` on an unscoreable `RUBRIC_CRITERIA` and the gate holds it
    # back with a reason; a non-empty map records a candidate that would
    # otherwise be lost without trace.
    discarded_at_scoring: dict[str, str] = Field(default_factory=dict)

    # Spend for this iteration, per role, in USD. Measured from each route's
    # own report where it makes one and from list price otherwise
    # (`llm/cost_meter.py`). `unpriced_calls` is how many calls had neither,
    # i.e. how much of the total is missing rather than zero.
    cost_solver: float = 0.0
    cost_judge: float = 0.0
    cost_proposer: float = 0.0
    cost_critic: float = 0.0
    unpriced_calls: int = 0

    # What each candidate scored against the already-judged pairs before the
    # loop paid to score it, keyed by candidate id — once as the proposer was
    # shown it, and once after its revision leg. Both, because "shown X,
    # produced Y" is the only thing that says whether the leg bought
    # anything. Empty for an iteration with no adjudicated pair to check
    # against.
    precheck: dict[str, dict] = Field(default_factory=dict)
    precheck_after_revision: dict[str, dict] = Field(default_factory=dict)
    # 1 when the proposer answered in one call, 2 once it took its revision
    # or its retry. Never more: the bound is the design, not a budget, and a
    # 3 here means something re-entered the loop.
    proposer_legs: int = 1

    # The redundancy term for the SELECTED candidate, and the score it is
    # reported with. The penalty is measured but kept out of the score
    # selection sorts on, so the two differ: `evaluations[...].metrics.score`
    # is what won the iteration and `score_with_redundancy` is what the filter
    # this iteration would deploy is charged.
    redundancy_penalty: float = 0.0
    score_with_redundancy: float | None = None


class RunConfig(BaseModel):
    """Snapshot of the config used to launch a run (`config.yaml`).

    Everything that decides what a run optimizes is recorded, so a finished
    run can say what it was configured to do. `meta_filter.score_tool` and
    `meta_filter.rule_tool` read the train pool and the scorer settings back
    from this file.
    """

    seed: int
    iterations: int
    candidates_per_iteration: int
    llm_provider: str
    llm_model: str | None
    # Weights of the reported scalar loss, not of the selection objective;
    # `score_weights` is the one that decides runs.
    loss_weights: dict[str, float]
    # The stripped copy of the train pool inside the run directory (the file
    # a tool-using proposer and its tools open). The source pool's sha256 is
    # `train_data_source_sha256`; its path is recorded outside the workspace,
    # in `runs/_metadata_free/<run_id>.json`.
    train_data: str
    validation_data: str
    initial_filter: str
    score_weights: dict[str, float] = {}
    pair_budget: int | None = None
    max_pairs: int | None = None
    truth_includes_consensus: bool | None = None
    consensus_audit_share: float | None = None
    redundancy_penalty: float | None = None
    redundancy_threshold: float | None = None
    # How many consensus-audit pairs the next proposer was shown alongside
    # the misordered-pair blocks (`--pair-evidence`).
    pair_evidence: int | None = None
    # The degeneracy guard's floor. `null` means it was derived from the pool
    # size (`ranking.default_min_distinct_levels`), so the effective number is
    # written into each iteration's `ranking.json` as well.
    min_distinct_levels: int | None = None
    # Which pair-verdict cache the run drew on.
    pair_judge_cache_dir: str | None = None
    # The hidden rows the acceptance gate graded on, how many pairs it drew
    # and from which seed, and the gate's z and discordant-pair floor.
    selection_data: str | None = None
    # An int, or "all" for every unordered pair. None when the pairs came
    # from a prebuilt file (`selection_set`).
    selection_pairs: int | str | None = None
    selection_seed: int | None = None
    # The prebuilt set (`scripts.build_selection_set`), so runs pointed at the
    # same file are graded on the same pairs. None means the pairs were
    # planned and judged by this run at start.
    selection_set: str | None = None
    gate_z: float | None = None
    gate_min_discordant: int | None = None
    # Early stop: how many consecutive kept-baseline iterations end the run,
    # and the floor before that can happen. None = never.
    patience: int | None = None
    min_iterations: int | None = None
    # How many of the proposed candidates were fully scored each iteration,
    # chosen by their precheck rank on the already-judged pairs. None means
    # every proposed candidate was scored.
    candidates_scored: int | None = None
    # The solver's rubric scorer, exactly as `meta_filter.score_tool` needs it
    # to score a criterion under the run's own model, request body and cache.
    # `solver_backend` is "llm" or "heuristic"; `solver_provider` is the
    # `build_client` keyword ("openai", "mock"), not a class name, because the
    # tool rebuilds the client from it.
    solver_backend: str | None = None
    solver_provider: str | None = None
    solver_model: str | None = None
    solver_base_url: str | None = None
    solver_extra_body: dict[str, object] | None = None
    solver_cache_dir: str | None = None
    # Dollar cap per proposer leg on the criterion-scoring tool
    # (`--proposer-score-budget`); its spend is folded into `cost_proposer`.
    # None when the tool is not offered.
    proposer_score_budget: float | None = None
    # The run hard cap as numbers (`--complexity-ceiling`):
    # `{component: largest count allowed}`. None when not set.
    complexity_ceiling: dict[str, int] | None = None
    # Which of rubric / rules / policy the run froze (`--freeze rules,policy`):
    # a candidate that changes a frozen file is invalid at precheck. None
    # means nothing was frozen.
    freeze: list[str] | None = None
    # The sha256 of the pool `train_data` was stripped from.
    train_data_source_sha256: str | None = None
