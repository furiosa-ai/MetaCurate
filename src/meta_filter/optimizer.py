"""Optimizer — the outer loop that drives the whole system.

F = initial_filter
for k in K:
    candidates = Proposer.propose(F, misordered pairs, history)
    candidates = precheck, one revision leg, gate      # train pairs only
    for F_i in [F] + candidates:                       # baseline first
        scores = Solver.run(F_i)
        concordance = against the pair judge's train verdicts
    F_next = paired McNemar gate on the hidden selection pairs
    F = F_next (re-scored and confirmed on acceptance)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from meta_filter.complexity import measure_filter_complexity
from meta_filter.concurrency import AdaptiveGovernor, MemorySampler
from meta_filter.concurrency import log_plan as log_concurrency_plan
from meta_filter.constraints import (
    ConstraintLimits,
    complexity_score,
    declared_unread_criteria,
    rubric_defects,
)
from meta_filter.criterion_lift import (
    render_criterion_signal,
    single_criterion_concordance,
    zero_lift_items,
    zero_lift_note,
)
from meta_filter.criterion_stats import charge_redundancy, redundancy_penalty
from meta_filter.critics import (
    RubricCritic,
    RubricCriticReport,
    RuleCritic,
    RuleCriticReport,
    TrajectoryCritic,
)
from meta_filter.dead_axis import check as dead_axis_check
from meta_filter.dead_axis import log as dead_axis_log
from meta_filter.experience_store import ExperienceStore
from meta_filter.filter_loader import FilterValidationError, LoadedFilter, load_filter
from meta_filter.llm.base import UsageLimitError
from meta_filter.llm.cost_meter import LEDGER as COST_LEDGER
from meta_filter.llm.cost_meter import metered, unwrap
from meta_filter.llm_scorer import LLMRubricScorer, cache_key_trajectory_id, sanitize_trajectory_id
from meta_filter.memory import RegressionMemory
from meta_filter.metadata_free import (
    METADATA_FREE_TRAIN_FILE,
    PSEUDONYM_SALT,
    file_sha256,
    filter_outcome_reads,
    pseudonymize_selection_set,
    selection_set_view,
)
from meta_filter.metrics import DEFAULT_SCORE_WEIGHTS, compute_score
from meta_filter.paired_test import (
    critical_b_table,
    one_sided_p_from_z,
    pair_correctness,
    paired_stats,
    render_critical_b_table,
    stats_from_counts,
    z_threshold_for,
)
from meta_filter.probe import ProbeSet
from meta_filter.probe import append_report as probe_append_report
from meta_filter.probe import latest_decisions as probe_latest_decisions
from meta_filter.probe import load_reports as probe_load_reports
from meta_filter.probe import render_for_prompt as probe_render_for_prompt
from meta_filter.proposal_shape import report as shape_report
from meta_filter.proposer import (
    PROMPT_MISORDER_PAIRS,
    Proposer,
    ProposerResult,
    build_proposer_system,
)
from meta_filter.pseudonym import Pseudonymizer
from meta_filter.ranking import DEGENERACY_PENALTY, default_min_distinct_levels
from meta_filter.reports import render_iteration_report, unified_diff
from meta_filter.resume import rebuild as resume_state
from meta_filter.schemas import (
    CandidateEvaluation,
    CounterExample,
    Decision,
    FilterDecision,
    IterationRecord,
    RunConfig,
    Trajectory,
)
from meta_filter.score_tool import (
    ScoreBudgetLedger,
    ScoreToolAccess,
    access_for,
    merge_staged_cells,
    staging_dir_for,
)
from meta_filter.selection_set import (
    PairCount,
    SelectionSet,
    assert_disjoint,
    baseline_scores_path,
    judge_selection_pairs,
    plan_selection_pairs,
    scores_from_decisions,
    selection_row_ids_in_text,
    selection_set_path,
)
from meta_filter.selector import STRONG_EXTRA_Z, SelectionResult, Selector
from meta_filter.solver import HeuristicRubricScorer, Solver
from meta_filter.trajectory import load_trajectories
from meta_filter.validator import Validator
from meta_filter.view_policy import pair_view_signature

logger = logging.getLogger(__name__)

# How many proposer failures in a row before the run is declared broken.
# One is a timeout or a 5xx; three in a row has never been anything but a
# misconfiguration — a bad auth mode, a CLI that refuses to run, a missing
# key — and every iteration after the first is wasted time.
MAX_CONSECUTIVE_PROPOSER_FAILURES = 3


class MetadataFreeReadError(RuntimeError):
    """A metadata-free proposer read a path outside its workspace.

    Codex's sandbox cannot deny reads, so this is enforced after the fact:
    the proposal is invalidated (`Optimizer._invalidate_proposal`) rather
    than scored, and the iteration counts as a proposer failure.
    """

    def __init__(self, iteration: int, leg: str, reads: list[str]) -> None:
        self.iteration, self.leg, self.reads = iteration, leg, tuple(reads)
        super().__init__(
            f"iteration {iteration} ({leg} leg): the proposer read {len(reads)} path(s) "
            "outside its workspace in a metadata-free run"
        )


# Written at the run root when the loop stops on its own terms: `run_id`,
# `exit_status` and `finished_at`, plus why the loop stopped.
RUN_COMPLETE_FILE = "_RUN_COMPLETE.json"


@dataclass
class OptimizerConfig:
    iterations: int = 3
    candidates_per_iteration: int = 3
    seed: int = 17
    # Weights of the reported scalar loss (`Validator`); the selection
    # objective is `score_weights`.
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "lambda_fn": 5.0,
            "lambda_fp": 4.0,
            "lambda_review": 0.5,
            "lambda_cost": 0.1,
            "lambda_complexity": 0.05,
        }
    )
    boundary_confidence: float = 0.55
    boundary_margin: float = 1.0
    judge_parallel: int = 8
    # Disagreement pairs adjudicated per iteration, sampled stratified across
    # score deciles so supervision covers the whole distribution rather than
    # one cut boundary.
    pair_budget: int = 48
    # How many of the pool's pairs are classified at all. Classification is
    # sign arithmetic over scores already paid for; the cap bounds the
    # disagreement pool the judge budget samples from.
    max_pairs: int = 4000
    # Whether pairs the candidates already agree on count as truth. They are
    # free (no judge call) but identical for every candidate, so they carry no
    # selection signal while making up most of the truth set, and they reward
    # a candidate for conforming to its siblings.
    truth_includes_consensus: bool = True
    # Share of the judge budget spent auditing pairs the candidates already
    # agree on. Above zero the judge sees consensus pairs every iteration,
    # which is what makes "all candidates agree" falsifiable: a committee that
    # ranks the pool backwards in unison produces no disagreement to judge.
    consensus_audit_share: float = 0.0
    # A scorer that emits fewer distinct values than this cannot express a
    # ranking; its concordance is zeroed (a degeneracy guard, not a style
    # rule). None scales the floor with the pool, `max(10, 0.4 * n_train)`
    # (`ranking.default_min_distinct_levels`).
    min_distinct_levels: int | None = None
    # How many consensus-audit pairs the winner mis-ordered to show the next
    # proposer, with the judge's one-line rationale. 0 disables. Drawn only
    # from the audit stratum and deduplicated against the misordered-pair
    # blocks, so it carries pairs every candidate ordered the same wrong way,
    # which a disagreement-driven sample cannot reach.
    pair_evidence: int = 0
    # Graded penalty for inter-criterion redundancy: the max tie-aware
    # |Spearman| between two live criteria above `redundancy_threshold` ramps
    # linearly to `redundancy_penalty` at 1.0. It is computed, logged and
    # shown to the proposer, and charged only to the reported
    # `score_with_redundancy` of the selected filter; selection never sorts on
    # it. |rho| is only measurable after the pool is scored, so as a selection
    # term it would charge every new criterion for its first contact with the
    # incumbents before it could differentiate itself.
    redundancy_penalty: float = 0.5
    redundancy_threshold: float = 0.7
    # Per-iteration train subsampling schedule. Position k gives the fraction
    # of the train pool used at iteration k; missing tail positions fall back
    # to 1.0. A single float means "use this rate every iteration".
    train_sample_schedule: list[float] | float | None = None
    # Number of candidates whose (prefetch + solver.run) runs concurrently.
    # 0 means one worker per candidate, so an iteration costs the slowest
    # candidate instead of the sum; 1 is the serial flow. Each candidate
    # spawns its own scorer pool, so effective concurrency is
    # `candidates_parallel * solver_parallel`.
    candidates_parallel: int = 0
    # Selection-objective overrides; anything absent falls back to
    # `metrics.DEFAULT_SCORE_WEIGHTS`, which also holds the `cost_budget` and
    # `complexity_budget` normalizers.
    score_weights: dict[str, float] = field(default_factory=dict)
    # Size of the curated evidence set the rubric critic reads.
    curated_evidence_budget: int = 6
    # Optional LLM recommendation pass in RuleCritic. Deterministic per-rule
    # stats always run; this only adds keep/move/split/drop recommendations
    # and does not enter `rule_quality`.
    rule_critic_llm_recommendations: bool = False
    # Let an LLM score the rule set's design and fold that into
    # `rule_quality`. It moves the selection score only through a non-zero
    # `lambda_rule_quality`.
    rule_design_review: bool = False
    # Continue an existing run directory instead of starting at iteration 0.
    # The tool-using proposer is not seed-deterministic, so a restart does not
    # reproduce the iterations already paid for.
    resume: bool = False
    # Pseudonymize trajectory ids on proposer-visible surfaces and route
    # held-out metrics outside the run directory. Set for the tool-using
    # proposer, which reads the run directory itself.
    agentic_critics: bool = False
    # Where PairJudge caches its verdicts.
    pair_judge_cache_dir: str | Path = "data/pair_judge_cache"
    # The hidden selection rows: disjoint from train and probe, scored for
    # every candidate each iteration, and the rows the paired acceptance gate
    # is graded on. Required by `Optimizer`.
    selection_data: Path | None = None
    # Without `selection_set`: how many pairs to plan over the selection rows
    # at run start, or "all" for every unordered pair with no stratification.
    selection_pairs: PairCount = 500
    selection_seed: int = 4242
    # A set built once outside any run (`scripts.build_selection_set`) and
    # copied into `runs/_selection/<run_id>.json` instead of planned and
    # judged at start, so every run pointed at it is graded on the same
    # pairs. Its rows must be a subset of `selection_data`.
    selection_set: Path | None = None
    # The paired acceptance gate. `gate_z` None derives the per-comparison
    # threshold from the number of scored candidates
    # (`paired_test.z_threshold_for`, ~2.43 for three); a number pins it.
    gate_z: float | None = None
    gate_min_discordant: int = 5
    # Stop after this many consecutive iterations that scored at least one
    # loadable candidate and kept the baseline. None never stops early.
    # `min_iterations` is the floor before patience can end a run.
    patience: int | None = None
    min_iterations: int = 4
    # The proposer writes `candidates_per_iteration` directories, the
    # precheck ranks them on the already-judged pairs, and only the top
    # `candidates_scored` go through the full train + selection solver pass.
    # The rest stay on disk (the next proposer may read them) and are
    # recorded in the iteration's `precheck.json` as `gated_out`. None scores
    # every proposed candidate. Constraint-invalid, duplicate, no-op and
    # second deletion candidates are gated regardless of this number.
    candidates_scored: int | None = None
    # Dollar cap, per proposer leg, on the criterion-scoring tool the
    # tool-using proposer may call (`meta_filter.score_tool`): it scores a
    # criterion the proposer is drafting on a few train rows under the run's
    # own scorer before the proposer commits to it. Spend is folded into
    # `cost_proposer`. Only offered to a tool-using proposer on the LLM
    # scorer; the prompt does not mention the tool otherwise.
    proposer_score_budget: float = 1.50
    # The run hard cap as numbers, `{component: max}`. Empty means 1.5x the
    # initial filter's size (`constraints.ConstraintLimits`).
    complexity_ceiling: dict[str, int] = field(default_factory=dict)
    # Files a candidate may not change (`--freeze rules,policy`). A changed
    # frozen file is `invalid` at precheck, whatever else the candidate does.
    freeze: tuple[str, ...] = ()
    # Instead of refusing a prebuilt selection set whose provenance does not
    # show it was judged on stripped rows, re-judge its pairs on the stripped
    # rows at run start (paid).
    rejudge_selection: bool = False


class Optimizer:
    def __init__(
        self,
        *,
        initial_filter_dir: str | Path,
        train_data: str | Path,
        validation_data: str | Path | None,
        runs_dir: str | Path,
        memory_dir: str | Path,
        proposer: Proposer,
        selector: Selector,
        solver: Solver,
        validator: Validator,
        config: OptimizerConfig,
        run_id: str | None = None,
        governor: AdaptiveGovernor | None = None,
        requested_solver_parallel: int | None = None,
        probe_data: str | Path | None = None,
    ) -> None:
        self.initial_filter_dir = Path(initial_filter_dir)
        self.train_path = Path(train_data)
        self.heldout_path = Path(validation_data) if validation_data else None
        if config.selection_data is None:
            raise ValueError(
                "selection_data is required: the paired gate on the hidden "
                "selection rows is what decides acceptance"
            )
        seed_reads = filter_outcome_reads(Path(initial_filter_dir))
        if seed_reads:
            # The precheck refuses these with no initial-filter exemption, so
            # every candidate copied from this seed would be invalid.
            raise ValueError(
                f"refusing initial filter {initial_filter_dir}: it "
                + "; ".join(f"{name} names {', '.join(v)}" for name, v in seed_reads.items())
            )
        # The strip happens here and nowhere else: `load_trajectories` keeps
        # only the allow-listed metadata keys before a `Trajectory` exists, so
        # no component downstream of these lists can see anything else.
        self.train = load_trajectories(train_data)
        self.heldout = load_trajectories(validation_data) if validation_data else []

        # The fixed probe set (see meta_filter.probe): constant data every
        # iteration's winner is run on so drift is observable. Loaded here,
        # evaluated at the advance step, never part of selection.
        self.probe = ProbeSet.from_path(probe_data) if probe_data else None
        self._probe_path = probe_data

        # The hidden selection rows. Loaded like the probe, scored by the same
        # solver, and never handed to the proposer: their ids are what
        # `_assert_no_selection_exposure` greps every saved prompt for.
        self.selection_rows: list[Trajectory] = load_trajectories(config.selection_data)
        # Refused here as well as at the CLI: the optimizer is also built
        # directly by tests and scripts, and an overlap makes the gate grade
        # on rows the proposer was shown the verdicts for.
        assert_disjoint(
            (t.trajectory_id for t in self.selection_rows),
            train_ids=(t.trajectory_id for t in self.train),
            probe_ids=(
                (t.trajectory_id for t in self.probe.trajectories) if self.probe is not None else ()
            ),
        )
        # Judged once at run start (or reloaded on resume); None until then.
        self.selection_set: SelectionSet | None = None
        # The baseline's scores on the selection rows. Carried across
        # iterations rather than re-scored: an unchanged filter keeps its
        # number, and a promoted candidate's number is the FRESH draw taken
        # at promotion, not the draw that won (see `_promotion_rescore`).
        self._baseline_selection_scores: dict[str, float] | None = None

        self.runs_dir = Path(runs_dir)
        self.memory_dir = Path(memory_dir)
        self.proposer = proposer
        self.selector = selector
        self.solver = solver
        if getattr(solver, "lazy_scoring", False):
            # Every row's score feeds the ordering, so no row is
            # decision-invariant — the lazy probe's premise does not hold.
            solver.lazy_scoring = False
            logger.info("lazy scoring disabled: every row's score feeds the ordering")
        self.validator = validator
        self.config = config
        # The evaluation set: every verdict this run buys, accumulated. It
        # only grows, so an unchanged filter keeps its score and today's
        # disagreements are in the exam — see `_evaluation_pairs`.
        self._accumulated_pairs: list[tuple[str, str]] = []
        self._accumulated_keys: set[frozenset[str]] = set()
        # Where each pair came from and when, for `ranking_pairs.jsonl` and
        # the pair-history lines of the proposer prompt.
        self._pair_provenance: dict[frozenset[str], dict] = {}
        # Which prompts have already carried each misordered pair. Repeats
        # are not suppressed — a pair the champion still gets wrong is still
        # the best evidence there is — but a pair the lineage has been shown
        # twice and still inverts says something a first showing does not.
        self._shown_misorder_pairs: dict[frozenset[str], list[int]] = {}
        self._eval_history: list[int] = []
        self.run_id = run_id or _new_run_id(self.runs_dir)
        # The governor re-plans every iteration, so it needs what was
        # originally *asked for* — scaling from the already-scaled value
        # would ratchet the run down and never let it back up.
        self.governor = governor
        self.requested_solver_parallel = (
            requested_solver_parallel
            if requested_solver_parallel is not None
            else self._solver_workers()
        )
        self.requested_judge_parallel = config.judge_parallel

        # The proposer must be told the objective the selector will apply,
        # including any weight the config overrode.
        self.proposer.system_prompt = build_proposer_system(
            {**DEFAULT_SCORE_WEIGHTS, **config.score_weights},
            # Applied outside `compute_score`, so it has to be handed over
            # explicitly or the proposer would not be told how it is reported.
            redundancy_penalty=config.redundancy_penalty,
            redundancy_threshold=config.redundancy_threshold,
            # The floor scales with the pool. This is the full-pool figure; a
            # run with a train subsample schedule sees a smaller floor in the
            # subsampled iterations, and the per-iteration number it was
            # graded against is rendered in the ranking telemetry block and
            # written into `ranking.json`.
            min_distinct_levels=(
                config.min_distinct_levels
                if config.min_distinct_levels is not None
                else default_min_distinct_levels(len(self.train))
            ),
        )
        # The edit-discipline block prints the growth caps as numbers against
        # the run's initial filter, and says how many candidates the precheck
        # lets through to a full scoring pass.
        self.proposer.initial_complexity = measure_filter_complexity(self.initial_filter_dir)[0]
        self.proposer.candidates_scored = config.candidates_scored
        # The frozen files, named in the edit-discipline block.
        self.proposer.frozen_files = tuple(config.freeze)
        # The last winner's full-pass `criterion_stats` — the numbers the
        # criterion-health block showed the proposer. A candidate's NEW dead
        # axes and redundant pairs are read against exactly these, falling
        # back to the parent's precheck readout when there is none yet
        # (iteration 0, a resume).
        self._baseline_criterion_stats: dict | None = None
        # The ranking mission reads many artifacts and prototypes in its
        # workspace; a 30-minute call is too short for it.
        client = getattr(self.proposer, "client", None)
        if getattr(client, "timeout", None) and client.timeout < 3600.0:
            from meta_filter.llm.cost_meter import replace_client

            # The client may be wrapped by the cost meter, which delegates
            # `timeout` but does not own it; replace_client sees through.
            self.proposer.client = replace_client(client, timeout=3600.0)
            logger.info("proposer timeout raised to 3600s")

        self.pseudonymizer: Pseudonymizer | None = None
        if config.agentic_critics:
            self.pseudonymizer = Pseudonymizer.for_pool(
                [t.trajectory_id for t in (*self.train, *self.heldout)]
            )
            self.pseudonymizer.save(self.runs_dir / "_idmap" / f"{self.run_id}.json")
            self.proposer.pseudonymizer = self.pseudonymizer

        self.store = ExperienceStore(
            run_dir=self.runs_dir / self.run_id,
            pseudonymizer=self.pseudonymizer,
            heldout_sink=(
                self.runs_dir / "_heldout" / f"{self.run_id}.jsonl"
                if config.agentic_critics
                else None
            ),
        )
        self.memory = RegressionMemory(base_dir=self.memory_dir)

        # The criterion-scoring tool (`meta_filter.score_tool`). Advertised to
        # the proposer only where it can work: a tool-using proposer that has
        # a shell to run it from, and an LLM scorer whose model and cache the
        # tool can rebuild from `config.yaml`. The static proposer answers
        # from its prompt and could not call it; the heuristic scorer reads
        # no rubric text and could tell it nothing.
        scorer = getattr(self.solver, "rubric_scorer", None)
        self.score_tool_enabled = bool(
            getattr(self.proposer, "agentic", False) and isinstance(scorer, LLMRubricScorer)
        )
        self.score_tool: ScoreToolAccess | None = (
            access_for(self.store.run_dir, budget_usd=config.proposer_score_budget)
            if self.score_tool_enabled
            else None
        )
        self.proposer.score_tool = self.score_tool
        # Per-criterion cells of the last selection-row pass per filter
        # directory, so the promotion re-score can compare the winning draw
        # with the fresh one criterion by criterion (`criterion_stability`).
        self._selection_cells: dict[str, dict[str, dict[str, float]]] = {}

        # The critics reuse the proposer's LLM client under their own cost
        # role. The rubric critic is static: one call per iteration over the
        # curated evidence pack, no tools. RuleCritic's deterministic per-rule
        # stats always run; its LLM passes are opt-in.
        critic_client = metered(self.proposer.client, "critic")
        rule_critic_client = (
            critic_client
            if (self.config.rule_critic_llm_recommendations or self.config.rule_design_review)
            else None
        )
        if self.config.agentic_critics:
            logger.info(
                "rubric critic runs static despite --agentic-critics: it is "
                "diagnostic now (lambda_rubric_quality=0), so it gets one "
                "call over the curated cases and no tools"
            )
        self.rubric_critic = RubricCritic(client=critic_client)
        self.rule_critic = RuleCritic(
            client=rule_critic_client,
            recommend=self.config.rule_critic_llm_recommendations,
            score_design=self.config.rule_design_review,
        )

        random.seed(self.config.seed)

    # ---------------------------------------------------------------- main

    # ------------------------------------------------------------- governor

    def _accumulate_eval_pairs(
        self,
        resolved: list[tuple[str, str]],
        *,
        audit_keys: set[frozenset[str]] | None = None,
        iteration: int | None = None,
    ) -> int:
        """Add this iteration's purchased verdicts to the evaluation set.

        Scoring only on the pairs just bought, and then discarding them, would
        re-measure an unchanged filter against a fresh sample of a moving
        population, so a retained baseline's score would wander by as much as
        a whole run's gain.

        Accumulating rather than freezing is what makes the two obvious
        objections cancel: today's disagreements are in the exam because they
        were just bought and added, and an unchanged filter keeps its score
        because nothing is ever removed. Acquisition is untouched — every
        verdict still comes from `disagreement_pairs + audit_pairs`.
        """

        added = 0
        for pair in resolved:
            key = frozenset(pair)
            # Recorded before the dedup check and never overwritten: a pair
            # re-drawn in a later iteration was still BOUGHT in the first
            # one, and that is the iteration whose evidence it belongs to.
            self._pair_provenance.setdefault(
                key,
                {
                    "source": "audit" if key in (audit_keys or set()) else "disputed",
                    "judged_iteration": iteration,
                },
            )
            if key in self._accumulated_keys:
                continue
            self._accumulated_keys.add(key)
            self._accumulated_pairs.append(pair)
            added += 1
        return added

    def _record_shown_misorders(self, over_ranked: list[CounterExample], *, iteration: int) -> None:
        """Note which misordered pairs the iteration-`iteration` prompt carries.

        Called once the winner is known, because the winner's records are the
        ones the next proposer call renders. Only the first
        `PROMPT_MISORDER_PAIRS` are recorded: the rest are written to
        `critic_outputs/` for a proposer that goes looking, and claiming they
        were "already shown" would be false.
        """

        for example in over_ranked[:PROMPT_MISORDER_PAIRS]:
            key = _misorder_pair_key(example)
            if "" in key:
                continue
            self._shown_misorder_pairs.setdefault(key, []).append(iteration)

    def _evaluation_pairs(self, traj_by_id: dict) -> list[tuple[str, str]]:
        """The accumulated set restricted to pairs that lie in this pool.

        Restricted to the *pool*, not to the candidates' common scored ids:
        every candidate is offered the identical pair list and pays for its
        own gaps (`ranking.concordance_detail(unscored="loss")`). Intersecting
        with the common ids instead would hand candidates different exams
        inside one iteration, which is the defect this change exists to
        remove.
        """

        return [p for p in self._accumulated_pairs if p[0] in traj_by_id and p[1] in traj_by_id]

    @staticmethod
    def _ensure_baseline_dir(parent_dir: Path, parent_filter, iteration: int) -> None:
        """Re-materialize the baseline candidate copy if the proposer removed it.

        The tool-using proposer shares the filesystem with the candidates and
        can delete `iter*_baseline` (an agent running `rm -rf` on it). The
        baseline is by definition a copy of the filter under edit, so it can
        be rebuilt from `parent_filter.directory`; a vanished *proposed*
        candidate has no other source and is left to fail loudly.
        """

        parent_dir = Path(parent_dir)
        if parent_dir.is_dir():
            return
        logger.error(
            "iter %d: baseline dir %s vanished before the precheck (proposer "
            "escape suspected) — re-materializing from %s",
            iteration,
            parent_dir,
            parent_filter.directory,
        )
        shutil.copytree(parent_filter.directory, parent_dir)

    def _precheck_and_revise(
        self,
        proposal: ProposerResult,
        *,
        iteration: int,
        parent_dir: Path,
        parent_filter,
        active_train: list,
        candidates_root: Path,
    ) -> PrecheckOutcome:
        """Precheck the candidates, then spend the one revision leg.

        Returns the proposal to actually score, plus the precheck readouts
        before and after the revision. Both are recorded because the pair
        that matters is the pair: "the proposer was shown X and produced Y"
        is the only thing that says whether the leg bought anything, and a
        run that keeps only the final numbers cannot answer it later.

        Three guards decide what happens, and each one has a reason:

        * No already-judged pair lies in this pool. That is iteration 0:
          nothing has been adjudicated, so there is no measurement to hand
          over and a revision leg would be a second call with no new
          information in it. Skipped — but the growth and deletion caps are
          file-level facts that need no pairs, so they are still read and
          still gate, just without a rank.
        * Leg 1 produced no loadable candidate. Then there is nothing to
          precheck and the last leg is worth more as a retry than as a
          revision — `Proposer.revise` reads that off `candidate_dirs` and
          switches prompts itself, so the decision lives in one place.

        The "after" precheck is not extra spend: the revised candidates go
        straight into the real solver pass, which needs the same cells for
        the same rows, and the scorer cache is shared.

        Candidate gating comes last: whichever readout is final — the "after"
        one when a revision happened, the "before" one otherwise — decides
        which candidates the solver pays for. `gate_candidates` has the rules;
        the returned proposal carries only the survivors, and `gated_out`
        names the rest with the reason each one was held back.
        """

        from meta_filter.precheck import render_precheck_block, run_precheck

        traj_by_id = {t.trajectory_id: t for t in active_train}
        pairs = self._evaluation_pairs(traj_by_id)
        # The last measured lift report's zero-lift axes, so the deletion
        # floor stops counting a criterion that moves no pair (see
        # `constraints.check_candidate`). Empty before the first measurement.
        kwargs = dict(
            zero_lift_criteria=getattr(self, "_zero_lift_criteria", ()),
            zero_lift_rules=getattr(self, "_zero_lift_rules", ()),
            parent_filter=parent_filter,
            trajectories_by_id=traj_by_id,
            truth_pairs=pairs,
            scorer=getattr(self.solver, "rubric_scorer", None) or HeuristicRubricScorer(),
            lazy_scoring=getattr(self.solver, "lazy_scoring", True),
            score_weights={**DEFAULT_SCORE_WEIGHTS, **self.config.score_weights},
            redundancy_weight=self.config.redundancy_penalty,
            redundancy_threshold=self.config.redundancy_threshold,
            # Every readout carries a constraint report measured against the
            # run's starting filter; the parent is the per-iteration baseline.
            # It is also where an outcome read is refused.
            initial_dir=self.initial_filter_dir,
            hard_caps=dict(self.config.complexity_ceiling) or None,
            # The baseline's stats as the proposer saw them are what
            # a NEW dead axis or redundant pair is measured against; the
            # frozen files are refused.
            baseline_stats=self._baseline_criterion_stats,
            freeze=self.config.freeze,
        )

        if not pairs and proposal.candidate_dirs:
            logger.info(
                "iter %d precheck: no already-judged pair lies in this pool "
                "(%d accumulated) — nothing to rank on, so no revision leg; "
                "growth and deletion caps still apply",
                iteration,
                len(self._accumulated_pairs),
            )
            # `write=False`: a precheck.json full of "no judged pair" would be
            # noise dressed as a measurement. The constraint reports are the
            # only thing taken from this pass, and they cost a file read.
            self._ensure_baseline_dir(parent_dir, parent_filter, iteration)
            results = run_precheck(proposal.candidate_dirs, write=False, **kwargs)
            kept, gated_out = gate_candidates(
                proposal.candidate_dirs, results, candidates_scored=None
            )
            return PrecheckOutcome(
                proposal=replace(proposal, candidate_dirs=kept),
                gated_out=gated_out,
                constraints=_constraint_dicts(results),
            )

        before: dict[str, dict] = {}
        results = {}
        block = ""
        parent_result = None
        if proposal.candidate_dirs:
            # The parent goes through the identical measurement so the
            # proposer is handed a reference point rather than an absolute
            # number it has no scale for. It is free: the baseline's cells for
            # this pool were paid for at the top of the iteration.
            self._ensure_baseline_dir(parent_dir, parent_filter, iteration)
            parent_result = run_precheck([parent_dir], **kwargs)[parent_dir.name]
            if kwargs.get("baseline_stats") is None:
                # No full pass has measured this baseline yet (iteration 0
                # with seeded pairs, or a resumed run): the parent's readout
                # over the judged rows is the next best statement of what is
                # dead in it.
                kwargs["baseline_stats"] = parent_result.criterion_stats
            results = run_precheck(proposal.candidate_dirs, **kwargs)
            before = {name: r.to_dict() for name, r in results.items()}
            block = render_precheck_block(
                results,
                parent=parent_result,
                redundancy_threshold=self.config.redundancy_threshold,
            )

        def _gated(
            final_proposal: ProposerResult,
            final,
            after: dict[str, dict],
            *,
            ranked: bool = True,
        ) -> PrecheckOutcome:
            kept, gated_out = gate_candidates(
                final_proposal.candidate_dirs,
                final,
                # A readout with no pairs behind it has no rank to cut on —
                # iteration 0's retry path — so only the caps gate there.
                candidates_scored=self.config.candidates_scored if ranked else None,
                # The parent's readout on the same pairs: what a no-op
                # candidate would be identical to. Same parent before and
                # after the revision leg, so measured once.
                parent=parent_result if ranked else None,
            )
            if gated_out:
                logger.info(
                    "iter %d: %d of %d candidate(s) gated out before scoring — %s",
                    iteration,
                    len(gated_out),
                    len(final_proposal.candidate_dirs),
                    "; ".join(f"{name}: {g['reason']}" for name, g in gated_out.items()),
                )
            return PrecheckOutcome(
                proposal=replace(final_proposal, candidate_dirs=kept),
                before=before,
                after=after,
                gated_out=gated_out,
                constraints=_constraint_dicts(final),
            )

        legs_before = proposal.legs
        self._reset_score_budget(iteration, "revise")
        try:
            proposal = self.proposer.revise(
                result=proposal,
                precheck_block=block,
                candidates_root=candidates_root,
                iteration=iteration,
            )
        except UsageLimitError:
            raise
        except Exception:
            # Leg 1's candidates are on disk and are perfectly scoreable. A
            # revision call that dies must cost the revision, not the
            # iteration — the loop has already paid for the generation leg.
            logger.exception(
                "iter %d: the proposer's revision leg failed — scoring leg 1's "
                "candidates as they stand",
                iteration,
            )
            return _gated(proposal, results, {})
        finally:
            self._fold_score_budget(iteration, "revise")

        if proposal.legs == legs_before:
            # The static proposer declines the leg — it answers from its
            # prompt alone and has no workspace to revise in. Nothing changed
            # on disk, so re-measuring would report the same numbers under a
            # heading that claims a revision happened.
            return _gated(proposal, results, {})

        self.store.save_proposer_io(iteration, proposal.prompt, proposal.response.text)
        self._assert_no_selection_exposure(proposal.prompt, iteration=iteration, leg="revise")
        self._assert_no_forbidden_reads(proposal.response, iteration=iteration, leg="revise")
        after_results = {}
        # `pairs` can be empty here only on the retry path, where leg 1 wrote
        # nothing and there was never anything to check against. Writing a
        # precheck.json full of "no judged pair" for those candidates would
        # be noise dressed as a measurement — but the caps are file-level
        # facts and still gate (same as the no-pairs branch above), and with
        # no score there is no rank, so `candidates_scored` does not cut.
        if proposal.candidate_dirs and pairs:
            after_results = run_precheck(proposal.candidate_dirs, **kwargs)
        elif proposal.candidate_dirs:
            self._ensure_baseline_dir(parent_dir, parent_filter, iteration)
            after_results = run_precheck(proposal.candidate_dirs, write=False, **kwargs)
            return _gated(proposal, after_results, {}, ranked=False)
        return _gated(
            proposal, after_results, {name: r.to_dict() for name, r in after_results.items()}
        )

    def _solver_workers(self) -> int:
        """Worker count the rubric scorer is currently configured for."""

        return int(getattr(getattr(self.solver, "rubric_scorer", None), "max_workers", 1) or 1)

    def _apply_governor(self, k: int, n_candidates: int) -> int:
        """Re-plan concurrency for iteration `k` and push it into the scorer.

        Returns how many candidates may run at once. `solver_parallel` lives on
        the scorer rather than in the config, so it has to be written back
        there for the next prefetch to see it.
        """

        if self.governor is None:
            return _candidate_workers(self.config.candidates_parallel, n_candidates)

        requested_candidates = self.config.candidates_parallel or n_candidates
        plan = self.governor.plan(
            candidates=min(requested_candidates, n_candidates),
            solver=self.requested_solver_parallel,
            judge=self.requested_judge_parallel,
        )
        log_concurrency_plan(plan)

        scorer = getattr(self.solver, "rubric_scorer", None)
        if scorer is not None and hasattr(scorer, "max_workers"):
            scorer.max_workers = plan.solver
        self.config.judge_parallel = plan.judge
        return max(1, min(plan.candidates, n_candidates))

    def _scorer_failed_rows(
        self, decision_lists: list[list[FilterDecision]] | None = None
    ) -> set[str]:
        """Rows this iteration's scoring pass could not produce a score for.

        Two sources, unioned, because each sees something the other cannot.
        The decisions carry `scorer_failed`, which names the row that failed
        on the candidate it failed on — including a refusal, which never
        reaches the parse counter. The scorer's own `failed_row_ids` catches
        rows that failed during the parallel prefetch, where the exception is
        swallowed before a decision is ever built.

        Empty for the heuristic scorer, which cannot fail either way, and for
        any scorer predating the counter — the loop asks rather than assumes,
        so a swapped-in scorer is not required to grow the attribute.
        """

        scorer = getattr(self.solver, "rubric_scorer", None)
        failed = set(getattr(scorer, "failed_row_ids", set()) or set())
        for decisions in decision_lists or []:
            failed.update(d.trajectory_id for d in decisions if d.scorer_failed)
        return failed

    def run(self) -> Path:
        """Run the full optimization loop and return the run directory."""

        cfg = RunConfig(
            seed=self.config.seed,
            iterations=self.config.iterations,
            candidates_per_iteration=self.config.candidates_per_iteration,
            llm_provider=type(unwrap(self.proposer.client)).__name__,
            llm_model=getattr(self.proposer.client, "model", None),
            loss_weights=self.config.loss_weights,
            train_data=str(self.train_path),
            validation_data=str(self.heldout_path) if self.heldout_path else "",
            initial_filter=str(self.initial_filter_dir),
            # Effective values, not the CLI strings: what the run actually
            # used, including defaults the launch line never mentioned.
            score_weights={**DEFAULT_SCORE_WEIGHTS, **self.config.score_weights},
            pair_budget=self.config.pair_budget,
            max_pairs=self.config.max_pairs,
            truth_includes_consensus=self.config.truth_includes_consensus,
            consensus_audit_share=self.config.consensus_audit_share,
            redundancy_penalty=self.config.redundancy_penalty,
            redundancy_threshold=self.config.redundancy_threshold,
            pair_evidence=self.config.pair_evidence,
            min_distinct_levels=self.config.min_distinct_levels,
            pair_judge_cache_dir=str(self.config.pair_judge_cache_dir),
            selection_data=str(self.config.selection_data),
            selection_pairs=(
                self.config.selection_pairs if self.config.selection_set is None else None
            ),
            selection_seed=(
                self.config.selection_seed if self.config.selection_set is None else None
            ),
            selection_set=(
                str(self.config.selection_set) if self.config.selection_set is not None else None
            ),
            gate_z=self.config.gate_z,
            gate_min_discordant=self.config.gate_min_discordant,
            patience=self.config.patience,
            min_iterations=self.config.min_iterations if self.config.patience else None,
            candidates_scored=self.config.candidates_scored,
            **self._solver_settings(),
            proposer_score_budget=(
                self.config.proposer_score_budget if self.score_tool_enabled else None
            ),
            complexity_ceiling=dict(self.config.complexity_ceiling) or None,
            freeze=list(self.config.freeze) if self.config.freeze else None,
        )
        # A tool-using proposer reads files, not lists: pointed at the pool the
        # CLI named it would open the raw rows. So the run records — and its
        # tools load — a stripped copy inside the run directory, which is also
        # where the proposer is confined to look.
        stripped = self._write_metadata_free_train()
        cfg.train_data_source_sha256 = file_sha256(self.train_path)
        cfg.train_data = str(stripped)
        self._write_metadata_free_sources()
        self.store.save_config(cfg)

        resumed = resume_state(self.store.run_dir) if self.config.resume else None

        # Iteration directories at or past the starting point are leftovers of
        # an aborted attempt — a relaunch under the same run id, or the killed
        # tail of the run being resumed. They must not share directories with
        # the work about to happen: several generations of
        # `iterations/000/candidates` side by side cannot be told apart
        # without reading validation.json. Candidate collection would be safe
        # (it snapshots what pre-exists), but the agentic proposer
        # investigates the run directory with tools.
        moved = self._quarantine_partial_iterations(
            resumed.start_iteration if resumed is not None else 0
        )

        if resumed is not None:
            # Resuming reuses the filter the killed iteration was working from,
            # rather than re-copying the initial filter over it. That snapshot
            # may itself live in a just-quarantined directory — follow it.
            filter_dir = resumed.filter_dir
            for old, new in moved.items():
                if filter_dir == old or old in filter_dir.parents:
                    filter_dir = new / filter_dir.relative_to(old)
                    break
            current = load_filter(filter_dir)
            logger.info(
                "resuming at iteration %d from %s",
                resumed.start_iteration,
                filter_dir,
            )
        else:
            # Bootstrap: snapshot the initial filter into the run so we can trace it.
            active_dir = self.store.run_dir / "iterations" / "000" / "current_filter"
            active_dir.parent.mkdir(parents=True, exist_ok=True)
            if active_dir.exists():
                shutil.rmtree(active_dir)
            shutil.copytree(self.initial_filter_dir, active_dir)
            current = load_filter(active_dir)

        history: list[CandidateEvaluation] = []
        all_evaluations: list[CandidateEvaluation] = []
        pool_overview = _summarize_trajectory_pool(self.train)

        # The held-out pool is adjudicated once, up front. Unlike the training
        # map these verdicts never borrow from candidate consensus: a number
        # the loop can influence is no evidence that the rubric generalizes.
        heldout_verdicts = self._resolve_heldout_verdicts()

        # The previous iteration's rubric critic report and ranking state,
        # carried into the next proposer call. On iteration 0 the report is
        # None and the prompt renders a "(not available yet)" placeholder.
        prev_rubric_report_dict: dict | None = None
        prev_ranking_state: dict = {}
        prev_false_negatives: list[str] = []
        prev_false_positives: list[str] = []

        # Option 3 self-continuity signals — accumulate across iterations so
        # each proposer call sees its own past diagnoses, quality-metric
        # trajectory, and the sequence of winning edits. All three are
        # derived from artifacts the optimizer already writes, so no extra
        # LLM calls or new persistence layer is needed.
        past_diagnoses: list[dict] = []
        quality_trajectory: list[dict] = []
        past_winners: list[dict] = []

        # Stagnation telemetry for the proposer. A rubric can survive a whole
        # run byte-identical while most candidates edit it and every one
        # loses selection, and no prompt would say so. These are measured here and rendered into the
        # proposer prompt so the escalation instruction is keyed to signals
        # the proposer can actually see. On resume the counters restart from
        # the resume point; an undercounted rubric_age only softens the
        # escalation message, never breaks anything.
        rubric_age = 0
        rubric_ever_changed = False
        rubric_edit_candidates = 0
        rubric_edit_wins = 0
        prev_rubric_hash = hashlib.md5(current.rubric_md.encode("utf-8")).hexdigest()
        novelty_history: list[float] = []
        prev_stagnation_note = ""
        probe_trend = ""
        criterion_lift_line = ""
        # Criteria the LAST measured lift report put at exactly zero. They
        # describe the baseline these candidates were edited from, which is
        # what the deletion floor needs: a zero-lift axis is not discriminating
        # power, so removing it is credited back (constraints.check_candidate).
        self._zero_lift_criteria: tuple[str, ...] = ()
        self._zero_lift_rules: tuple[str, ...] = ()
        ranking_block = ""
        pair_evidence_block = ""
        criterion_health_block = ""
        criterion_stability_block = ""
        criterion_signal_block = ""
        rationale_themes_block = ""

        start_iteration = 0
        if resumed is not None:
            # Without these the resumed proposer would reason as if it were at
            # iteration 0 — blind to its own past diagnoses, which looks like
            # working rather than like failing.
            start_iteration = resumed.start_iteration
            past_diagnoses = list(resumed.past_diagnoses)
            quality_trajectory = list(resumed.quality_trajectory)
            past_winners = list(resumed.past_winners)

        # The hidden selection set is judged before iteration 0 spends
        # anything: with no judge it cannot be built, and finding that out
        # after a proposer call is the late failure every other start-up
        # check here exists to avoid. On resume the saved file is reloaded,
        # so the exam does not move under the run.
        self._ensure_selection_set(current)
        self._log_gate_thresholds()

        # Patience. Consecutive iterations that scored a candidate and kept
        # the baseline. Restarts from zero on resume, like the stagnation
        # counters: an undercount only delays the stop, never breaks it.
        patience_counter = 0
        stop_reason = "iterations_exhausted"
        last_iteration = self.config.iterations - 1

        consecutive_proposer_failures = 0
        for k in range(start_iteration, self.config.iterations):
            logger.info("=== iteration %d ===", k)
            self.store.save_current_filter(k, current)
            # Both counters are per iteration. A scorer parse failure belongs
            # to the rubric that provoked it, and spend belongs to the row it
            # is reported on, so each iteration starts from a clean mark.
            reset_failures = getattr(
                getattr(self.solver, "rubric_scorer", None), "reset_failure_stats", None
            )
            if callable(reset_failures):
                reset_failures()
            cost_mark = COST_LEDGER.snapshot()

            # Determine this iteration's active training subset. Regression
            # memory is always replayed in full below; only the fresh scan
            # is subsampled.
            active_train = self._active_train_for_iteration(k)
            if len(active_train) != len(self.train):
                logger.info(
                    "iter %d: training on subset %d/%d (rate=%.2f)",
                    k,
                    len(active_train),
                    len(self.train),
                    len(active_train) / max(1, len(self.train)),
                )

            candidates_root = self.store.iteration_dir(k) / "candidates"
            candidates_root.mkdir(parents=True, exist_ok=True)

            # 1. Materialize the baseline and collect its solver decisions.
            #    We do NOT run critics on it yet — the entire premise of this
            #    loop is that we have no gold, so single-filter FN/FP has no
            #    meaning until we have a disagreement cohort.
            baseline_dir = candidates_root / f"iter{k:03d}_baseline"
            if baseline_dir.exists():
                shutil.rmtree(baseline_dir)
            shutil.copytree(current.directory, baseline_dir)
            baseline_loaded = load_filter(baseline_dir)
            # If the solver's scorer is LLM-backed, prefetch scores in parallel
            # so the sequential Solver.run below hits the cache. With lazy
            # scoring the solver prefetches only the rows whose decision
            # actually reads the rubric — a blanket prefetch here would pay
            # for every skipped row and erase the saving.
            if not getattr(self.solver, "lazy_scoring", False):
                self._prefetch_scorer(baseline_loaded, active_train)
            baseline_solver_decisions = self.solver.run(baseline_loaded, active_train)

            # 1b. Rank lift for the proposer, when nothing carries one in.
            #
            # The measurement below normally runs at the END of an iteration,
            # on the winner, so iteration k's propose call reads what
            # iteration k-1 produced — and iteration 0 reads nothing. That is
            # the worst place for the blind spot to be: a dead axis is usually
            # INHERITED from the seed filter rather than invented by a
            # candidate, and iteration 0 is where the seed rubric is edited
            # wholesale. Without the measurement a candidate can spend its
            # slot raising a zero-lift criterion's weight instead of removing
            # it.
            #
            # Everything it needs is already here: `current` is the incoming
            # filter and `baseline_solver_decisions` are its decisions over
            # `active_train`, both from step 1 above. Ablation re-runs
            # `decide()` on scores already paid for, so this costs no LLM
            # call. It is also the stricter counterfactual — the incoming
            # filter measured against its OWN decisions, rather than last
            # iteration's winner standing in for it.
            #
            # Only when nothing carries in, so every existing number is
            # untouched: iteration 0, a resume, or an iteration whose
            # end-of-loop measurement raised.
            if not criterion_lift_line:
                try:
                    from meta_filter.criterion_lift import (
                        measure_ranking as _measure_rank_lift,
                    )
                    from meta_filter.criterion_lift import (
                        render_ranking as _render_rank_lift,
                    )

                    seed_lift = _measure_rank_lift(current, active_train, baseline_solver_decisions)
                    criterion_lift_line = _render_rank_lift(seed_lift)
                    criterion_lift_line = _append_rule_pricing(
                        criterion_lift_line, current, baseline_solver_decisions, active_train
                    )
                    _zl_crit, _zl_rules = zero_lift_items(seed_lift)
                    self._zero_lift_criteria = tuple(_zl_crit)
                    self._zero_lift_rules = tuple(_zl_rules)
                    self.proposer.zero_lift_note = zero_lift_note(seed_lift)
                    logger.info("iter %d (pre-propose): %s", k, criterion_lift_line)
                    # Recorded with a stage marker so the file still reads one
                    # record per iteration for the end-of-loop measurement,
                    # and the run can still show what the proposer was told.
                    with (self.store.run_dir / "criterion_lift.jsonl").open(
                        "a", encoding="utf-8"
                    ) as fp:
                        fp.write(
                            json.dumps({"iteration": k, "stage": "pre_propose", **seed_lift}) + "\n"
                        )
                except Exception:
                    logger.exception("iter %d: pre-propose rank lift failed — continuing", k)

            # 2. PROPOSER — sees the current filter, its misordered pairs from
            #    the previous iteration (none at iteration 0) and the loop's
            #    telemetry.
            provisional_baseline_metrics = _stub_metrics(
                len(active_train), baseline_solver_decisions
            )
            self._reset_score_budget(k, "propose")
            try:
                proposal = self.proposer.propose(
                    loaded_filter=current,
                    candidates_root=candidates_root,
                    iteration=k,
                    current_metrics=provisional_baseline_metrics,
                    recent_false_negatives=prev_false_negatives,
                    recent_false_positives=prev_false_positives,
                    history=history,
                    pareto_snapshot=all_evaluations,
                    pool_overview=pool_overview,
                    rubric_critic_report=prev_rubric_report_dict,
                    past_diagnoses=past_diagnoses[-3:],  # last 3 iters only, budget
                    quality_trajectory=quality_trajectory[-6:],
                    past_winners=past_winners[-3:],
                    ranking_state=self._ranking_state(prev_ranking_state, active_train),
                    loop_telemetry={
                        "rubric_age": rubric_age,
                        "rubric_never_changed": not rubric_ever_changed,
                        "rubric_edit_candidates": rubric_edit_candidates,
                        "rubric_edit_wins": rubric_edit_wins,
                        "recent_novelty": novelty_history[-6:],
                        "critic_note": prev_stagnation_note,
                        "probe_trend": probe_trend,
                        "criterion_lift": criterion_lift_line,
                        "ranking_block": ranking_block,
                        "pair_evidence": pair_evidence_block,
                        "criterion_health": criterion_health_block,
                        "criterion_stability": criterion_stability_block,
                        "criterion_signal": criterion_signal_block,
                        "rationale_themes": rationale_themes_block,
                    },
                )
            # A proposer LLM call can time out or the provider can 5xx. One bad
            # call must not discard the other iterations' work — but a *run* of
            # them is not bad luck, it is a broken configuration, and skipping
            # through it would let every iteration fail identically and the
            # run still print "run complete" and exit 0. A run that cannot
            # propose is over; it should say so.
            # A cap is conclusive on its first occurrence: the account cannot
            # call at all, so counting to three only buys three identical
            # 400s and a cause line pointing at configuration. Stop here, and
            # let the cap's own message be the reason.
            except UsageLimitError as exc:
                raise RuntimeError(
                    f"iteration {k}: the API account is out of allowance, so no "
                    f"further call in this run can succeed — {exc}"
                ) from exc
            except Exception as exc:
                consecutive_proposer_failures += 1
                logger.exception("iter %d: proposer failed — skipping iteration", k)
                if consecutive_proposer_failures >= MAX_CONSECUTIVE_PROPOSER_FAILURES:
                    raise RuntimeError(
                        f"proposer failed {consecutive_proposer_failures} times in a "
                        f"row (iterations {k - consecutive_proposer_failures + 1}-{k}); "
                        "this is a configuration failure, not a transient one"
                    ) from exc
                continue
            finally:
                # Whatever the leg did, what its tool calls cost is the
                # proposer's spend this iteration.
                self._fold_score_budget(k, "propose")
            self.store.save_proposer_io(k, proposal.prompt, proposal.response.text)
            self._assert_no_selection_exposure(proposal.prompt, iteration=k, leg="propose")
            try:
                self._assert_no_forbidden_reads(proposal.response, iteration=k, leg="propose")
            except MetadataFreeReadError as exc:
                if self._count_invalidated(k, proposal, exc, consecutive_proposer_failures):
                    consecutive_proposer_failures += 1
                continue

            # 2b. PRECHECK + the one revision leg (fix #9). Between writing
            #     candidates and paying to score them, the loop can already
            #     say what each one scores: the truth pairs from earlier
            #     iterations are judged, and every criterion the candidate
            #     left alone is a cache hit. So measure it, hand it back, and
            #     let the proposer repair what it exposes — once.
            #
            #     The leg budget is two and the bound is load-bearing. A
            #     precheck the proposer can iterate against is an exam it can
            #     memorize, on pairs bought before this iteration, with no new
            #     verdict to catch the overfit; and each extra leg is another
            #     full agentic call against a candidate set nobody has judged.
            try:
                precheck_outcome = self._precheck_and_revise(
                    proposal,
                    iteration=k,
                    parent_dir=baseline_dir,
                    parent_filter=baseline_loaded,
                    active_train=active_train,
                    candidates_root=candidates_root,
                )
            except MetadataFreeReadError as exc:
                if self._count_invalidated(k, proposal, exc, consecutive_proposer_failures):
                    consecutive_proposer_failures += 1
                continue
            # A proposal that survived both legs' audits: the proposer works.
            consecutive_proposer_failures = 0
            proposal = precheck_outcome.proposal
            precheck_before = precheck_outcome.before
            precheck_after = precheck_outcome.after
            # One file per iteration for the gate's decisions, beside the
            # per-candidate readouts. `gated_out` directories stay on disk —
            # the next proposer may read what was held back and why — but
            # nothing below this line scores them.
            self._write_iteration_precheck(k, precheck_outcome)

            # 3. Run solver on every proposed candidate. We collect *all*
            #    candidate decisions before touching the critic, because the
            #    consensus / disagreement partition is what tells us where
            #    the critic actually needs to work.
            candidate_solver: list[tuple[str, Path, bool, object, list[FilterDecision]]] = [
                (baseline_dir.name, baseline_dir, True, baseline_loaded, baseline_solver_decisions)
            ]

            # What the two backstops below dropped, candidate id -> reason.
            # The precheck is meant to have gated an unscoreable candidate
            # already (`constraints.rubric_criteria_defect`), so anything
            # landing here is a hole in that gate, and it is recorded: without
            # this, precheck.json would still list every candidate as scored,
            # `gated_out` would be empty, and summary.csv would show only a
            # smaller `n_candidates`.
            discarded_at_scoring: dict[str, str] = {}

            def _evaluate_candidate(cdir_arg: Path):
                try:
                    return _evaluate_candidate_inner(cdir_arg)
                except Exception:
                    # A candidate is code an LLM wrote this iteration, and it
                    # can simply be wrong — a rules.py reading a dict key that
                    # does not exist. Uncaught, the KeyError would travel out
                    # of the candidate, out of the solver, out of the thread
                    # pool, and end a run that had already paid for its
                    # earlier iterations.
                    #
                    # A candidate whose own code raises is a broken candidate,
                    # not a broken run — the same thing a contract violation
                    # already means here, so it takes the same exit: discarded
                    # loudly, with the run continuing on its siblings.
                    logger.exception(
                        "%s: candidate code raised during evaluation — "
                        "candidate discarded, run continues",
                        cdir_arg.name,
                    )
                    return cdir_arg, None, None, "candidate code raised during evaluation"

            def _evaluate_candidate_inner(cdir_arg: Path):
                loaded = load_filter(cdir_arg)
                # Ranking contract, enforced BEFORE any scoring is paid: a
                # candidate without RUBRIC_CRITERIA gets nothing scored, so
                # its decide() falls back to one constant score for every
                # row — a silent blank sheet. A contract violation must be a
                # loud discard, not a quietly broken candidate.
                #
                # This is a backstop rather than the enforcement: the
                # precheck fires `invalid` on the same contract and the gate
                # drops the candidate before this runs. Reaching here means
                # the precheck missed a shape — so it is an ERROR, and the
                # name goes into the iteration record.
                criteria = getattr(loaded.policy_module, "RUBRIC_CRITERIA", None)
                if not criteria:
                    reason = (
                        "the ranking contract requires a non-empty RUBRIC_CRITERIA "
                        "in policy.py — candidate discarded before scoring"
                    )
                    logger.error("%s: %s", cdir_arg.name, reason)
                    return cdir_arg, None, None, reason
                if not getattr(self.solver, "lazy_scoring", False):
                    self._prefetch_scorer(loaded, active_train)
                decisions = self.solver.run(loaded, active_train)
                return cdir_arg, loaded, decisions, ""

            proposal_dirs = list(proposal.candidate_dirs)
            # Telemetry: how many candidates even try to move the rubric.
            for cand_dir in proposal_dirs:
                try:
                    cand_rubric = (cand_dir / "rubric.md").read_text(encoding="utf-8")
                except OSError:
                    continue
                if hashlib.md5(cand_rubric.encode("utf-8")).hexdigest() != prev_rubric_hash:
                    rubric_edit_candidates += 1
            # Observable, not enforced: a rewrite can still be the right answer,
            # but a run that drifts back to rewriting should say so rather than
            # be inferred from a flat score six iterations later.
            proposal_shapes = shape_report(
                self.store.run_dir / "iterations" / f"{k:03d}" / "current_filter",
                proposal_dirs,
            )
            # Re-derive the ceiling every iteration. A filter that grows is the
            # point of this loop, and the prompt carries the whole rubric, so a
            # per-call figure measured at iteration 0 is stale a few
            # iterations later.
            workers = self._apply_governor(k, len(proposal_dirs))

            with MemorySampler() as sampler:
                if workers > 1:
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        for cdir_arg, loaded, decisions, discard in pool.map(
                            _evaluate_candidate, proposal_dirs
                        ):
                            if loaded is None:
                                # Contract violation, already logged. Recorded
                                # here — in the consuming loop rather than in
                                # the worker — so the map's own thread owns
                                # every write to the dict.
                                discarded_at_scoring[cdir_arg.name] = discard
                                continue
                            candidate_solver.append(
                                (cdir_arg.name, cdir_arg, False, loaded, decisions)
                            )
                else:
                    for cdir in proposal_dirs:
                        cdir_arg, loaded, decisions, discard = _evaluate_candidate(cdir)
                        if loaded is None:
                            discarded_at_scoring[cdir_arg.name] = discard
                            continue  # contract violation, already logged
                        candidate_solver.append((cdir_arg.name, cdir_arg, False, loaded, decisions))

            if discarded_at_scoring:
                logger.error(
                    "iter %d: %d of %d candidate(s) vanished between the precheck and "
                    "the solver pass — %s",
                    k,
                    len(discarded_at_scoring),
                    len(proposal_dirs),
                    "; ".join(f"{n}: {r}" for n, r in sorted(discarded_at_scoring.items())),
                )

            if self.governor is not None:
                per_call = self.governor.observe(
                    sampler=sampler,
                    concurrent_calls=workers * self._solver_workers(),
                )
                logger.info(
                    "iter %d: peak %.1f GB over baseline, per-call estimate now %.2f GB",
                    k,
                    (sampler.peak - sampler.baseline) / 1024**3,
                    per_call,
                )

            # 4. Every scored candidate on the hidden selection
            #    rows, same solver and same worker count as the train pass.
            #    The baseline's scores are carried, not re-scored — see
            #    `_baseline_selection_scores`. Nothing from here reaches the
            #    proposer; the correctness maps feed the gate and the CSV.
            selection_scores_by_cand: dict[str, dict[str, float]] = {}
            if self._baseline_selection_scores is None:
                # Only a resumed run without a usable `baseline_scores.json`
                # gets here: a fresh run scored the initial filter while
                # building the set. The re-score reads the canonical cache,
                # so it is the same draw the gate last measured against.
                logger.info("iter %d: re-scoring the resumed baseline on the selection rows", k)
                self._baseline_selection_scores = self._score_on_selection_rows(baseline_loaded)
            selection_scores_by_cand[baseline_dir.name] = dict(self._baseline_selection_scores)
            selection_scores_by_cand.update(
                self._score_candidates_on_selection(
                    [
                        (cand_id, loaded)
                        for cand_id, _, is_baseline, loaded, _ in candidate_solver
                        if not is_baseline
                    ],
                    workers=workers,
                )
            )

            # 5. Partition trajectories by candidate agreement. Any trajectory
            #    where at least two candidates give different decisions is a
            #    disagreement case; unanimous cases become pseudo-truth for
            #    free. This is what removes the need for a gold dataset.
            all_decision_lists = [d for (_, _, _, _, d) in candidate_solver]
            consensus_truth = _consensus_from_decisions(all_decision_lists, active_train)

            # Rows the scorer could not read an answer for, on any candidate.
            # They leave the exam entirely rather than being charged to the
            # candidate that hit them: the row carries no score because the
            # model's reply was unreadable, which is a property of the call
            # and not of the filter, and `unscored="loss"` would otherwise
            # bill a candidate for the router's bad minute. Excluding them
            # here removes them from pair selection, from concordance, and
            # from everything downstream that reads these score maps.
            scorer_failed_rows = self._scorer_failed_rows(all_decision_lists)
            if scorer_failed_rows:
                logger.warning(
                    "iter %d: %d row(s) had no readable score and are excluded "
                    "from this iteration's pairs: %s",
                    k,
                    len(scorer_failed_rows),
                    ", ".join(sorted(scorer_failed_rows)[:5]),
                )
            disagreement_ids = [
                t.trajectory_id
                for t in active_train
                if consensus_truth.get(t.trajectory_id) is None
            ]

            # 6. Ask the critic backend to resolve the disagreements. If the
            #    backend is the LLM judge, this is the only place LLM budget
            #    is spent per trajectory. Cached answers are reused across
            #    iterations, so the effective budget shrinks as the filter
            #    stabilizes.
            backend = getattr(self.validator, "critic_backend", None)
            truth_map: dict[str, Decision | None] = dict(consensus_truth)
            traj_by_id = {t.trajectory_id: t for t in active_train}

            # Supervision lives on ordered pairs, not on per-row verdicts.
            # Pairs every candidate orders the same way are free
            # pseudo-truth; a decile-stratified sample of the disputed pairs
            # goes to the pair judge.
            ranking_scores_by_cand: dict[str, dict[str, float]] = {}
            ranking_truth_pairs: list[tuple[str, str]] = []
            ranking_resolved_pairs: list[tuple[str, str]] = []
            ranking_tied_pairs: list[tuple[str, str]] = []
            ranking_audit_keys: set[frozenset[str]] = set()
            ranking_pair_judge = None
            from meta_filter import ranking as ranking_math
            from meta_filter.pair_judge import PairJudge

            for cand_id, _, _, _, decisions in candidate_solver:
                ranking_scores_by_cand[cand_id] = {
                    d.trajectory_id: float(d.rank_score)
                    for d in decisions
                    if d.rank_score is not None
                    and not d.scorer_failed
                    and d.trajectory_id not in scorer_failed_rows
                }
            # A candidate that emits no scores cannot vote on ordering —
            # including it would empty the id intersection and erase the
            # truth pairs for everyone. It still gets judged (concordance
            # 0 via the degeneracy guard); it just does not define truth.
            scoring_cands = {cid: m for cid, m in ranking_scores_by_cand.items() if m}
            pair_truth = ranking_math.truth_pairs(
                scoring_cands or {"_none": {}},
                max_pairs=self.config.max_pairs,
                disagreement_budget=self.config.pair_budget,
                seed=self.config.seed + k * 31,
                consensus_share=self.config.consensus_audit_share,
            )
            resolved_pairs: list[tuple[str, str]] = []
            audit_resolved: list[tuple[str, str]] = []
            to_judge = pair_truth.disagreement_pairs + pair_truth.audit_pairs
            if backend is not None and to_judge:
                ranking_pair_judge = PairJudge(
                    client=backend.client,
                    cache_dir=self.config.pair_judge_cache_dir,
                )
                # Both A/B orders, and only an agreement counts. The TIE
                # list is kept rather than discarded: those pairs may
                # already sit in the accumulated eval set from an earlier
                # iteration's single-order verdict, and they have to leave
                # the exam rather than be scored at half credit.
                resolution = ranking_pair_judge.resolve_pairs_detail(
                    to_judge,
                    traj_by_id,
                    max_workers=self.config.judge_parallel,
                )
                all_resolved = resolution.resolved
                ranking_tied_pairs = list(resolution.tied)
                logger.info(
                    "iter %d pair judge: %d resolved, %d TIE (%.1f%% of judged), %d unanswered",
                    k,
                    len(resolution.resolved),
                    len(resolution.tied),
                    100.0 * resolution.tie_share,
                    len(resolution.unresolved),
                )
                audit_keys = {frozenset(p) for p in pair_truth.audit_pairs}
                resolved_pairs = [p for p in all_resolved if frozenset(p) not in audit_keys]
                audit_resolved = [p for p in all_resolved if frozenset(p) in audit_keys]
            if pair_truth.audit_pairs:
                # The audit's own readout: the candidates ordered these
                # pairs one way, the judge ordered them independently.
                # Nothing here feeds selection — it measures whether the
                # consensus that makes up most of the truth set is right.
                consensus_order = set(pair_truth.audit_pairs)
                agreed = sum(1 for p in audit_resolved if p in consensus_order)
                # Two denominators, because they answer different
                # questions. `audit_resolved` is what the judge actually
                # ruled on, and is the right base for the agreement rate.
                # `audit_pairs` is what the audit PAID for; the gap is
                # pairs whose two presentation orders disagreed or that
                # went unanswered. Reporting only the first made a
                # shrinking denominator read as a shrinking audit, which
                # is exactly backwards — the audit spent the same budget
                # and learned less.
                logger.info(
                    "iter %d consensus audit: judge agreed with %d of %d "
                    "adjudicated consensus pair(s) (%.1f%%); %d pair(s) "
                    "were paid for, the other %d came back TIE or "
                    "unanswered and carry no verdict",
                    k,
                    agreed,
                    len(audit_resolved),
                    100.0 * agreed / max(len(audit_resolved), 1),
                    len(pair_truth.audit_pairs),
                    len(pair_truth.audit_pairs) - len(audit_resolved),
                )
            # The audit pairs belong here too. A pair every candidate
            # ordered the same way and the judge reversed is the lineage's
            # blind spot, which makes it the most informative example the
            # proposer can be shown — more so than a pair the candidates
            # were already arguing about. Scoring them without ever showing
            # them spends the judge call and throws the lesson away.
            ranking_resolved_pairs = resolved_pairs + audit_resolved
            ranking_audit_keys = {frozenset(p) for p in audit_resolved}
            ranking_truth_pairs = (
                (pair_truth.consensus_pairs if self.config.truth_includes_consensus else [])
                + resolved_pairs
                + audit_resolved
            )
            # The accumulated verdict history replaces the per-iteration
            # sample for SCORING only. The sampling above keeps running
            # untouched — it is the acquisition function, and what it buys
            # is added to the exam right here rather than discarded.
            added = self._accumulate_eval_pairs(
                ranking_resolved_pairs,
                audit_keys=ranking_audit_keys,
                iteration=k,
            )
            # A TIE carries a verdict and no order, so it never enters
            # the accumulated set — but it was paid for, and the dump
            # reports it, so its provenance is recorded here.
            paid_audit_keys = {frozenset(p) for p in pair_truth.audit_pairs}
            for tied_pair in ranking_tied_pairs:
                self._pair_provenance.setdefault(
                    frozenset(tied_pair),
                    {
                        "source": (
                            "audit" if frozenset(tied_pair) in paid_audit_keys else "disputed"
                        ),
                        "judged_iteration": k,
                    },
                )
            scoreable = self._evaluation_pairs(traj_by_id)
            if scoreable:
                ranking_truth_pairs = scoreable
                logger.info(
                    "iter %d ranking: scoring on the accumulated eval set "
                    "— %d of %d pairs lie in this pool (%d new this iteration); "
                    "the %d sampled pairs were acquisition only",
                    k,
                    len(scoreable),
                    len(self._accumulated_pairs),
                    added,
                    len(pair_truth.disagreement_pairs) + len(pair_truth.audit_pairs),
                )
            elif not self._accumulated_pairs:
                # Nothing resolved yet (every judged pair so far was a TIE, or
                # none was judged): there is no accumulated exam to score on,
                # which is an expected state rather than a fault.
                logger.warning(
                    "iter %d ranking: no judged pair has been resolved yet — "
                    "scoring on the sampled pairs.",
                    k,
                )
            else:
                logger.error(
                    "iter %d ranking: the accumulated eval set shares NO "
                    "pair with this pool (%d accumulated) — falling back "
                    "to the sampled pairs.",
                    k,
                    len(self._accumulated_pairs),
                )
            # The accumulated set only grows; the applicable subset can
            # shrink when the pool changes under a curriculum, so the
            # invariant belongs on the accumulated count.
            if self._eval_history and len(self._accumulated_pairs) < self._eval_history[-1]:
                logger.error(
                    "iter %d ranking: the accumulated eval set SHRANK "
                    "(%d -> %d) — it is append-only by construction, so "
                    "this is a bug, not a configuration",
                    k,
                    self._eval_history[-1],
                    len(self._accumulated_pairs),
                )
            self._eval_history.append(len(self._accumulated_pairs))
            if not ranking_truth_pairs:
                # With zero truth pairs every candidate's concordance is
                # zero and selection degenerates to the secondary terms,
                # which can crown a filter that cannot rank. Say it as
                # loudly as a log can.
                logger.error(
                    "iter %d ranking: ZERO truth pairs — the agreement "
                    "term is blind this iteration; selection rests on "
                    "critic factors alone",
                    k,
                )
            logger.info(
                "iter %d ranking: universe=%d consensus_pairs=%d disputed_sampled=%d resolved=%d",
                k,
                pair_truth.sampled_universe,
                len(pair_truth.consensus_pairs),
                len(pair_truth.disagreement_pairs),
                len(resolved_pairs),
            )
            # Written twice. This one lands before any candidate has been
            # scored, so an iteration that dies in the solver still leaves
            # a legible record of the verdicts it paid for; the second
            # write, after scoring, adds `by_candidate`.
            self._write_ranking_pairs(
                k,
                traj_by_id=traj_by_id,
                tied_pairs=ranking_tied_pairs,
                judge=ranking_pair_judge,
                scores_by_candidate=None,
            )
            logger.info(
                "iter %d: consensus=%d disagreement=%d",
                k,
                len(active_train) - len(disagreement_ids),
                len(disagreement_ids),
            )

            # 7. Run the validator on each candidate with the precomputed
            #    solver decisions and the combined verdict map, so nothing is
            #    re-executed and every candidate is judged against the same
            #    second opinion.
            bundles = []
            pair_misorders_by_cand: dict[str, tuple[list, list]] = {}
            # The pool-applicable slice of everything this run has bought,
            # which is the same list `ranking_truth_pairs` scores on. Falls
            # back to the fresh buy only when the accumulated set shares no
            # pair with this pool — the same fallback scoring takes, so the
            # evidence and the exam never disagree about what exists.
            misorder_evidence_pairs = self._evaluation_pairs(traj_by_id) or ranking_resolved_pairs

            def _pair_rationale(winner_id: str, loser_id: str) -> str:
                """The judge's own words on one pair, or ''. Never judges."""

                if ranking_pair_judge is None:
                    return ""
                return ranking_pair_judge.rationale_for(
                    winner_id,
                    loser_id,
                    # The rationale lives under the same key as the verdict,
                    # so it needs the same label view.
                    pair_view_signature(traj_by_id.get(winner_id), traj_by_id.get(loser_id)),
                )

            for cand_id, cdir, is_baseline, loaded, decisions in candidate_solver:
                bundle = self.validator.evaluate(
                    loaded_filter=loaded,
                    candidate_id=cand_id,
                    is_baseline=is_baseline,
                    train_trajectories=active_train,
                    heldout_trajectories=self.heldout,
                    memory=self.memory,
                    solver=self.solver,
                    iteration=k,
                    precomputed_train_decisions=decisions,
                    train_verdicts=truth_map,
                    heldout_verdicts=heldout_verdicts,
                )
                bundles.append((cand_id, cdir, is_baseline, loaded, bundle))
                self.store.save_solver_output(k, cand_id, bundle.train_decisions)
                # The judge speaks about pairs, so the FN/FP
                # the validator derives from per-row consensus are not what
                # this run's second opinion actually said. Replace them with
                # the judge's own disagreements rather than writing both.
                fn_records = bundle.false_negatives
                fp_records = bundle.false_positives
                misorders: list[CounterExample] = []
                # Note the missing `and misorder_evidence_pairs`. An
                # iteration with no adjudicated pair at all must write NO
                # judge evidence, not fall back to the validator's records:
                # those are derived from candidate consensus, and publishing
                # the committee's own opinion under the second-opinion
                # framing is the boundary self-copy defect one channel over.
                # The builder over an empty pair list returns two empty
                # lists, which is the honest artifact.
                fn_records, fp_records = _pair_misorder_examples(
                    scores=ranking_scores_by_cand.get(cand_id, {}),
                    # The accumulated exam, not this iteration's buy: the exam
                    # is what the candidate is actually scored on.
                    # `_select_misordered_pairs` keeps a minority of the slots
                    # for the fresh buy.
                    resolved_pairs=misorder_evidence_pairs,
                    traj_by_id=traj_by_id,
                    decisions_by_id={d.trajectory_id: d for d in decisions},
                    iteration=k,
                    filter_version=loaded.version,
                    rationale_for=_pair_rationale,
                    provenance=self._pair_provenance,
                    shown_before=self._shown_misorder_pairs,
                )
                misorders = fn_records + fp_records
                pair_misorders_by_cand[cand_id] = (fn_records, fp_records)
                if not misorder_evidence_pairs:
                    logger.warning(
                        "iter %d: %s has no judge-derived evidence — no "
                        "adjudicated pair lies in this pool, so "
                        "critic_outputs/ is empty by construction rather "
                        "than by the filter being right",
                        k,
                        cand_id,
                    )
                self.store.save_critic_output(
                    k,
                    cand_id,
                    fn_records,
                    fp_records,
                    misorders,
                )
                # The filter grading its own decisions. Kept, but out of
                # `critic_outputs/` — see `save_boundary_flags`.
                self.store.save_boundary_flags(k, cand_id, bundle.boundary)

            evaluations = [b.evaluation for *_, b in bundles]

            # ---- RubricCritic + RuleCritic ---------------------------------
            # Both run once per iteration. The audit targets the baseline
            # (that's what the misfires and FN/FP live on); every candidate
            # in this iter inherits the same quality signals for scoring.
            baseline_tuple = next(b for b in bundles if b[2])  # is_baseline
            _, _, _, baseline_loaded_local, baseline_bundle_obj = baseline_tuple
            baseline_evaluation = baseline_bundle_obj.evaluation

            curated_evidence, curated_ids = self._build_curated_evidence(
                backend=backend,
                active_train=active_train,
                baseline_evaluation=baseline_evaluation,
                baseline_decisions=baseline_solver_decisions,
                iteration=k,
            )
            rubric_report = self._run_rubric_critic(
                baseline_loaded_local.rubric_md, curated_evidence
            )
            rule_report = self._run_rule_critic(
                loaded_filter=baseline_loaded_local,
                trajectories=active_train,
                truth_map=truth_map,
            )

            # The agreement term is pairwise concordance with
            # the internally derived truth pairs. A degenerate scorer (too few
            # distinct values to express an ordering) gets zero — it cannot
            # rank, whatever its pair luck says. Distribution shape is saved
            # as DIAGNOSTICS for the critic and the operator, never scored:
            # rewarding a shape invites faking it with noise.
            ranking_diag: dict[str, dict] = {}
            from meta_filter import ranking as ranking_math
            from meta_filter.criterion_stats import criterion_stats

            decisions_by_cand = {name: decs for (name, _, _, _, decs) in candidate_solver}
            min_levels = (
                self.config.min_distinct_levels
                if self.config.min_distinct_levels is not None
                else ranking_math.default_min_distinct_levels(len(active_train))
            )
            for ev in evaluations:
                scores = ranking_scores_by_cand.get(ev.candidate_id, {})
                levels = ranking_math.distinct_levels(scores)
                # `unscored="loss"`: a row this candidate emitted no score
                # for costs this candidate its pairs, instead of removing
                # them from everyone's exam. See ranking.concordance_detail.
                detail = ranking_math.concordance_detail(
                    scores,
                    ranking_truth_pairs,
                    unscored="loss",
                    # A pair whose two presentation orders named different
                    # winners carries no verdict. Half credit would score
                    # every candidate on a pair nothing is known about.
                    ties=ranking_tied_pairs,
                )
                conc = detail.concordance
                if detail.n_unscored:
                    logger.warning(
                        "iter %d: %s violated the ranking contract — no "
                        "score for %d row(s), costing it %d of %d truth "
                        "pairs. Charged to this candidate, not to its "
                        "rivals.",
                        k,
                        ev.candidate_id,
                        detail.missing_ids,
                        detail.n_unscored,
                        detail.n_offered,
                    )
                degenerate = levels < min_levels
                # Below-chance concordance is a failure code, not a low
                # grade: a coin-flip scorer sits near 0.5 on any real
                # pair set, so anything under it means inverted or empty
                # scoring — say so, don't just let selection absorb it.
                if ranking_truth_pairs and conc < 0.5 and not degenerate:
                    logger.warning(
                        "iter %d: %s concordance %.3f is BELOW CHANCE "
                        "over %d truth pairs — inverted or broken scoring",
                        k,
                        ev.candidate_id,
                        conc,
                        len(ranking_truth_pairs),
                    )
                ev.metrics.critic_agreement = 0.0 if degenerate else conc
                ranking_diag[ev.candidate_id] = {
                    "concordance": conc,
                    "distinct_levels": levels,
                    "degenerate": degenerate,
                    # Contract coverage. Persisted because otherwise the
                    # only trace a row went unscored would be a log line.
                    "pairs_offered": detail.n_offered,
                    "pairs_scored": detail.n_scored,
                    "unscored_pairs": detail.n_unscored,
                    "unscored_rows": detail.missing_ids,
                    # Judge TIEs that left the exam. Same number for
                    # every candidate — it is a property of the judge,
                    # not of the filter — and the only record of what
                    # two-order adjudication cost this iteration.
                    "tied_pairs": detail.n_tied,
                    "distribution": ranking_math.distribution_report(scores, None),
                    # Deadness + redundancy over the candidate's own
                    # per-criterion scores; reported, never selected on.
                    "criterion_stats": criterion_stats(decisions_by_cand.get(ev.candidate_id, [])),
                }
            self._write_ranking_pairs(
                k,
                traj_by_id=traj_by_id,
                tied_pairs=ranking_tied_pairs,
                judge=ranking_pair_judge,
                scores_by_candidate=ranking_scores_by_cand,
            )
            score_weights = {**DEFAULT_SCORE_WEIGHTS, **self.config.score_weights}
            for ev in evaluations:
                ev.metrics.rubric_quality = rubric_report.rubric_quality
                ev.metrics.rule_quality = rule_report.rule_quality
                ev.metrics.score = compute_score(ev.metrics, weights=score_weights)
                # A degenerate scorer (too few distinct values to express an
                # ordering) must not merely lose the agreement term — when
                # the whole channel is blind, zero concordance is everyone's
                # score and the degenerate candidate can win on secondary
                # terms. Make it unable to win, period.
                diag = ranking_diag.get(ev.candidate_id)
                pen = 0.0
                if diag and self.config.redundancy_penalty > 0:
                    pen = redundancy_penalty(
                        diag.get("criterion_stats"),
                        weight=self.config.redundancy_penalty,
                        threshold=self.config.redundancy_threshold,
                    )
                    # Recorded whether or not it bit, so the next proposer
                    # prompt can state what this term actually cost the
                    # winner rather than only the weight it could cost.
                    diag["redundancy_penalty"] = pen
                    if pen > 0:
                        stats = diag["criterion_stats"]
                        logger.info(
                            "iter %d: %s redundancy penalty -%.3f (max |rho| %.2f between %s) — %s",
                            k,
                            ev.candidate_id,
                            pen,
                            stats["max_abs_corr"],
                            " / ".join(stats["max_pair"]),
                            "reported only, not charged to selection",
                        )
                # Which of the two scores the penalty lands in is decided in
                # one place, off the UNCHARGED score, so neither number can
                # end up paying it twice: selection is left alone and only
                # the reported score is charged (see
                # `criterion_stats.charge_redundancy` for why).
                selection_score, reported_score = charge_redundancy(ev.metrics.score, pen)
                ev.metrics.score = selection_score
                if diag and diag.get("degenerate"):
                    ev.metrics.score -= DEGENERACY_PENALTY
                    reported_score -= DEGENERACY_PENALTY
                    logger.warning(
                        "iter %d: %s is degenerate (%d distinct score levels) "
                        "— fatal score penalty applied",
                        k,
                        ev.candidate_id,
                        diag.get("distinct_levels", 0),
                    )
                if diag is not None:
                    # Both numbers, always, whichever mode ran: `score` is
                    # what selection sorted on and `score_with_redundancy` is
                    # what the filter is reported with. In select mode the
                    # penalty is already inside `score` and the two are equal.
                    diag["score"] = ev.metrics.score
                    diag["score_with_redundancy"] = reported_score

            def _write_ranking_json() -> None:
                """`ranking.json`, written now and again after selection.

                The paired-gate statistics per candidate only exist once the
                selector has run, and it runs after this file is first
                needed; rewriting the same payload with `ranking_diag`
                enriched keeps one file rather than two.
                """

                (self.store.iteration_dir(k) / "ranking.json").write_text(
                    json.dumps(
                        {
                            "truth_pairs": len(ranking_truth_pairs),
                            "tied_pairs": len(ranking_tied_pairs),
                            "min_distinct_levels": min_levels,
                            # Rows dropped before pair selection because the
                            # scorer could not read an answer for them. Zero
                            # is the expected value; anything else says this
                            # iteration was graded on a smaller pool than the
                            # one it was given.
                            "scorer_failed_rows": len(scorer_failed_rows),
                            "scorer_failed_row_ids": sorted(scorer_failed_rows),
                            # Per iteration, not per run: the exam grows every
                            # iteration, so one run-level hash cannot say
                            # what any given iteration was graded on.
                            "eval_set": _eval_set_identity(
                                ranking_truth_pairs,
                                accumulated=len(self._accumulated_pairs),
                            ),
                            # Which regime this iteration's selection ran
                            # under, beside the weight it ran with — the two
                            # together are what make `score` here readable.
                            "redundancy_mode": "report",
                            "redundancy_penalty_weight": self.config.redundancy_penalty,
                            "redundancy_threshold": self.config.redundancy_threshold,
                            # How many judged pairs the acceptance gate
                            # graded on.
                            "selection_pairs": (
                                len(self.selection_set) if self.selection_set else None
                            ),
                            "candidates": ranking_diag,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            _write_ranking_json()
            self.store.save_v2_critic_reports(
                k,
                rubric_report=rubric_report,
                rule_report=rule_report,
                curated_trajectory_ids=curated_ids,
            )
            # Carried into the next iteration's proposer call.
            prev_rubric_report_dict = rubric_report.to_dict()

            # Section E/C: what the gate held back is recorded where the
            # scored candidates are, with `invalid_reason` saying why and no
            # metrics — the solver never ran on them. Appended after the
            # ranking diagnostics so a held-back candidate is not charged
            # "no score for any row" by the contract check above.
            for name, held in precheck_outcome.gated_out.items():
                evaluations.append(
                    CandidateEvaluation(
                        candidate_id=name,
                        is_baseline=False,
                        metrics=_stub_metrics(len(active_train), []),
                        invalid_reason=held["reason"],
                    )
                )
            for cid, diag in ranking_diag.items():
                diag["constraint_status"] = (
                    "baseline" if cid == baseline_dir.name else precheck_outcome.status_of(cid)
                )
            self.store.save_validation(k, evaluations)

            # 4. SELECT. The decision is the paired McNemar gate on the
            #    hidden pairs; the train-side `score` is computed above and
            #    reported, but it does not pick the winner.
            assert self.selection_set is not None
            correctness_by_cand: dict[str, dict[tuple[str, str], float]] = {
                cid: pair_correctness(
                    scores, self.selection_set.pairs, ties=self.selection_set.tied
                )
                for cid, scores in selection_scores_by_cand.items()
            }
            # The full pass confirms what the precheck measured on the judged
            # rows. A candidate that introduces a dead axis or a redundant
            # pair on the whole pool needs strong evidence (the precheck's own
            # firings are kept).
            strong_ids = self._rubric_health_on_full_pass(
                k, ranking_diag, baseline_dir.name, precheck_outcome
            )
            result = self.selector.choose_paired(
                evaluations,
                correctness_by_cand,
                z_threshold=self.config.gate_z,
                min_discordant=self.config.gate_min_discordant,
                # A candidate over a soft cap clears a threshold one z-unit
                # higher (`STRONG_EXTRA_Z`), read off the final precheck's
                # constraint reports.
                strong_evidence=strong_ids,
                bootstrap_seed=self.config.seed + k,
                # A candidate smaller than the baseline in every complexity
                # component is accepted on `b >= c` alone, and two of them
                # are separated by size.
                simplification=precheck_outcome.simplification(),
                complexity_by_cand={
                    cid: complexity_score(report.get("candidate") or {})
                    for cid, report in precheck_outcome.constraints.items()
                },
            )
            for cid, stats in result.per_candidate.items():
                diag = ranking_diag.get(cid)
                if diag is not None:
                    diag["sel_concordance"] = stats["sel_concordance"]
                    diag["sel_b"] = stats["b"]
                    diag["sel_c"] = stats["c"]
                    diag["sel_z"] = stats["z"]
            base_diag = ranking_diag.get(baseline_dir.name)
            if base_diag is not None:
                base_diag["sel_concordance"] = result.baseline_sel_concordance
                base_diag["sel_b"] = 0
                base_diag["sel_c"] = 0
                base_diag["sel_z"] = 0.0
            _write_ranking_json()
            if result.winner == baseline_dir.name:
                gate_decision = "kept"
            elif result.simplification_winner:
                gate_decision = "accepted_simplification"
            else:
                gate_decision = "accepted"

            # Promotion re-score and confirmation. The winner's selection
            # number for the iterations ahead is a fresh draw, not the one
            # that won — and the fresh draw also has to agree with the
            # acceptance, because scorer cells are samples: a borderline
            # gate can pass on one cell draw and fail on the next. A
            # promotion needs draw 1 to pass, the re-draw to still favour
            # the candidate (b2 > c2), and the two draws pooled to clear the
            # same threshold draw 1 did. A candidate that fails that keeps
            # the baseline for this iteration; its directory stays on disk
            # as evidence.
            #
            # One acceptance skips the re-score entirely — a simplification
            # whose draw 1 read b = c = 0. See `_settled_without_a_redraw`.
            promoted_rescore_delta: float | None = None
            confirmation: PromotionConfirmation | None = None
            if result.winner != baseline_dir.name:
                winner_id = result.winner
                winner_loaded = next(b[3] for b in bundles if b[0] == winner_id)
                no_redraw = _settled_without_a_redraw(result)
                fresh: dict[str, float] | None = None
                delta: float | None = None
                if no_redraw:
                    logger.info(
                        "iter %d: %s ordered all %d selection pairs as the baseline did "
                        "(b=0, c=0) and is smaller in every complexity component — "
                        "confirming without a second draw, so no promotion re-score is bought",
                        k,
                        winner_id,
                        len(correctness_by_cand[baseline_dir.name]),
                    )
                else:
                    winning_cells = dict(
                        self._selection_cells.get(str(winner_loaded.directory), {})
                    )
                    fresh, delta = self._promotion_rescore(
                        k,
                        winner_id,
                        winner_loaded,
                        winning_scores=selection_scores_by_cand.get(winner_id, {}),
                    )
                    if fresh is not None:
                        # Two scorings of one filter on the same rows: the one
                        # measurement of per-criterion scorer consistency the
                        # loop ever takes, so it is kept and shown (3b).
                        criterion_stability_block = self._write_criterion_stability(
                            k,
                            winner_id,
                            winning_cells,
                            self._selection_cells.get(str(winner_loaded.directory), {}),
                        )
                confirmation = confirm_promotion(
                    b1=result.b,
                    c1=result.c,
                    fresh_correctness=(
                        pair_correctness(
                            fresh, self.selection_set.pairs, ties=self.selection_set.tied
                        )
                        if fresh is not None
                        else None
                    ),
                    baseline_correctness=correctness_by_cand[baseline_dir.name],
                    p_threshold=float(result.per_candidate[winner_id]["p_threshold"]),
                    simplification=result.simplification_winner,
                )
                if confirmation.confirmed:
                    if fresh is not None:
                        self._adopt_baseline_selection_scores(fresh, winner_loaded, iteration=k)
                        promoted_rescore_delta = delta
                        logger.info(
                            "iter %d: promotion of %s confirmed on the re-draw — %s",
                            k,
                            winner_id,
                            confirmation.reason,
                        )
                    else:
                        assert no_redraw
                        self._carry_baseline_selection_scores(
                            selection_scores_by_cand.get(winner_id, {}),
                            winner_loaded,
                            iteration=k,
                        )
                        logger.info(
                            "iter %d: promotion of %s confirmed without a re-draw — %s",
                            k,
                            winner_id,
                            confirmation.reason,
                        )
                        result = replace(
                            result,
                            reason=(
                                f"{result.reason} PROMOTION of {winner_id} CONFIRMED: "
                                f"{confirmation.reason}."
                            ),
                        )
                else:
                    gate_decision = "rejected_on_redraw"
                    logger.warning(
                        "iter %d: promotion of %s REJECTED on the re-draw — %s; keeping the "
                        "baseline (its selection scores are unchanged)",
                        k,
                        winner_id,
                        confirmation.reason,
                    )
                    result = replace(
                        result,
                        winner=baseline_dir.name,
                        reason=(
                            f"{result.reason} REJECTED ON RE-DRAW of {winner_id}: "
                            f"{confirmation.reason}; kept baseline."
                        ),
                    )
                    self._reread_baseline_selection_scores(k, baseline_loaded)
            logger.info("iter %d: %s", k, result.reason)
            self.store.save_selection(
                k,
                result.winner,
                result.reason,
                pareto_ids_complexity=result.pareto_ids_complexity,
                pareto_ids_cost=result.pareto_ids_cost,
                winner_pareto_safe=result.winner_pareto_safe,
                gate=_gate_record(result, gate_decision, confirmation),
                per_candidate=result.per_candidate,
            )

            selected = next(b for b in bundles if b[0] == result.winner)
            _, selected_dir, selected_is_baseline, selected_loaded, selected_bundle = selected

            # Patience bookkeeping. A proposer failure never reaches this
            # line (its `continue` is above), so those iterations leave the
            # counter alone by construction.
            scored_a_candidate = any(not is_b for _, _, is_b, _, _ in candidate_solver)
            if not selected_is_baseline:
                patience_counter = 0
            elif scored_a_candidate:
                patience_counter += 1

            # The misordered-pair blocks of the next proposer prompt: the judge's
            # pair verdicts the WINNER got backwards. Only judge-derived records,
            # never the validator's consensus-derived ones, and empty lists when
            # no adjudicated pair lies in the pool.
            prev_false_negatives, prev_false_positives = pair_misorders_by_cand[result.winner]
            # Only the head of the list reaches a prompt, so only the head
            # counts as shown. `_format_pair_misorders` renders
            # `over_ranked[:PROMPT_MISORDER_PAIRS]`, which is exactly this
            # slice, and the prompt it lands in is the next iteration's.
            self._record_shown_misorders(prev_false_negatives, iteration=k + 1)
            # Regression memory is not written: its buckets are defined on
            # per-row verdicts, and the ranking loop's supervision is pairs.

            # 6. Diff + report.
            diff_md = unified_diff(current.directory, selected_dir)
            self.store.save_diff(k, diff_md)
            report_md = render_iteration_report(
                iteration=k,
                baseline_id=baseline_dir.name,
                selected_id=result.winner,
                selection_reason=result.reason,
                evaluations=evaluations,
                diff_md=diff_md,
                new_fn_ids=[c.trajectory_id for c in selected_bundle.false_negatives],
                new_fp_ids=[c.trajectory_id for c in selected_bundle.false_positives],
                new_boundary_ids=[c.trajectory_id for c in selected_bundle.boundary],
                pareto_ids_complexity=result.pareto_ids_complexity,
                pareto_ids_cost=result.pareto_ids_cost,
                winner_pareto_safe=result.winner_pareto_safe,
            )
            self.store.save_report(k, report_md)

            # One reading of the ledger per iteration, shared by the record
            # and the summary row so the two can never disagree.
            iteration_costs = COST_LEDGER.columns(cost_mark)
            unpriced = COST_LEDGER.unpriced_calls(cost_mark)
            if unpriced:
                logger.warning(
                    "iter %d: %d call(s) had neither a reported cost nor a "
                    "list price, so this iteration's cost columns are a lower "
                    "bound",
                    k,
                    unpriced,
                )

            # What the selected filter is REPORTED with, as opposed to what
            # selection sorted on. Identical in select mode, where the
            # penalty is already inside `metrics.score`.
            winner_ranking_diag = ranking_diag.get(result.winner) or {}
            winner_redundancy = float(winner_ranking_diag.get("redundancy_penalty") or 0.0)
            winner_score_with_redundancy = float(
                winner_ranking_diag.get(
                    "score_with_redundancy",
                    selected_bundle.evaluation.metrics.score,
                )
            )

            record = IterationRecord(
                iteration=k,
                current_filter_version=current.version,
                selected_candidate=result.winner,
                selected_reason=result.reason,
                evaluations=evaluations,
                new_false_negatives=[c.trajectory_id for c in selected_bundle.false_negatives],
                new_false_positives=[c.trajectory_id for c in selected_bundle.false_positives],
                new_boundary=[c.trajectory_id for c in selected_bundle.boundary],
                diff=diff_md,
                scorer_failed_rows=len(scorer_failed_rows),
                scorer_failed_row_ids=sorted(scorer_failed_rows),
                discarded_at_scoring=dict(discarded_at_scoring),
                cost_solver=iteration_costs["cost_solver"],
                cost_judge=iteration_costs["cost_judge"],
                cost_proposer=iteration_costs["cost_proposer"],
                cost_critic=iteration_costs["cost_critic"],
                unpriced_calls=unpriced,
                precheck=precheck_before,
                precheck_after_revision=precheck_after,
                proposer_legs=proposal.legs,
                redundancy_penalty=winner_redundancy,
                score_with_redundancy=winner_score_with_redundancy,
            )
            self.store.save_iteration_record(record)

            selected_metrics = selected_bundle.evaluation.metrics
            # Held-out agreement is deliberately absent from this row. It lives
            # in the `_heldout` sink outside the run tree, because the agentic
            # proposer reads summary.csv and a generalization gap it can see is
            # a generalization gap it can optimize away.
            self.store.append_summary(
                {
                    "iteration": k,
                    "selected": result.winner,
                    "is_baseline": selected_is_baseline,
                    "loss": selected_metrics.loss,
                    "score": selected_metrics.score,
                    # The redundancy term, reported rather than selected on:
                    # `score` above is the train-side score, and
                    # `score_with_redundancy` is what the deployable filter
                    # is charged.
                    "redundancy_penalty": winner_redundancy,
                    "score_with_redundancy": winner_score_with_redundancy,
                    "critic_agreement": selected_metrics.critic_agreement,
                    "rubric_quality": selected_metrics.rubric_quality,
                    "rule_quality": selected_metrics.rule_quality,
                    "filter_accept_critic_reject": selected_metrics.filter_accept_critic_reject,
                    "filter_reject_critic_accept": selected_metrics.filter_reject_critic_accept,
                    "review_rate": selected_metrics.review_rate,
                    "complexity_score": selected_metrics.complexity_score,
                    "estimated_filter_cost": selected_metrics.estimated_filter_cost,
                    "n_candidates": len(bundles),
                    # Median share of each candidate that was new relative to
                    # the filter it started from. Near 0 means the proposer
                    # edited; near 1 means it rewrote and nothing accumulated.
                    "proposal_novelty": round(_median_novelty(proposal_shapes), 4),
                    "memory_must_accept": len(self.memory.must_accept()),
                    "memory_must_reject": len(self.memory.must_reject()),
                    # Rows the scorer could not read, excluded from the exam.
                    "scorer_failed_rows": len(scorer_failed_rows),
                    # Spend per role for THIS iteration, from what each route
                    # reported (`llm/cost_meter.py`). Four columns rather than
                    # one total: the roles differ by two orders of magnitude
                    # in unit price, so a single number cannot be acted on.
                    **iteration_costs,
                    # The gate's columns. Empty cells mean "not measured"
                    # rather than zero.
                    **_selection_summary_columns(
                        result,
                        gate_decision,
                        promoted_rescore_delta=promoted_rescore_delta,
                        patience=patience_counter,
                        selection_pairs=(
                            len(self.selection_set) if self.selection_set is not None else None
                        ),
                    ),
                    # The precheck gate's columns, after the selection gate's
                    # so the column order is fixed.
                    **_gating_summary_columns(
                        precheck_outcome,
                        winner=result.winner,
                        winner_is_baseline=selected_is_baseline,
                    ),
                    # The re-draw confirmation, appended last. Empty when no
                    # promotion was attempted this iteration.
                    **_confirmation_summary_columns(confirmation),
                }
            )

            # Option 3 self-continuity: accumulate signals that get fed
            # back into the NEXT iteration's proposer prompt. All three
            # rows are computed from artifacts this iter already wrote,
            # so no extra LLM call.
            try:
                pr_response = json.loads(proposal.response.text)
            except (json.JSONDecodeError, AttributeError):
                pr_response = {}
            proposer_candidates = pr_response.get("candidates") or []
            past_diagnoses.append(
                {
                    "iteration": k,
                    "diagnosis": list(pr_response.get("diagnosis") or []),
                    "candidate_notes": [(c or {}).get("notes", "") for c in proposer_candidates],
                }
            )
            quality_trajectory.append(
                {
                    "iteration": k,
                    "score": selected_metrics.score,
                    "rubric_quality": selected_metrics.rubric_quality,
                    "rule_quality": selected_metrics.rule_quality,
                    "critic_agreement": selected_metrics.critic_agreement,
                    "review_rate": selected_metrics.review_rate,
                }
            )
            # What the next proposer's STATE_JSON reports. It cannot come
            # from `current_metrics` there: that is `_stub_metrics`, whose
            # score and agreement are zeros by design.
            winner_diag = ranking_diag.get(result.winner) or {}
            prev_ranking_state = {
                "pair_concordance": winner_diag.get("concordance"),
                "distinct_levels": winner_diag.get("distinct_levels"),
                "n_truth_pairs": len(ranking_truth_pairs),
                "n_tied_pairs": len(ranking_tied_pairs),
                "accumulated_truth_pairs": len(self._accumulated_pairs),
                "redundancy_penalty": winner_diag.get("redundancy_penalty", 0.0),
                "complexity_score": selected_metrics.complexity_score,
            }
            past_winners.append(
                {
                    "iteration": k,
                    "winner": result.winner,
                    "reason": result.reason,
                    "winner_notes": _lookup_winner_notes(result.winner, proposer_candidates),
                }
            )

            # 7. Advance.
            history.append(selected_bundle.evaluation)
            # Record every candidate (baseline + proposed) for future Pareto
            # snapshots. Meta-Harness lets its proposer read the raw frontier
            # from the filesystem; we embed a compact form in the prompt.
            all_evaluations.extend(bundle.evaluation for _, _, _, _, bundle in bundles)
            # The tool-using proposer shares the filesystem with these
            # directories and can delete the live baseline between the solver
            # pass and this reload, which would kill the run. The baseline is by
            # definition a copy of `current`, so it can be re-materialized;
            # a vanished *proposed* candidate has no other source and stays
            # fatal.
            if not Path(selected_dir).is_dir():
                if not selected_is_baseline:
                    raise FilterValidationError(
                        f"selected candidate directory vanished mid-iteration: "
                        f"{selected_dir} — proposer escape suspected"
                    )
                logger.error(
                    "baseline dir %s vanished mid-iteration (proposer escape "
                    "suspected) — re-materializing from %s",
                    selected_dir,
                    current.directory,
                )
                shutil.copytree(current.directory, selected_dir)
            current = load_filter(selected_dir)
            # What the next iteration's precheck reads a candidate's NEW
            # dead axes and redundant pairs against: the winner's own
            # deadness / redundancy over this pass.
            self._baseline_criterion_stats = (ranking_diag.get(result.winner) or {}).get(
                "criterion_stats"
            )

            # Stagnation telemetry bookkeeping for the next proposer call.
            novelty_history.append(_median_novelty(proposal_shapes))
            winner_rubric_hash = hashlib.md5(current.rubric_md.encode("utf-8")).hexdigest()
            if winner_rubric_hash == prev_rubric_hash:
                rubric_age += 1
            else:
                rubric_age = 0
                rubric_ever_changed = True
                rubric_edit_wins += 1
                prev_rubric_hash = winner_rubric_hash

            # Per-criterion contribution of the winner's rubric: how many
            # decisions flip when the rubric (whole, then each criterion) is
            # ablated and decide() re-runs on the scores already paid for.
            # Zero marginal LLM cost; feedback only — it names the criteria
            # that earn their place and exposes decision-inert rubric prose,
            # which the text-reviewing critic alone cannot see.
            # Clear before measuring, not after: the block below is wrapped
            # in a bare `except`, and these two names survive across
            # iterations. A failed measurement would otherwise leave the
            # PREVIOUS winner's line and zero-lift set standing while
            # `current` had already advanced — a stale name that, under
            # "remove it in every candidate", would direct the whole
            # iteration at an axis that may since have been renamed or
            # re-weighted, and would credit the deletion floor for removing
            # something no longer measured inert. Silence is the honest
            # degradation.
            criterion_lift_line = ""
            self._zero_lift_criteria = ()
            self._zero_lift_rules = ()
            self.proposer.zero_lift_note = ""
            try:
                winner_decisions = next(
                    d for (n, _, _, _, d) in candidate_solver if n == result.winner
                )
                from meta_filter.criterion_lift import (
                    measure_ranking as _measure_rank_lift,
                )
                from meta_filter.criterion_lift import (
                    render_ranking as _render_rank_lift,
                )

                lift_report = _measure_rank_lift(current, active_train, winner_decisions)
                criterion_lift_line = _render_rank_lift(lift_report)
                criterion_lift_line = _append_rule_pricing(
                    criterion_lift_line, current, winner_decisions, active_train
                )
                _zl_crit, _zl_rules = zero_lift_items(lift_report)
                self._zero_lift_criteria = tuple(_zl_crit)
                self._zero_lift_rules = tuple(_zl_rules)
                self.proposer.zero_lift_note = zero_lift_note(lift_report)
                # Each criterion alone against the judged pairs,
                # beside its ablation lift — the table that separates a
                # redundant axis from a noisy one.
                alone = single_criterion_concordance(
                    winner_decisions, ranking_truth_pairs, ties=ranking_tied_pairs
                )
                lift_report["single_criterion_concordance"] = alone
                criterion_signal_block = render_criterion_signal(
                    alone,
                    lift_report.get("criterion_rank_shift"),
                    (ranking_diag.get(result.winner) or {}).get("criterion_stats"),
                )
                logger.info("iter %d: %s", k, criterion_lift_line)
                # A criterion whose scores pile onto one value, or that reads a
                # metadata field its role never receives, cannot be answered.
                # Two of `initial_ranking`'s six were in that state for the
                # whole labelled lineage while the evidence sat in this very
                # file at 0.0085 and 0.0109, and in the criterion-health block
                # at 99.5% and 94.5% modal share. See meta_filter.dead_axis.
                winner_diag = ranking_diag.get(result.winner) or {}
                dead = dead_axis_check(
                    current.rubric_md,
                    getattr(current.policy_module, "RUBRIC_CRITERIA", []),
                    lift_report.get("criterion_rank_shift") or {},
                    criterion_stats=winner_diag.get("criterion_stats"),
                )
                dead_axis_log(dead, iteration=k)
                lift_path = self.store.run_dir / "criterion_lift.jsonl"
                with lift_path.open("a", encoding="utf-8") as fp:
                    record = {"iteration": k, **lift_report}
                    if dead:
                        record["dead_axis_warnings"] = dead
                    fp.write(json.dumps(record) + "\n")
            except Exception:
                logger.exception("iter %d: criterion lift failed — continuing", k)

            # Ranking telemetry for the next proposer call: the winner's
            # concordance and score-distribution shape. Diagnostics, so the
            # proposer can see saturation or collapse; never part of the score.
            if ranking_diag:
                from meta_filter.ranking import render_distribution

                wd = ranking_diag.get(result.winner)
                if wd:
                    # The floor is named, and the penalty is described as
                    # the code applies it. "Zeroed" was half the story: a
                    # degenerate candidate has its concordance forced to 0
                    # AND a further 1.0 taken off its score, so its score
                    # goes negative and no secondary term can recover it.
                    # A proposer told only "zeroed" can reasonably read the
                    # guard as a cap it might trade against.
                    ranking_block = (
                        f"winner pair concordance {wd['concordance']:.3f} over "
                        f"{len(ranking_truth_pairs)} truth pairs; "
                        f"distinct score levels {wd['distinct_levels']} "
                        f"(degeneracy floor {min_levels}: below it a "
                        f"candidate's concordance is forced to 0.000 and a "
                        f"further {DEGENERACY_PENALTY:.3f} is "
                        f"subtracted from its score)"
                        + (
                            " — THIS CANDIDATE IS DEGENERATE and paid both"
                            if wd["degenerate"]
                            else ""
                        )
                        + "\n"
                        + render_distribution(wd["distribution"])
                    )
                    from meta_filter.criterion_stats import (
                        dead_axes,
                        dead_axis_reasons,
                        render_criterion_health,
                    )

                    winner_criteria = list(
                        getattr(current.policy_module, "RUBRIC_CRITERIA", []) or []
                    )
                    # How far the run's deletion floor lets the proposer take
                    # "remove it". The floor is measured against the INITIAL
                    # filter, so a run that starts at six criteria may remove
                    # one — not two, which the precheck would then refuse.
                    # Both counts come from `measure_filter_complexity` over
                    # rubric.md, the file the floor is checked against.
                    initial_counts = getattr(self.proposer, "initial_complexity", None)
                    criteria_floor = (
                        ConstraintLimits.DEFAULT.floor_for(
                            "rubric_criteria", int(initial_counts["rubric_criteria"])
                        )
                        if initial_counts
                        else None
                    )
                    removals_left = (
                        None
                        if criteria_floor is None
                        else max(
                            0,
                            int(measure_filter_complexity(current.directory)[0]["rubric_criteria"])
                            - criteria_floor,
                        )
                    )
                    criterion_health_block = render_criterion_health(
                        wd.get("criterion_stats"),
                        threshold=self.config.redundancy_threshold,
                        # WHY each dead axis is dead — its input absent on
                        # every row, or anchors that give one level.
                        dead_reasons=dead_axis_reasons(
                            current.rubric_md,
                            winner_criteria,
                            dead_axes(wd.get("criterion_stats")),
                            active_train,
                        ),
                        # The other way an axis costs a call per row and
                        # orders nothing: declared in RUBRIC_CRITERIA and
                        # never read by decide(). Deadness is in the scores,
                        # this is in policy.py, and neither is visible from
                        # the other.
                        declared_unread=declared_unread_criteria(current.directory),
                        removals_left=removals_left,
                        criteria_floor=criteria_floor,
                    )
                    # What the judge says on the pairs the winner gets wrong,
                    # by theme, against the criteria that exist; also written
                    # to `rationale_themes.json`.
                    rationale_themes_block = self._write_rationale_themes(
                        k,
                        winner_scores=ranking_scores_by_cand.get(result.winner, {}),
                        truth_pairs=ranking_truth_pairs,
                        judge=ranking_pair_judge,
                        traj_by_id=traj_by_id,
                        rubric_md=current.rubric_md,
                        criteria=winner_criteria,
                        # The winner's rules, so a theme a RULE already asks
                        # about stops reading as uncovered.
                        rule_functions=current.rule_functions,
                    )
                # The blind-spot channel (`--pair-evidence`). Drawn from the
                # same list as the misordered-pair blocks it would be a strict
                # subset of them — prompt lines for zero information — so it
                # carries what the blocks above deliberately do not.
                # `_select_misordered_pairs` ranks consensus-audit pairs last,
                # because as RANKING evidence they are near-noise — they order
                # candidates far less like the hidden selection exam than
                # disputed pairs do. But that is a
                # statement about ranking candidates, not about what the pairs
                # reveal: every candidate ordered an audit pair the same way
                # and the judge reversed it, which is the one failure no
                # disagreement-driven sample can ever surface. So the audit
                # stratum gets its own four lines, deduplicated against the
                # blocks above, and the two channels stop paying twice for the
                # same pairs.
                pair_evidence_block = ""
                if self.config.pair_evidence > 0 and ranking_pair_judge is not None:
                    winner_scores = ranking_scores_by_cand.get(result.winner, {})
                    audit_pool, missed = _audit_blind_spot_pairs(
                        scores=winner_scores,
                        pairs=misorder_evidence_pairs,
                        provenance=self._pair_provenance,
                        exclude={
                            _misorder_pair_key(ce)
                            for ce in prev_false_negatives[:PROMPT_MISORDER_PAIRS]
                        },
                    )
                    lines = []
                    for winner_id, loser_id, gap, when in missed[: self.config.pair_evidence]:
                        why = ranking_pair_judge.rationale_for(
                            winner_id,
                            loser_id,
                            # The rationale lives under the same key as the
                            # verdict, so it needs the same label view.
                            pair_view_signature(
                                traj_by_id.get(winner_id), traj_by_id.get(loser_id)
                            ),
                        )
                        lines.append(
                            f"  * judge preferred {winner_id} over {loser_id}"
                            + (f' — "{why}"' if why else "")
                            + f"; the selected filter scored them the other way "
                            f"round (gap {gap:.1f})"
                            + (f"; audited at iter {when}" if when is not None else "")
                        )
                    if lines:
                        pair_evidence_block = (
                            f"{len(missed)} of the {len(audit_pool)} consensus-audit "
                            "pairs in this pool are mis-ordered by the selected "
                            "filter and named nowhere else in this prompt. EVERY "
                            "candidate ordered these the same way and the judge "
                            "reversed them, so no disagreement could have exposed "
                            "them; the most confidently wrong:\n" + "\n".join(lines)
                        )

            # Fixed probe pass: run the winner on the constant probe set and
            # record the drift. Monitoring only — nothing here feeds selection.
            if self.probe is not None:
                try:
                    prev_probe = probe_latest_decisions(self.store.run_dir)
                    probe_report = self.probe.evaluate(current, self.solver)
                    # Ranking-mode validation in the proper sense: the loop's
                    # OWN objective (pair concordance, adjudicated by the
                    # loop's own pair judge — no external gold anywhere) on
                    # fixed data and fixed pairs. Judged once, cached
                    # globally; every later iteration and run re-reads the
                    # same verdicts, so the curve is comparable.
                    validation_concordance = None
                    if probe_report.scores and backend is not None:
                        from meta_filter.pair_judge import PairJudge
                        from meta_filter.probe import fixed_validation_pairs
                        from meta_filter.ranking import concordance as rank_concordance

                        fixed_pairs = fixed_validation_pairs(
                            [t.trajectory_id for t in self.probe.trajectories]
                        )
                        probe_by_id = {t.trajectory_id: t for t in self.probe.trajectories}
                        judged = PairJudge(
                            client=backend.client,
                            cache_dir=self.config.pair_judge_cache_dir,
                        ).resolve_pairs_detail(
                            fixed_pairs,
                            probe_by_id,
                            max_workers=self.config.judge_parallel,
                        )
                        if judged.resolved:
                            validation_concordance = rank_concordance(
                                probe_report.scores, judged.resolved
                            )
                            logger.info(
                                "iter %d: validation concordance %.3f over %d "
                                "fixed judged pairs (%d TIE, excluded)",
                                k,
                                validation_concordance,
                                len(judged.resolved),
                                len(judged.tied),
                            )
                    probe_append_report(
                        self.store.run_dir,
                        k,
                        probe_report,
                        prev_probe,
                        validation_concordance=validation_concordance,
                    )
                    probe_trend = probe_render_for_prompt(
                        probe_load_reports(self.store.run_dir)[-5:]
                    )
                except Exception:
                    logger.exception("iter %d: probe pass failed — continuing", k)

            # Section F: stop when the loop has kept the baseline `patience`
            # times in a row, once the floor is met. Checked after the
            # iteration's artifacts are all written, so a stopped run is a
            # complete one up to `last_iteration`.
            if (
                self.config.patience is not None
                and k + 1 >= self.config.min_iterations
                and patience_counter >= self.config.patience
            ):
                logger.info(
                    "iter %d: early stop — baseline kept for %d consecutive scored "
                    "iteration(s) (patience %d, min_iterations %d)",
                    k,
                    patience_counter,
                    self.config.patience,
                    self.config.min_iterations,
                )
                stop_reason = "early_stop"
                last_iteration = k
                break

        # A proposer failure on the *last* iteration would otherwise vanish:
        # the `continue` above falls out of the loop, the run returns normally
        # and the CLI prints "run complete" with one summary row missing. The
        # streak counter is still non-zero exactly when the run ended inside
        # a failure, so it is also the test for whether the ending was clean.
        if consecutive_proposer_failures:
            last = self.config.iterations - 1
            raise RuntimeError(
                f"run ended on a failed iteration: the proposer failed for the "
                f"final {consecutive_proposer_failures} iteration(s) up to and "
                f"including {last}, so this run has "
                f"{self.config.iterations - consecutive_proposer_failures} of "
                f"{self.config.iterations} iterations and must not be recorded "
                f"as complete"
            )

        self._write_run_complete(stop_reason=stop_reason, last_iteration=last_iteration)
        return self.store.run_dir

    def _active_train_for_iteration(self, k: int) -> list[Trajectory]:
        """Return the training trajectory subset for iteration k."""

        schedule = self.config.train_sample_schedule
        rate = 1.0
        if isinstance(schedule, list) and schedule:
            rate = float(schedule[min(k, len(schedule) - 1)])
        elif isinstance(schedule, int | float):
            rate = float(schedule)
        rate = max(0.0, min(1.0, rate))
        if rate >= 1.0 or not self.train:
            return list(self.train)
        n = max(1, int(round(len(self.train) * rate)))
        # Deterministic per (seed, iter) so multiple runs with the same seed
        # sample the same subset — regression replay stays reproducible.
        rng = random.Random(self.config.seed + k * 131 + 17)
        return rng.sample(list(self.train), n)

    # --------------------------------------------------- critic helpers

    def _write_ranking_pairs(
        self,
        iteration: int,
        *,
        traj_by_id: dict[str, Trajectory],
        tied_pairs: list[tuple[str, str]],
        judge,
        scores_by_candidate: dict[str, dict[str, float]] | None,
    ) -> int:
        """Dump the judged pair set to `iterations/<NNN>/ranking_pairs.jsonl`.

        The proposer's only other views of the truth set are a count in
        `ranking.json` and the capped, evidence-rich slice in
        `critic_outputs/`; the verdicts themselves live in the pair-judge
        cache, outside the workspace the agentic proposer may read. Neither
        the prompt's pair budget nor the `critic_outputs/` cap changes — this
        is disk, for a proposer that goes looking.
        """

        from meta_filter.pair_dump import FILENAME, ranking_pair_rows, write_ranking_pairs

        rows = ranking_pair_rows(
            accumulated_pairs=self._accumulated_pairs,
            tied_pairs=tied_pairs,
            provenance=self._pair_provenance,
            traj_by_id=traj_by_id,
            judge=judge,
            scores_by_candidate=scores_by_candidate,
        )
        written = write_ranking_pairs(
            self.store.iteration_dir(iteration) / FILENAME,
            rows,
            pseudonymizer=self.pseudonymizer,
        )
        logger.info(
            "iter %d: wrote %d pair row(s) to %s%s",
            iteration,
            written,
            FILENAME,
            "" if scores_by_candidate is None else " with per-candidate results",
        )
        return written

    def _ranking_state(self, previous: dict, active_train: list[Trajectory]) -> dict:
        """The ranking numbers the proposer's STATE_JSON reports.

        Two halves, on purpose. The guard terms — the degeneracy floor and the
        redundancy weight and threshold — are config- and pool-derived, so
        they are recomputed here exactly as this iteration will apply them and
        are correct on iteration 0, before any pair exists. The measured terms
        come from the previous iteration's winner and override the defaults,
        because nothing has measured them yet when the run starts.
        """

        from meta_filter.ranking import default_min_distinct_levels

        return {
            "min_distinct_levels": (
                self.config.min_distinct_levels
                if self.config.min_distinct_levels is not None
                else default_min_distinct_levels(len(active_train))
            ),
            "redundancy_penalty_weight": self.config.redundancy_penalty,
            "redundancy_threshold": self.config.redundancy_threshold,
            **previous,
        }

    def _build_curated_evidence(
        self,
        *,
        backend,
        active_train: list[Trajectory],
        baseline_evaluation: CandidateEvaluation,
        baseline_decisions: list[FilterDecision],
        iteration: int,
    ) -> tuple[list[dict], list[str]]:
        """Assemble the 6-case evidence pack RubricCritic reads.

        Returns `(evidence_dicts, selected_ids)`. If the backend is not the
        LLM judge we still return recent FN/FP evidence so RubricCritic can
        run against baseline mistakes (just without the judge misfire label).
        """

        # CandidateEvaluation stores FN/FP as trajectory-id strings, not
        # CounterExample objects. Copy defensively so callers can mutate.
        fn_ids = list(baseline_evaluation.false_negatives)
        fp_ids = list(baseline_evaluation.false_positives)

        if isinstance(backend, TrajectoryCritic):
            selected_ids = backend.select_for_proposer(
                false_negative_ids=fn_ids,
                false_positive_ids=fp_ids,
                audit_mismatch_ids=[],
                candidate_trajectory_ids=[t.trajectory_id for t in active_train],
                seed=self.config.seed + iteration * 251 + 3,
                budget=self.config.curated_evidence_budget,
            )
        else:
            # Fallback: FN then FP up to the budget. Non-LLM backends have
            # no `select_for_proposer` and no misfire labels to key on.
            selected_ids = (fn_ids + fp_ids)[: self.config.curated_evidence_budget]

        traj_by_id = {t.trajectory_id: t for t in active_train}
        decisions_by_id = {d.trajectory_id: d for d in baseline_decisions}
        judge_verdicts = {}
        if isinstance(backend, TrajectoryCritic):
            for tid in selected_ids:
                v = backend.verdict_for(tid)
                if v is not None:
                    judge_verdicts[tid] = v

        evidence: list[dict] = []
        for tid in selected_ids:
            traj = traj_by_id.get(tid)
            if traj is None:
                continue
            filter_dec = decisions_by_id.get(tid)
            verdict = judge_verdicts.get(tid)
            evidence.append(
                {
                    "trajectory_id": tid,
                    "task": (traj.task or "")[:400],
                    "action_histogram": _tiny_histogram(traj),
                    "filter": (
                        {
                            "decision": filter_dec.decision.value,
                            "rubric_scores": filter_dec.rubric_scores,
                            "triggered_rules": filter_dec.triggered_rules,
                            "reasoning": filter_dec.reasoning_summary[:300],
                        }
                        if filter_dec is not None
                        else None
                    ),
                    "judge": (
                        {
                            "decision": (
                                verdict.decision.value
                                if verdict is not None and verdict.decision is not None
                                else None
                            ),
                            "rationale": verdict.rationale if verdict else "",
                            "misfire": verdict.misfire if verdict else "none",
                            "misfire_note": verdict.misfire_note if verdict else "",
                        }
                        if verdict is not None
                        else None
                    ),
                }
            )
        return evidence, selected_ids

    def _run_rubric_critic(
        self, rubric_md: str, curated_evidence: list[dict]
    ) -> RubricCriticReport:
        try:
            return self.rubric_critic.review(
                rubric_md=rubric_md,
                curated_evidence=curated_evidence,
            )
        except Exception as exc:
            logger.warning("RubricCritic failed, using neutral defaults: %s", exc)
            return RubricCriticReport(rubric_quality=0.5)

    def _run_rule_critic(
        self,
        *,
        loaded_filter: LoadedFilter,
        trajectories: list[Trajectory],
        truth_map: dict[str, Decision | None],
    ) -> RuleCriticReport:
        try:
            rules_source = (loaded_filter.directory / "rules.py").read_text(encoding="utf-8")
        except OSError:
            rules_source = ""
        try:
            policy_source = (loaded_filter.directory / "policy.py").read_text(encoding="utf-8")
        except OSError:
            policy_source = ""
        try:
            return self.rule_critic.review(
                rules_source=rules_source,
                policy_source=policy_source,
                rule_functions=loaded_filter.rule_functions,
                trajectories=trajectories,
                truth_map=truth_map,
                rubric_md=loaded_filter.rubric_md,
            )
        except Exception as exc:
            logger.warning("RuleCritic failed, using neutral defaults: %s", exc)
            return RuleCriticReport(rule_quality=0.5)

    def _quarantine_partial_iterations(self, start_iteration: int) -> dict[Path, Path]:
        """Move leftover iteration directories aside before writing new ones.

        Returns {old_path: new_path} for whatever was moved. Leftovers are kept
        rather than deleted — they contain paid-for LLM outputs and are the
        only record of what an aborted attempt did — but they live under
        `_aborted/` so no lineage ever shares a directory with a dead one.
        """

        iterations_dir = self.store.run_dir / "iterations"
        if not iterations_dir.is_dir():
            return {}

        stale = [
            d
            for d in sorted(iterations_dir.iterdir())
            if d.is_dir() and d.name.isdigit() and int(d.name) >= start_iteration
        ]
        if not stale:
            return {}

        aborted_root = self.store.run_dir / "_aborted"
        aborted_root.mkdir(exist_ok=True)
        # Deterministic, collision-free sequence number per aborted attempt.
        attempt = f"{len(list(aborted_root.iterdir())):03d}"
        dest_root = aborted_root / attempt
        dest_root.mkdir()

        moved: dict[Path, Path] = {}
        for d in stale:
            dest = dest_root / d.name
            shutil.move(str(d), str(dest))
            moved[d] = dest
        logger.warning(
            "quarantined %d leftover iteration dir(s) from an aborted attempt into %s: %s",
            len(moved),
            dest_root,
            ", ".join(d.name for d in stale),
        )
        return moved

    def _resolve_heldout_verdicts(self) -> dict[str, Decision | None]:
        """Ask the critic backend for a verdict on every held-out trajectory.

        Runs once per run. A verdict is a judgement about a trajectory, not
        about the filter being scored, so re-asking every iteration would buy
        nothing but LLM spend.
        """

        backend = getattr(self.validator, "critic_backend", None)
        if not self.heldout:
            return {}
        if backend is None:
            return {}
        if isinstance(backend, TrajectoryCritic):
            new_calls = backend.prefetch_parallel(
                list(self.heldout),
                max_workers=self.config.judge_parallel,
            )
            logger.info(
                "held-out adjudication: %d trajectories, %d new judge calls",
                len(self.heldout),
                new_calls,
            )
        return {t.trajectory_id: backend.critic_verdict(t) for t in self.heldout}

    def _prefetch_scorer(
        self, loaded_filter: LoadedFilter, trajectories: list[Trajectory] | None = None
    ) -> None:
        """If the solver's rubric_scorer is LLM-backed, prefetch scores for
        every training trajectory in parallel before Solver.run runs
        sequentially. This is a no-op for the heuristic scorer."""

        scorer = getattr(self.solver, "rubric_scorer", None)
        if not isinstance(scorer, LLMRubricScorer):
            return
        criteria = getattr(loaded_filter.policy_module, "RUBRIC_CRITERIA", [])
        if not criteria:
            return
        scorer.prefetch_parallel(
            list(trajectories) if trajectories is not None else list(self.train),
            criteria,
            loaded_filter.rubric_md,
        )
        if self.heldout:
            scorer.prefetch_parallel(
                list(self.heldout),
                criteria,
                loaded_filter.rubric_md,
            )

    # --------------------------------------------------------- selection

    def _rejudge_selection_set(
        self, current: LoadedFilter, prebuilt: SelectionSet, source: Path, path: Path
    ) -> None:
        """Judge a label-visible set's pairs again, on the stripped rows.

        The pair plan is kept — resolved, tied and unanswered pairs alike, so
        the exam asks the same questions — and only the verdicts are bought
        fresh. They cannot come from the cache: the stripped rows' label view
        is part of the pair judge's key (`view_policy.pair_view_signature`),
        so every verdict formed with the labels in view is a different file.
        """

        backend = getattr(self.validator, "critic_backend", None)
        if backend is None:
            raise RuntimeError("re-judging the selection set needs a judge backend")
        from meta_filter.pair_judge import PairJudge

        planned = [*prebuilt.pairs, *prebuilt.tied, *prebuilt.unresolved]
        logger.info(
            "metadata-free: re-judging the %d planned pairs of %s on stripped rows",
            len(planned),
            source,
        )
        judge = PairJudge(client=backend.client, cache_dir=self.config.pair_judge_cache_dir)
        rejudged = judge_selection_pairs(
            planned,
            {t.trajectory_id: t for t in self.selection_rows},
            judge,
            max_workers=self.config.judge_parallel,
            provenance={
                **prebuilt.provenance,
                "metadata_free": True,
                "rejudged_from": str(source),
                "selection_data": str(self.config.selection_data),
            },
        )
        if not rejudged.pairs:
            raise RuntimeError(f"re-judging {source}: the judge resolved 0 of {len(planned)} pairs")
        path.parent.mkdir(parents=True, exist_ok=True)
        rejudged.save(path)
        self.selection_set = rejudged
        initial_scores = self._score_on_selection_rows(current)
        self._baseline_selection_scores = initial_scores
        self._save_baseline_selection_scores(current, iteration=None)

    def _metadata_free_record_path(self) -> Path:
        """Where the source pools of a metadata-free run are recorded.

        Beside `runs/_selection/`, outside the run directory, so a tool-using
        proposer confined to its workspace is never shown where the raw,
        labelled rows live.
        """

        return self.runs_dir / "_metadata_free" / f"{self.run_id}.json"

    def _write_metadata_free_sources(self) -> None:
        sources = {
            "train": self.train_path,
            "validation": self.heldout_path,
            "probe": Path(self._probe_path) if self._probe_path else None,
            "selection": Path(self.config.selection_data) if self.config.selection_data else None,
        }
        record = {
            "run_id": self.run_id,
            "sources": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in sources.items()
                if path is not None and Path(path).is_file()
            },
            "pseudonym_salt": PSEUDONYM_SALT,
        }
        path = self._metadata_free_record_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        logger.info("metadata-free: source pools recorded outside the workspace in %s", path)

    def _write_metadata_free_train(self) -> Path:
        """Write the stripped train rows to `<run>/train_pool.metadata_free.jsonl`.

        Written from `self.train`, which `load_trajectories` already stripped,
        so the file is the rows every role in this run sees and nothing else.
        """

        path = self.store.run_dir / METADATA_FREE_TRAIN_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            for traj in self.train:
                fp.write(json.dumps(traj.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return path

    def _ensure_selection_set(self, current: LoadedFilter) -> None:
        """Load or build `runs/_selection/<run_id>.json` before iteration 0.

        A resumed run reloads the file so it is graded on the pairs it
        started with. A fresh run scores the initial filter on the selection
        rows first, because the pair plan stratifies on its score gaps, then
        judges the planned pairs once in both orders through the same
        `PairJudge` (and cache) the train side uses. The initial scores double
        as the baseline's selection scores for iteration 0.

        Raises:
            RuntimeError: No judge backend to build the set with, or the
                judge resolved no pair at all — a gate that can never accept
                is a misconfiguration, not a run.
        """

        if not self.selection_rows:
            return
        path = selection_set_path(self.store.run_dir)
        if self.config.selection_set is not None:
            self._adopt_prebuilt_selection_set(current, path)
            return
        if path.is_file():
            self.selection_set = SelectionSet.load(path)
            logger.info(
                "selection set reloaded from %s: %d rows, %d pairs (%d TIE, %d unresolved)",
                path,
                len(self.selection_set.row_ids),
                len(self.selection_set),
                len(self.selection_set.tied),
                len(self.selection_set.unresolved),
            )
            self._baseline_selection_scores = self._load_baseline_selection_scores(current)
            return

        backend = getattr(self.validator, "critic_backend", None)
        if backend is None:
            raise RuntimeError(
                "selection_data was given but there is no judge backend to "
                "adjudicate the selection pairs with"
            )
        from meta_filter.pair_judge import PairJudge

        initial_scores = self._score_on_selection_rows(current)
        ids = [t.trajectory_id for t in self.selection_rows]
        plan = plan_selection_pairs(
            ids,
            n_pairs=self.config.selection_pairs,
            seed=self.config.selection_seed,
            initial_scores=initial_scores,
        )
        logger.info(
            "selection set: %d rows, %d pairs planned (%s: %d hard, %d uniform), judging",
            len(ids),
            len(plan.pairs),
            "every pair"
            if plan.pairs_mode == "all"
            else ("stratified" if plan.stratified else "uniform"),
            plan.n_hard,
            plan.n_uniform,
        )
        judge = PairJudge(client=backend.client, cache_dir=self.config.pair_judge_cache_dir)
        selection_set = judge_selection_pairs(
            plan.pairs,
            {t.trajectory_id: t for t in self.selection_rows},
            judge,
            max_workers=self.config.judge_parallel,
            provenance={
                **plan.provenance(),
                "selection_data": str(self.config.selection_data),
                "initial_filter": str(self.initial_filter_dir),
                "metadata_free": True,
            },
        )
        if not selection_set.pairs:
            raise RuntimeError(
                f"selection set: the judge resolved 0 of {len(plan.pairs)} planned "
                f"pairs ({len(selection_set.tied)} TIE, {len(selection_set.unresolved)} "
                "unanswered) — the acceptance gate could never fire"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        selection_set.save(path)
        self.selection_set = selection_set
        self._baseline_selection_scores = initial_scores
        self._save_baseline_selection_scores(current, iteration=None)
        logger.info(
            "selection set saved to %s: %d rows, %d judged pairs, %d TIE, %d unresolved; "
            "initial filter selection concordance %.4f",
            path,
            len(selection_set.row_ids),
            len(selection_set),
            len(selection_set.tied),
            len(selection_set.unresolved),
            selection_set.concordance(initial_scores),
        )

    def _adopt_prebuilt_selection_set(self, current: LoadedFilter, path: Path) -> None:
        """Install `config.selection_set` as this run's exam instead of building one.

        The set was judged once, outside any run, so every run that points
        at the same file is graded on the same pairs — the property the
        per-run plan could not give (see `OptimizerConfig.selection_pairs`).
        The file is copied byte-for-byte to `runs/_selection/<run_id>.json`,
        where a resume reloads it exactly as it reloads a planned set; a run
        directory that already holds a set must hold the SAME pairs, because
        a `--resume` that swapped the exam under the run would leave its
        earlier iterations graded on one set and its later ones on another.

        The initial filter is still scored on the selection rows: those
        scores are the iteration-0 baseline. They are just not used to plan.

        Raises:
            ValueError: The set names a row that is not in `selection_data`.
            RuntimeError: The run already has a different selection set, or
                the prebuilt set has no resolved pair.
        """

        source = Path(self.config.selection_set or "")
        # The rows were pseudonymized at load; bring the set into the same
        # namespace (a no-op for a set built on stripped or instance-free rows).
        prebuilt = pseudonymize_selection_set(SelectionSet.load(source))
        available = {t.trajectory_id for t in self.selection_rows}
        missing = sorted((set(prebuilt.row_ids) | prebuilt.pair_ids()) - available)
        if missing:
            raise ValueError(
                f"selection set {source} names {len(missing)} row(s) not in "
                f"{self.config.selection_data}: {', '.join(missing[:10])}"
            )
        if not prebuilt.pairs:
            raise RuntimeError(
                f"selection set {source} has no resolved pair — the gate could never fire"
            )
        why = selection_set_view(prebuilt.provenance)
        if why is not None:
            existing = SelectionSet.load(path) if path.is_file() else None
            if existing is not None and existing.provenance.get("metadata_free") is True:
                # A resume of a run that already re-judged: its exam is on disk.
                self.selection_set = existing
                self._baseline_selection_scores = self._load_baseline_selection_scores(current)
                logger.info(
                    "selection set reloaded from %s (re-judged on stripped rows from %s)",
                    path,
                    source,
                )
                return
            if not self.config.rejudge_selection:
                raise RuntimeError(
                    f"refusing selection set {source}: {why}. Build one with "
                    "`scripts.build_selection_set` (see configs/metadata_free/README.md) "
                    "or pass --rejudge-selection to re-judge its pairs at run start"
                )
            self._rejudge_selection_set(current, prebuilt, source, path)
            return
        if path.is_file():
            existing = SelectionSet.load(path)
            if existing.pair_signature() != prebuilt.pair_signature():
                raise RuntimeError(
                    f"{path} already holds a selection set with different pairs than "
                    f"{source} ({len(existing)} vs {len(prebuilt)} resolved) — a resumed "
                    "run must keep the exam it started on"
                )
            self.selection_set = existing
            logger.info(
                "selection set reloaded from %s (matches prebuilt %s): %d rows, %d pairs "
                "(%d TIE, %d unresolved)",
                path,
                source,
                len(existing.row_ids),
                len(existing),
                len(existing.tied),
                len(existing.unresolved),
            )
            self._baseline_selection_scores = self._load_baseline_selection_scores(current)
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        # Saved in the pseudonymized namespace the run reads, so a resume
        # compares like with like.
        prebuilt.save(path)
        self.selection_set = prebuilt
        initial_scores = self._score_on_selection_rows(current)
        self._baseline_selection_scores = initial_scores
        self._save_baseline_selection_scores(current, iteration=None)
        logger.info(
            "prebuilt selection set %s copied to %s: %d rows, %d judged pairs, %d TIE, "
            "%d unresolved (judge %s, %s); initial filter selection concordance %.4f",
            source,
            path,
            len(prebuilt.row_ids),
            len(prebuilt),
            len(prebuilt.tied),
            len(prebuilt.unresolved),
            prebuilt.provenance.get("judge_model", "?"),
            prebuilt.provenance.get("pairs_mode", "?"),
            prebuilt.concordance(initial_scores),
        )

    def _log_gate_thresholds(self) -> None:
        """Log and persist what `b` the paired gate needs at each `b + c`.

        Once at run start, for the family threshold and for the
        strong-evidence one, as `paired_test.render_critical_b_table` blocks
        in the log and as `runs/_selection/<run_id>.gate.json` beside the
        selection set — so a reader of a kept-baseline line ("b=18 c=9,
        needed b ≥ 20") can see the whole curve without recomputing the
        binomial. The family threshold is derived here from the configured
        candidate count; an iteration with fewer valid challengers is held to
        a slightly lower one (`paired_test.z_threshold_for`), which
        `selection.json` records per candidate.
        """

        n_candidates = max(1, self.config.candidates_scored or self.config.candidates_per_iteration)
        family_z = (
            self.config.gate_z if self.config.gate_z is not None else z_threshold_for(n_candidates)
        )
        strong_z = family_z + STRONG_EXTRA_Z
        record: dict[str, object] = {
            "n_candidates": n_candidates,
            "gate_z_pinned": self.config.gate_z is not None,
            "min_discordant": self.config.gate_min_discordant,
            "strong_extra_z": STRONG_EXTRA_Z,
        }
        for name, z in (("family", family_z), ("strong_evidence", strong_z)):
            alpha = one_sided_p_from_z(z)
            label = f"{name} z={z:.2f}, {n_candidates} candidates"
            logger.info("paired gate\n%s", render_critical_b_table(alpha, label))
            record[name] = {
                "z": z,
                "alpha": alpha,
                "critical_b": [[n, b] for n, b in critical_b_table(alpha)],
            }
        path = selection_set_path(self.store.run_dir).with_suffix(".gate.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def _score_on_selection_rows(self, loaded_filter: LoadedFilter) -> dict[str, float]:
        """One filter's scores on the selection rows, cells prefetched in parallel.

        Deliberately not `_prefetch_scorer`: that helper also warms the
        held-out rows, which the train pass has already done for this
        candidate. Cost lands on the solver role through the scorer's own
        metered client.
        """

        scorer = getattr(self.solver, "rubric_scorer", None)
        criteria = getattr(loaded_filter.policy_module, "RUBRIC_CRITERIA", [])
        if isinstance(scorer, LLMRubricScorer) and criteria:
            scorer.prefetch_parallel(list(self.selection_rows), criteria, loaded_filter.rubric_md)
        decisions = self.solver.run(loaded_filter, self.selection_rows)
        self._selection_cells[str(loaded_filter.directory)] = {
            d.trajectory_id: dict(d.rubric_scores) for d in decisions if not d.scorer_failed
        }
        return scores_from_decisions(decisions)

    def _score_candidates_on_selection(
        self, candidates: list[tuple[str, LoadedFilter]], *, workers: int
    ) -> dict[str, dict[str, float]]:
        """Every scored candidate on the selection rows, `workers` at a time.

        A candidate whose code raises on a selection row it never saw on
        train gets an empty score map: `pair_correctness` then charges it
        every pair, which is the same "broken candidate, not broken run"
        outcome the train pass gives a candidate that raises there.
        """

        def _one(item: tuple[str, LoadedFilter]) -> tuple[str, dict[str, float]]:
            cand_id, loaded = item
            try:
                return cand_id, self._score_on_selection_rows(loaded)
            except Exception:
                logger.exception(
                    "%s: candidate code raised on the selection rows — scored as "
                    "empty, so it loses every selection pair",
                    cand_id,
                )
                return cand_id, {}

        if not candidates:
            return {}
        if workers > 1 and len(candidates) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as pool:
                return dict(pool.map(_one, candidates))
        return dict(_one(item) for item in candidates)

    def _promotion_rescore(
        self,
        iteration: int,
        winner_id: str,
        loaded_filter: LoadedFilter,
        *,
        winning_scores: dict[str, float],
    ) -> tuple[dict[str, float] | None, float | None]:
        """Re-score the accepted candidate fresh, making that draw canonical.

        Selection took the best of several noisy measurements, so the draw
        that won is biased upward: a champion re-scored on the same pairs
        reads below its winning number. Carrying the winning draw as
        the baseline's number would make every later challenger beat a
        ceiling that was partly luck.

        The fresh draw OVERWRITES the cached cells (`LLMRubricScorer.fresh_draw`)
        rather than being filed under a side key. Under a side key the
        baseline would read fresh cells while a next-iteration candidate with
        a byte-identical rubric.md read the lucky ones from the plain key —
        so `b − c` for a rules-only or policy-only edit would be scorer noise
        (about a fifth of cells move on an identical re-score), and a resumed
        run's unsalted re-score would restore the lucky draw as the baseline.
        With one canonical value per cell, the next iteration's candidates,
        the carried baseline scores and a resumed baseline all read the same
        numbers, and a rubric-identical candidate gets exactly `b = c = 0`.
        The heuristic scorer is deterministic and the delta is then zero.

        Cells the fresh pass could not score fall back to the winning draw
        for that row rather than being charged forever: a router's bad
        minute is not a property of the filter. A re-score that fails
        outright (transport error) returns no scores and no delta rather
        than raising: the caller treats an unconfirmable acceptance as a
        rejection, and the iteration goes on.

        This method measures; it does not commit. The caller adopts the
        fresh scores as the baseline's (`_adopt_baseline_selection_scores`)
        only once `confirm_promotion` has agreed with draw 1.

        Returns:
            ``(fresh_scores, fresh − winning)`` — the fresh draw on the
            selection rows and the concordance delta, also logged; ``(None,
            None)`` when the re-score failed.
        """

        assert self.selection_set is not None
        scorer = getattr(self.solver, "rubric_scorer", None)
        winning_concordance = self.selection_set.concordance(winning_scores)
        try:
            with _fresh_scorer_draw(scorer):
                fresh = self._score_on_selection_rows(loaded_filter)
        except Exception:
            logger.exception(
                "iter %d: promotion re-score of %s failed — the acceptance cannot be "
                "confirmed on a fresh draw (winning concordance %.4f); "
                "promoted_rescore_delta not measured",
                iteration,
                winner_id,
                winning_concordance,
            )
            return None, None
        missing = [tid for tid in winning_scores if tid not in fresh]
        if missing:
            logger.warning(
                "iter %d: promotion re-score of %s left %d row(s) unscored; "
                "keeping the winning draw for those rows",
                iteration,
                winner_id,
                len(missing),
            )
            fresh = {**{tid: winning_scores[tid] for tid in missing}, **fresh}
        fresh_concordance = self.selection_set.concordance(fresh)
        delta = fresh_concordance - winning_concordance
        logger.info(
            "iter %d: accepted %s re-scored on the selection rows — winning %.4f, "
            "fresh %.4f, promoted_rescore_delta %+.4f",
            iteration,
            winner_id,
            winning_concordance,
            fresh_concordance,
            delta,
        )
        return fresh, delta

    def _adopt_baseline_selection_scores(
        self, scores: dict[str, float], loaded_filter: LoadedFilter, *, iteration: int
    ) -> None:
        """Make `scores` the baseline's selection scores and persist them.

        Called on a confirmed promotion with the fresh draw, which is by then
        also what the scorer cache holds for every cell — so the carried
        baseline, a resumed run and next iteration's rubric-identical
        candidates all read one number per cell.
        """

        self._baseline_selection_scores = dict(scores)
        self._save_baseline_selection_scores(loaded_filter, iteration=iteration)

    def _carry_baseline_selection_scores(
        self, winning_scores: Mapping[str, float], loaded_filter: LoadedFilter, *, iteration: int
    ) -> None:
        """Adopt a no-re-draw winner's own draw-1 scores as the baseline's.

        Why no re-score at all for this promotion. `_promotion_rescore`
        exists to strip the winner's curse: selection took the best of
        several noisy draws, so the winning number is biased upward and
        carrying it would make later challengers beat a partly-lucky
        ceiling. A `b = c = 0` simplification was not picked for
        out-drawing anything — it tied the baseline on every pair, and won
        on size (`choose_paired` orders simplifications by complexity). There
        is no maximisation to undo, so the one thing the re-score buys is
        not on offer here, and it costs a full scoring pass of the selection
        rows. Skipping it also leaves the scorer cache alone: a re-score
        OVERWRITES every cell it touches, which is why a rejected re-draw
        has to re-read the baseline (`_reread_baseline_selection_scores`) to
        put the carried scores and the cache back in agreement. Nothing is
        overwritten here, so they never stop agreeing.

        Why the winner's own draw-1 scores and not the outgoing baseline's,
        which are pair-for-pair equivalent by construction. The invariant the
        next iteration depends on is that the carried baseline scores equal
        what `_score_on_selection_rows` returns for the baseline FILTER — that
        is what gives a rubric-identical candidate ``b = c = 0`` instead of
        noise. From here the baseline filter is the winner, and these are
        exactly its cache reads, so the invariant holds by construction. The
        outgoing baseline's numbers would only be safe under the stronger
        claim that the two filters score identically, and `b = c = 0` is the
        weaker claim that they order identically — equal credit on every
        pair, which two different score vectors can produce.

        `promoted_rescore_delta` is left unset: no second draw was taken, so
        there is no delta to report, and an empty cell says that honestly.
        """

        if not winning_scores:
            logger.warning(
                "iter %d: the confirmed simplification has no stored selection scores; "
                "keeping the outgoing baseline's, which order every pair the same way",
                iteration,
            )
            self._save_baseline_selection_scores(loaded_filter, iteration=iteration)
            return
        self._baseline_selection_scores = dict(winning_scores)
        self._save_baseline_selection_scores(loaded_filter, iteration=iteration)

    def _reread_baseline_selection_scores(self, iteration: int, baseline: LoadedFilter) -> None:
        """After a rejected re-draw, re-read the kept baseline from the cache.

        The rejected candidate's fresh draw overwrote every cell it shares
        with the baseline (a policy- or rules-only edit shares all of them),
        so the scores the baseline carries may no longer be what the cache
        holds. Reading the baseline back from the cache costs no LLM call —
        every cell is a hit, its own or the fresh one — and restores the
        invariant a rubric-identical candidate next iteration depends on:
        `b = c = 0`, not scorer noise between two draws. The rejected
        candidate's scores are adopted nowhere; the baseline's identity and
        its filter are unchanged. A failure here keeps the carried scores.
        """

        try:
            reread = self._score_on_selection_rows(baseline)
        except Exception:
            logger.exception(
                "iter %d: could not re-read the baseline's selection scores after the "
                "rejected re-draw; keeping the carried scores",
                iteration,
            )
            return
        self._baseline_selection_scores = reread
        self._save_baseline_selection_scores(baseline, iteration=iteration)

    def _assert_no_selection_exposure(self, text: str, *, iteration: int, leg: str) -> None:
        """Raise if a saved proposer prompt names a selection row.

        A raise, not a warning: the gate's whole value is that the proposer
        never saw the exam, and one leaked id means the run's acceptances
        can no longer be told from memorization. The ids are raw (selection
        rows are outside the pseudonymizer's pool), which is what makes the
        substring check sufficient.
        """

        if not self.selection_rows:
            return
        leaked = selection_row_ids_in_text(text, (t.trajectory_id for t in self.selection_rows))
        if leaked:
            raise RuntimeError(
                f"iteration {iteration} ({leg} leg): the proposer prompt names "
                f"{len(leaked)} selection row id(s) — {', '.join(leaked[:5])} — the "
                "hidden selection set is exposed and the run is invalid"
            )

    def _forbidden_read_roots(self) -> list[Path]:
        """Directories a tool-using proposer may never read.

        `runs/_selection/` holds the judge's order on every selection pair and
        the baseline's scores on the hidden rows; the pair-judge cache holds
        every train verdict ever bought, i.e. the exam of every iteration.
        Neither is in the proposer's workspace, and a read of either is a leak
        whatever the mode. The scorer cache is handled separately
        (`_scorer_cache_root`): it holds the hidden rows' cells AND the train
        rows' cells, and only the former are a leak.
        """

        roots = [
            selection_set_path(self.store.run_dir).parent,
            Path(self.config.pair_judge_cache_dir),
        ]
        return [Path(r).resolve() for r in roots]

    def _count_invalidated(
        self, iteration: int, proposal: object, exc: MetadataFreeReadError, failures: int
    ) -> bool:
        """Invalidate the proposal; count it as a proposer failure (always True).

        A proposer that keeps reading outside its workspace is a broken
        configuration, so the same consecutive-failure limit ends the run.
        """

        self._invalidate_proposal(iteration, proposal, exc)
        if failures + 1 >= MAX_CONSECUTIVE_PROPOSER_FAILURES:
            raise RuntimeError(
                f"the proposer read outside its workspace {failures + 1} times in a row; "
                "a metadata-free run cannot continue with it"
            ) from exc
        return True

    def _metadata_free_read_roots(self) -> list[Path]:
        """What a metadata-free proposer may read outside its workspace: the
        Python installation its tools run on, and nothing else."""

        roots = {Path(sys.prefix), Path(sys.base_prefix), Path(sys.executable).parent.parent}
        executable = getattr(self.score_tool, "executable", None)
        if executable:
            roots.add(Path(executable).parent.parent)
        return [r.resolve() for r in roots]

    def _invalidate_proposal(
        self, iteration: int, proposal: object, exc: MetadataFreeReadError
    ) -> None:
        """Take a proposal that read outside its workspace out of the run.

        Its candidate directories move to `runs/_invalidated/<run_id>/` —
        outside the workspace, so no later proposer copies from them — and the
        paths it read are recorded there and in the log. The workspace keeps a
        one-line note with the count only: the paths name where raw rows live.
        """

        dest = self.runs_dir / "_invalidated" / self.run_id / f"{iteration:03d}"
        dest.mkdir(parents=True, exist_ok=True)
        for cand in getattr(proposal, "candidate_dirs", []) or []:
            cand = Path(cand)
            if cand.is_dir():
                shutil.move(str(cand), str(dest / cand.name))
        (dest / "reads.json").write_text(
            json.dumps({"leg": exc.leg, "reads": list(exc.reads)}, indent=2) + "\n",
            encoding="utf-8",
        )
        note = self.store.run_dir / "iterations" / f"{iteration:03d}" / "proposer"
        note.mkdir(parents=True, exist_ok=True)
        (note / "invalidated.json").write_text(
            json.dumps(
                {
                    "invalidated": True,
                    "reason": "read outside the workspace in a metadata-free run",
                    "n_reads": len(exc.reads),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.error("iter %d: proposal invalidated — %s (first: %s)", iteration, exc, exc.reads[0])

    def _scorer_cache_root(self) -> Path | None:
        scorer_cache = getattr(getattr(self.solver, "rubric_scorer", None), "cache_dir", None)
        return Path(scorer_cache).resolve() if scorer_cache else None

    def _scorer_cache_read_is_forbidden(self, path: Path, root: Path) -> bool:
        """Whether one read under the scorer cache reaches a hidden row's cell.

        Per-criterion cells are named `<trajectory id>__crit_...json`, so the
        row a read touches is known from the path alone. A TRAIN row's cell is
        exempt: its score is already the proposer's evidence — every
        `solver_outputs/<candidate>.json` lists the same per-criterion scores
        for the same rows — and the criterion-scoring tool reads exactly
        these cells on the proposer's behalf, so forbidding them would abort a
        run for reading what the prompt hands over. A cell of any other row
        (the hidden selection rows, the probe) is the exam and stays
        forbidden, and so does anything under the root whose row cannot be
        read off the name — the directory itself, whose listing is every id
        the loop has ever scored, included.
        """

        if path == root:
            return True
        leading = cache_key_trajectory_id(path.name)
        if leading is None:
            return True
        train_keys = {sanitize_trajectory_id(t.trajectory_id) for t in self.train}
        return leading not in train_keys

    def _assert_no_forbidden_reads(self, response: object, *, iteration: int, leg: str) -> None:
        """Raise if the agent's audited reads reached an answer key.

        The Codex harness cannot block reads — `-C` anchors paths and the
        sandbox stops writes only — so `CodexCliClient` records every read
        outside the workspace on the response (`reads_outside_workspace`). A
        read under
        `runs/_selection/`, the pair-judge cache or the scorer cache is the
        same failure the prompt grep catches: the proposer saw the exam, and
        the run's acceptances can no longer be told from memorization. Every
        other outside read is logged at WARNING with the leg it came from.
        Responses without the field (the Claude Code harness enforces the
        boundary itself; the static proposer reads nothing) pass through.
        The scorer cache is judged cell by cell rather than as a root: a
        train row's cell is exempt, a hidden row's is not
        (`_scorer_cache_read_is_forbidden`). The tool's interpreter in the
        venv is under none of the roots and is logged as a harmless read.
        """

        reads = tuple(getattr(response, "reads_outside_workspace", ()) or ())
        if not reads:
            return
        roots = self._forbidden_read_roots()
        cache_root = self._scorer_cache_root()
        forbidden: list[str] = []
        harmless: list[str] = []
        for raw in reads:
            path = Path(str(raw)).resolve()
            if any(path == root or root in path.parents for root in roots):
                forbidden.append(str(raw))
            elif cache_root is not None and (path == cache_root or cache_root in path.parents):
                # Id-aware: a train row's cell is evidence, a hidden row's is
                # the exam. See `_scorer_cache_read_is_forbidden`.
                if self._scorer_cache_read_is_forbidden(path, cache_root):
                    forbidden.append(str(raw))
                else:
                    harmless.append(str(raw))
            else:
                harmless.append(str(raw))
        # Nothing outside the workspace is safe to read — the raw pools, other
        # runs and the caches all hold rows as they are on disk — except the
        # interpreter the proposer's tools run on.
        allowed = self._metadata_free_read_roots()
        outside_mode = [
            raw
            for raw in harmless
            if not any(
                (p := Path(str(raw)).resolve()) == root or root in p.parents for root in allowed
            )
        ]
        harmless = [raw for raw in harmless if raw not in outside_mode]
        if harmless:
            logger.warning(
                "iteration %d (%s leg): the proposer read %d path(s) outside its "
                "workspace; first: %s",
                iteration,
                leg,
                len(harmless),
                harmless[0],
            )
        if forbidden:
            raise RuntimeError(
                f"iteration {iteration} ({leg} leg): the proposer read "
                f"{len(forbidden)} path(s) under a selection or cache directory — "
                f"{', '.join(forbidden[:5])} — the hidden selection set is exposed "
                "and the run is invalid"
            )
        if outside_mode:
            raise MetadataFreeReadError(iteration, leg, outside_mode)

    @staticmethod
    def _filter_fingerprint(loaded_filter: LoadedFilter) -> str:
        """Hash of the three filter files, to tie saved scores to a filter."""

        digest = hashlib.md5()
        for name in ("rubric.md", "rules.py", "policy.py"):
            path = Path(loaded_filter.directory) / name
            digest.update(path.read_bytes() if path.is_file() else b"")
            digest.update(b"\0")
        return digest.hexdigest()

    def _save_baseline_selection_scores(
        self, loaded_filter: LoadedFilter, *, iteration: int | None
    ) -> None:
        """Persist `_baseline_selection_scores` beside the selection set.

        Written at set build (the initial filter) and at every promotion, so
        the file always describes the filter a resume would continue from.
        The fingerprint lets `_load_baseline_selection_scores` refuse a file
        that describes some other filter. Never fatal: a failed write costs
        a resume one re-score, which is now canonical anyway.
        """

        if self._baseline_selection_scores is None:
            return
        path = baseline_scores_path(self.store.run_dir)
        payload = {
            "run_id": self.run_id,
            "iteration": iteration,
            "saved_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "filter_fingerprint": self._filter_fingerprint(loaded_filter),
            "scores": dict(sorted(self._baseline_selection_scores.items())),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError:
            logger.exception("could not write the baseline selection scores to %s", path)

    def _load_baseline_selection_scores(self, current: LoadedFilter) -> dict[str, float] | None:
        """The persisted baseline scores, if they describe `current`.

        None (so the caller re-scores, from the now-canonical cache) when
        the file is missing, unreadable, or fingerprints a different filter.
        """

        path = baseline_scores_path(self.store.run_dir)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scores = {str(k): float(v) for k, v in payload["scores"].items()}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            logger.warning("unreadable baseline selection scores at %s; re-scoring", path)
            return None
        fingerprint = payload.get("filter_fingerprint")
        if fingerprint and fingerprint != self._filter_fingerprint(current):
            logger.warning(
                "baseline selection scores at %s describe another filter; re-scoring", path
            )
            return None
        logger.info(
            "baseline selection scores reloaded from %s (%d rows, saved at iteration %s)",
            path,
            len(scores),
            payload.get("iteration"),
        )
        return scores

    def _reset_score_budget(self, iteration: int, leg: str) -> None:
        """Open a fresh leg in `<run>/proposer_score_budget.json`.

        The tool reads the cap from this file, so the reset is what makes
        the budget per leg rather than per run. No-op when the tool is not
        advertised: the file then never exists.
        """

        if not self.score_tool_enabled:
            return
        ledger = ScoreBudgetLedger.load(
            self.store.run_dir, default_cap=self.config.proposer_score_budget
        )
        ledger.reset(iteration=iteration, leg=leg, cap_usd=self.config.proposer_score_budget)
        ledger.save()

    def _fold_score_budget(self, iteration: int, leg: str) -> None:
        """Bill the leg's tool calls to the proposer role and merge its cells.

        The tool runs in the agent's subprocess and cannot reach this
        process's cost ledger, so it records spend in the file and the
        optimizer moves it across here — under the proposer role, because
        the calls were made on the proposer's behalf, and `cost_solver` must
        keep meaning the loop's own scoring.

        The cells the tool bought are merged from `<run>/proposer_score_cache/`
        into the run cache here too, and this is the ONLY path by which
        anything written inside the workspace reaches the cache: train-row
        cells only, never over a cell the loop already holds
        (`score_tool.merge_staged_cells`). The counts are logged so a leg
        that staged a selection-row cell — which nothing legitimate does —
        is visible.
        """

        if not self.score_tool_enabled:
            return
        self._merge_staged_cells(iteration, leg)
        ledger = ScoreBudgetLedger.load(
            self.store.run_dir, default_cap=self.config.proposer_score_budget
        )
        if ledger.leg != leg or ledger.iteration != iteration or not ledger.calls:
            return
        COST_LEDGER.add(
            "proposer",
            SimpleNamespace(
                cost_usd=ledger.spent_usd,
                cost_source="score_tool",
                total_tokens=ledger.tokens,
                calls=ledger.calls,
            ),
        )
        logger.info(
            "iter %d (%s leg): the proposer scored criteria %d time(s) with the tool — "
            "$%.4f of its $%.2f budget, billed to cost_proposer",
            iteration,
            leg,
            ledger.calls,
            ledger.spent_usd,
            ledger.cap_usd,
        )

    def _merge_staged_cells(self, iteration: int, leg: str) -> None:
        scorer = getattr(self.solver, "rubric_scorer", None)
        staging = staging_dir_for(self.store.run_dir)
        if not isinstance(scorer, LLMRubricScorer) or not staging.is_dir():
            return
        counts = merge_staged_cells(
            staging, Path(scorer.cache_dir), (t.trajectory_id for t in self.train)
        )
        if not any(counts.values()):
            return
        rejected = sum(v for k, v in counts.items() if k != "merged")
        logger.log(
            logging.WARNING if counts["rejected_by_id"] else logging.INFO,
            "iter %d (%s leg): merged %d staged scorer cell(s) from the proposer's "
            "scoring tool into the run cache; rejected %d (%d not a train row, "
            "%d already in the cache, %d unreadable)",
            iteration,
            leg,
            counts["merged"],
            rejected,
            counts["rejected_by_id"],
            counts["rejected_existing"],
            counts["rejected_unreadable"],
        )

    def _rubric_health_on_full_pass(
        self,
        iteration: int,
        ranking_diag: dict[str, dict],
        baseline_id: str,
        outcome: PrecheckOutcome,
    ) -> set[str]:
        """The strong-evidence set the gate is handed.

        The precheck measured every candidate on the judged rows; the full
        pass has just measured them on the whole train pool, and the pool is
        what a NEW dead axis or a NEW redundant pair should be read on (the
        precheck's rows may be too few for `NEW_DEAD_MIN_ROWS`). Firings from
        either measurement count, and both readouts are recorded in
        `ranking.json` per candidate.

        The simplification set needs no such re-confirmation: it is a fact
        about file sizes, which the full pass cannot change.
        """

        strong = set(outcome.strong_evidence())
        baseline_stats = (ranking_diag.get(baseline_id) or {}).get("criterion_stats")
        for cid, diag in ranking_diag.items():
            if cid == baseline_id:
                continue
            new_dead, new_pairs = rubric_defects(diag.get("criterion_stats"), baseline_stats)
            diag["rubric_defects"] = {"new_dead_axes": new_dead, "new_redundant_pairs": new_pairs}
            if new_dead or new_pairs:
                strong.add(cid)
                logger.info(
                    "iter %d: %s introduces rubric defects on the full pass (dead: %s; "
                    "redundant: %s) — needs strong evidence",
                    iteration,
                    cid,
                    ", ".join(new_dead) or "none",
                    "; ".join(" vs ".join(p) for p in new_pairs) or "none",
                )
        for cid in outcome.simplification():
            if cid in ranking_diag:
                ranking_diag[cid]["simplification"] = True
        return strong

    def _write_rationale_themes(
        self,
        iteration: int,
        *,
        winner_scores: dict[str, float],
        truth_pairs: list[tuple[str, str]],
        judge,
        traj_by_id: dict[str, Trajectory],
        rubric_md: str,
        criteria: list[str],
        rule_functions: list | None = None,
    ) -> str:
        """`iterations/<k>/rationale_themes.json` and its prompt block.

        Reads the judge's cached rationales — both presentation orders when
        both exist — for every accumulated TRAIN truth pair the winner
        orders backwards, groups them by theme (`rationale_themes`) and says
        which current criterion, if any, asks about each theme. Train pairs
        only, read-only against the judge cache; the selection set is never
        touched. Empty without a judge or below `MIN_MISORDERED_PAIRS`.
        """

        from meta_filter.ranking import missed_pairs
        from meta_filter.rationale_themes import rationale_themes, render_rationale_themes

        if judge is None or not truth_pairs:
            return ""
        rationales: dict[tuple[str, str], list[str]] = {}
        for hi, lo, _gap in missed_pairs(winner_scores, truth_pairs):
            signature = pair_view_signature(traj_by_id.get(hi), traj_by_id.get(lo))
            texts: list[str] = []
            try:
                texts = [
                    str(entry.get("rationale") or "")
                    for entry in judge.presentations_for(hi, lo, signature)
                ]
                if not any(texts):
                    texts = [judge.rationale_for(hi, lo, signature)]
            except Exception:  # noqa: BLE001 — evidence prose is never fatal
                logger.debug("iter %d: no rationale readable for %s / %s", iteration, hi, lo)
            rationales[(hi, lo)] = texts
        report = rationale_themes(rationales, rubric_md, criteria, rule_functions or [])
        (self.store.iteration_dir(iteration) / "rationale_themes.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        block = render_rationale_themes(report)
        if block:
            logger.info("iter %d: %s", iteration, block.splitlines()[0])
        return block

    def _write_criterion_stability(
        self,
        iteration: int,
        candidate_id: str,
        winning_cells: dict[str, dict[str, float]],
        fresh_cells: dict[str, dict[str, float]],
    ) -> str:
        """`iterations/<k>/criterion_stability.json` and its prompt block.

        Written whether the promotion was then confirmed or rejected: a
        rejection on the re-draw is exactly the case where knowing which
        criterion moved is worth the most.
        """

        from meta_filter.criterion_stats import criterion_stability, render_criterion_stability

        stability = criterion_stability(winning_cells, fresh_cells)
        payload = {"iteration": iteration, "candidate": candidate_id, **stability}
        (self.store.iteration_dir(iteration) / "criterion_stability.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        for name, entry in stability["per_criterion"].items():
            logger.info(
                "iter %d: criterion stability %s — %d of %d rows moved between draws",
                iteration,
                name,
                entry["moved"],
                entry["n"],
            )
        return render_criterion_stability(stability, candidate=candidate_id)

    def _solver_settings(self) -> dict[str, object]:
        """The scorer as `config.yaml` records it, for `meta_filter.score_tool`.

        `solver_provider` is the `build_client` keyword, derived from the
        client's class, because the tool has to rebuild the client from the
        file and a class name is not a provider argument.
        """

        scorer = getattr(self.solver, "rubric_scorer", None)
        if not isinstance(scorer, LLMRubricScorer):
            return {"solver_backend": "heuristic"}
        client = unwrap(scorer.client)
        provider = _PROVIDER_BY_CLIENT.get(type(client).__name__, type(client).__name__.lower())
        extra_body = getattr(client, "extra_body", None)
        return {
            "solver_backend": "llm",
            "solver_provider": provider,
            "solver_model": str(getattr(client, "model", "") or ""),
            "solver_base_url": getattr(client, "base_url", None) or None,
            "solver_extra_body": dict(extra_body) if extra_body else None,
            "solver_cache_dir": str(scorer.cache_dir),
        }

    def _write_iteration_precheck(self, iteration: int, outcome: PrecheckOutcome) -> None:
        """`iterations/<k>/precheck.json`: the gate's decisions for one iteration.

        The per-candidate readouts already sit in `candidates/<id>/precheck.json`
        and in `record.json`; this file is the one place that says which of
        them were scored, which were held back, and what the caps said about
        every candidate. `candidates_scored` is repeated here because the
        rank reasons only mean something next to the number they were cut at.
        """

        payload = {
            "candidates_scored": self.config.candidates_scored,
            "scored": [d.name for d in outcome.proposal.candidate_dirs],
            "gated_out": outcome.gated_out,
            "constraints": outcome.constraints,
            "before": outcome.before,
            "after": outcome.after,
        }
        (self.store.iteration_dir(iteration) / "precheck.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _write_run_complete(self, *, stop_reason: str, last_iteration: int) -> None:
        """Record why the loop stopped: `run_id`, `exit_status`, `finished_at`
        and the reason, which is also logged."""

        payload = {
            "run_id": self.run_id,
            "exit_status": 0,
            "finished_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stop_reason": stop_reason,
            "iterations_run": last_iteration + 1,
            "last_iteration": last_iteration,
            "iterations_planned": self.config.iterations,
            "patience": self.config.patience,
            "min_iterations": self.config.min_iterations,
        }
        (self.store.run_dir / RUN_COMPLETE_FILE).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        logger.info(
            "run stopped: %s after iteration %d of %d",
            stop_reason,
            last_iteration,
            self.config.iterations,
        )


# ---------------------------------------------------------------------------
# Helpers for the disagreement-resolution flow
# ---------------------------------------------------------------------------


def _fresh_scorer_draw(scorer: object) -> AbstractContextManager[None]:
    """`scorer.fresh_draw()` when the scorer has one, else a no-op.

    The heuristic scorer and the test doubles have no cache to refresh; their
    scores are deterministic, so the "fresh" draw is the same draw.
    """

    fresh_draw = getattr(scorer, "fresh_draw", None)
    if callable(fresh_draw):
        return fresh_draw()
    return nullcontext()


@dataclass
class PrecheckOutcome:
    """What `_precheck_and_revise` hands back to the loop.

    `proposal` carries only the candidates the solver should pay for.
    `before` / `after` are the per-candidate readouts on either side of the
    revision leg, as before. `gated_out` names every candidate held back and
    why (`reason`, its precheck `score`, its `constraint_status`), and
    `constraints` is the final readout's constraint report for every
    candidate, kept or not — the selector reads `needs_strong_evidence` off
    it and the CSV reads the winner's status.
    """

    proposal: ProposerResult
    before: dict[str, dict] = field(default_factory=dict)
    after: dict[str, dict] = field(default_factory=dict)
    gated_out: dict[str, dict] = field(default_factory=dict)
    constraints: dict[str, dict] = field(default_factory=dict)

    def strong_evidence(self) -> set[str]:
        """Candidates that grew past a soft cap; their gate threshold is one
        z-unit higher (`selector.STRONG_EXTRA_Z`)."""

        return {
            name
            for name, report in self.constraints.items()
            if report.get("status") == "needs_strong_evidence"
        }

    def status_of(self, candidate_id: str) -> str | None:
        report = self.constraints.get(candidate_id)
        return report.get("status") if report else None

    def simplification(self) -> set[str]:
        """Candidates smaller than the baseline in every complexity component;
        the gate accepts them on `b >= c` alone."""

        return {name for name, report in self.constraints.items() if report.get("simplification")}


SECOND_DELETION_REASON = "second deletion candidate this iteration"
# Duplicate collapse: the held candidate's reason is
# "<DUPLICATE_REASON> <kept name> on the judged train pairs".
DUPLICATE_REASON = "duplicate of"
NOOP_REASON = "no-op on judged pairs (orders every already-judged pair as the baseline does)"


def _constraint_dicts(results) -> dict[str, dict]:
    return {
        name: r.constraint.to_dict()
        for name, r in results.items()
        if getattr(r, "constraint", None) is not None
    }


def _precheck_rank_key(result) -> float:
    """Higher is better; a candidate the precheck could not score ranks last.

    `error` covers a candidate that does not load or raised while scoring —
    the solver would discard it anyway, so spending one of the scored slots
    on it would be paying to learn what the precheck already said.
    """

    if result is None or getattr(result, "error", ""):
        return float("-inf")
    return float(result.score)


def gate_candidates(
    candidate_dirs: list[Path],
    results,
    *,
    candidates_scored: int | None,
    parent=None,
) -> tuple[list[Path], dict[str, dict]]:
    """Decide which proposed candidates the solver pays for.

    Four rules, applied in this order so each reads the survivors of the
    last, and every candidate held back is returned with the reason:

    1. A candidate whose constraint report is `invalid` (a hard cap or the
       deletion floor breached, or a rationale declaring two mechanisms) is
       never scored. The reasons are the report's own, so `validation.json`
       quotes the cap that fired.
    2. Duplicate collapse. Candidates whose per-pair correctness on the
       already-judged train pairs is identical are one behaviour written
       several times, so one copy is kept (the lowest
       `complexity`, proposal order on ties) and the rest are held with
       `DUPLICATE_REASON`. A candidate identical to the `parent` on those
       pairs is a no-op and is held with `NOOP_REASON`: it could only ever
       read `b = c = 0` on rubric-shared cells plus scorer noise — unless it
       is a simplification, for which `b = c = 0` is the expected
       result and an accepting one. The vectors come from the precheck, which
       scored exactly these pairs, so the collapse costs nothing and never
       touches the selection set. With no judged pairs (iteration 0) nothing
       has a vector and nothing is collapsed.
    3. At most one deletion candidate per iteration. Among the survivors
       whose report says `deletion`, the highest precheck score stays; the
       others are marked with `SECOND_DELETION_REASON`. Deletions are the
       edits whose damage a train-side number is worst at showing — a
       removed criterion frees complexity on every row and loses ordering
       only on the rows it used to separate — so the loop takes them one
       at a time.
    4. With `candidates_scored` set, the survivors are ranked by precheck
       score (stable in proposal order on ties) and the top N go through;
       the rest are "precheck rank k of m".

    A candidate with no constraint report passes rules 1 and 3 untouched; one
    with no correctness vector (an unscoreable candidate) passes rule 2.

    Args:
        candidate_dirs: The proposed candidates, in proposal order.
        results: ``name -> PrecheckResult`` (or anything with `score`,
            `error`, `constraint`, and optionally `correctness_key()` and
            `complexity`).
        candidates_scored: How many survivors the solver scores; None scores
            them all.
        parent: The baseline's own precheck readout on the same pairs, for
            the no-op check; None skips it.

    Returns:
        The kept directories in their original order, and the gated-out
        map ``name -> {"reason", "score", "constraint_status"}``.
    """

    gated_out: dict[str, dict] = {}

    def _hold(cand_dir: Path, reason: str) -> None:
        result = results.get(cand_dir.name)
        constraint = getattr(result, "constraint", None) if result is not None else None
        gated_out[cand_dir.name] = {
            "reason": reason,
            "score": (
                None
                if result is None or getattr(result, "error", "")
                else round(float(result.score), 6)
            ),
            "constraint_status": constraint.status if constraint is not None else None,
        }

    def _constraint(cand_dir: Path):
        result = results.get(cand_dir.name)
        return getattr(result, "constraint", None) if result is not None else None

    survivors: list[Path] = []
    for cand_dir in candidate_dirs:
        constraint = _constraint(cand_dir)
        if constraint is not None and constraint.status == "invalid":
            _hold(cand_dir, "constraint invalid: " + "; ".join(constraint.reasons))
            continue
        survivors.append(cand_dir)

    survivors = _collapse_duplicates(survivors, results, parent=parent, hold=_hold)

    # A simplification makes the filter smaller in every component,
    # so the damage a deletion can hide has nowhere to hide: it is not the
    # kind of deletion the one-per-iteration rule takes slowly, and it does
    # not use up the slot.
    deletions = [
        d
        for d in survivors
        if getattr(_constraint(d), "deletion", False)
        and not getattr(_constraint(d), "simplification", False)
    ]
    if len(deletions) > 1:
        best = max(deletions, key=lambda d: _precheck_rank_key(results.get(d.name)))
        for cand_dir in deletions:
            if cand_dir is not best:
                _hold(cand_dir, f"{SECOND_DELETION_REASON} (kept {best.name})")
        survivors = [d for d in survivors if d is best or d not in deletions]

    if candidates_scored is not None and len(survivors) > candidates_scored:
        ranked = sorted(
            survivors,
            key=lambda d: -_precheck_rank_key(results.get(d.name)),
        )
        cut = set(id(d) for d in ranked[:candidates_scored])
        total = len(survivors)
        for rank, cand_dir in enumerate(ranked, start=1):
            if id(cand_dir) not in cut:
                _hold(cand_dir, f"precheck rank {rank} of {total}")
        survivors = [d for d in survivors if id(d) in cut]

    return survivors, gated_out


def _correctness_key(result) -> tuple | None:
    """`PrecheckResult.correctness_key()` when the readout has one, else None."""

    key_of = getattr(result, "correctness_key", None)
    if not callable(key_of):
        return None
    return key_of()


def _collapse_duplicates(
    survivors: list[Path],
    results,
    *,
    parent,
    hold: Callable[[Path, str], None],
) -> list[Path]:
    """Rule 2 of `gate_candidates`: one candidate per distinct train behaviour.

    Groups the survivors by their per-pair correctness vector on the judged
    train pairs. A group equal to the parent's vector is held whole as a
    no-op; every other group of two or more keeps its lowest-complexity
    member (proposal order on ties) and holds the rest as duplicates of it.
    Survivors without a vector are left alone and keep their place.
    """

    parent_key = _correctness_key(parent) if parent is not None else None
    groups: dict[tuple, list[Path]] = {}
    for cand_dir in survivors:
        key = _correctness_key(results.get(cand_dir.name))
        if key is not None:
            groups.setdefault(key, []).append(cand_dir)

    def _is_simplification(cand_dir: Path) -> bool:
        constraint = getattr(results.get(cand_dir.name), "constraint", None)
        return bool(getattr(constraint, "simplification", False))

    held: set[int] = set()
    for key, members in groups.items():
        if parent_key is not None and key == parent_key:
            # Identical to the parent on every judged pair. That is a no-op
            # for a regular candidate — and exactly what a simplification
            # looks like: dropping a rule that never fired or an axis
            # that orders nothing changes no pair. Simplifications stay;
            # anything else in the group is held.
            kept_simplifications = [d for d in members if _is_simplification(d)]
            for cand_dir in members:
                if cand_dir not in kept_simplifications:
                    hold(cand_dir, NOOP_REASON)
                    held.add(id(cand_dir))
            members = kept_simplifications
        if len(members) < 2:
            continue

        def _complexity(cand_dir: Path) -> float:
            value = getattr(results.get(cand_dir.name), "complexity", None)
            return float(value) if value is not None else float("inf")

        # `min` is stable: the first of equal-complexity members wins.
        kept = min(members, key=_complexity)
        for cand_dir in members:
            if cand_dir is not kept:
                hold(cand_dir, f"{DUPLICATE_REASON} {kept.name} on the judged train pairs")
                held.add(id(cand_dir))
    return [d for d in survivors if id(d) not in held]


@dataclass
class PromotionConfirmation:
    """Whether the post-promotion re-draw agreed with the acceptance.

    `b1`/`c1` are draw 1's discordant counts (the ones the gate accepted on),
    `b2`/`c2` the same candidate's counts on its fresh selection scores
    against the SAME baseline correctness. `p_pooled` is the one-sided test
    on ``b1 + b2`` vs ``c1 + c2`` — `paired_test.stats_from_counts`, exact
    below 25 discordant pairs — held to the `p_threshold` draw 1 was held to
    (Šidák-corrected, one z-unit higher for a strong-evidence candidate). A
    re-score
    that failed leaves `b2`, `c2` and `p_pooled` None and is not confirmed.

    `b2`/`c2`/`p_pooled` are also None, with `confirmed` True and `method`
    ``"identical_ordering"``, for the one acceptance that is settled without
    a second draw: a simplification at ``b1 = c1 = 0``. There the
    two states are distinguished by `confirmed`, not by the empty counts.
    """

    b1: int
    c1: int
    b2: int | None
    c2: int | None
    z_pooled: float | None
    p_pooled: float | None
    method: str
    p_threshold: float
    confirmed: bool
    reason: str
    fresh_sel_concordance: float | None = None

    def to_dict(self) -> dict:
        return {
            "b1": self.b1,
            "c1": self.c1,
            "b2": self.b2,
            "c2": self.c2,
            "z_pooled": self.z_pooled,
            "p_pooled": self.p_pooled,
            "method": self.method,
            "p_threshold": self.p_threshold,
            "confirmed": self.confirmed,
            "reason": self.reason,
            "fresh_sel_concordance": self.fresh_sel_concordance,
        }


def _settled_without_a_redraw(result: SelectionResult) -> bool:
    """Whether this acceptance needs no second draw at all.

    True for exactly one shape: the gate's winner is a simplification and
    draw 1 read ``b = c = 0``. The candidate then ordered every pair of the
    hidden exam exactly as the baseline did, and the simplification rule
    already established it is strictly smaller in every complexity
    component — which together are the whole claim being made. A re-score
    can only disagree by scorer noise, so an identical prune would be kept or
    refused on a coin flip.

    Not true of a simplification with ``b > c``, which claims pairs it
    actually changed and still has to survive ``b2 >= c2``; not reachable
    for a regular candidate, which at ``b + c = 0`` is a no-op and never
    becomes the gate's winner in the first place (`Selector.choose_paired`
    blocks it on `min_discordant`).
    """

    return result.simplification_winner and result.b == 0 and result.c == 0


def confirm_promotion(
    *,
    b1: int,
    c1: int,
    fresh_correctness: Mapping[tuple[str, str], float] | None,
    baseline_correctness: Mapping[tuple[str, str], float],
    p_threshold: float,
    simplification: bool = False,
) -> PromotionConfirmation:
    """Decide whether an acceptance survives its own re-draw.

    Confirmed iff the re-draw still favours the candidate (``b2 > c2``) and
    the pooled statistic over both draws clears `p_threshold`. Draw 1 having
    passed is the caller's precondition — this is only ever called for the
    gate's winner. Pooling rather than requiring the re-draw to pass alone
    is deliberate: the re-draw is one more sample of the same comparison,
    and demanding that a single fresh draw clear a three-candidate threshold
    would reject most true improvements at 500–800 pairs; demanding that the
    two draws together clear it, with the re-draw at least pointing the same
    way, rejects the acceptances one lucky draw produced.

    The one acceptance that is settled without a second draw is a
    simplification at ``b1 = c1 = 0``: the candidate ordered every
    pair as the baseline did and is smaller in every complexity component,
    so there is nothing a fresh draw could add. `fresh_correctness` is then
    ignored, and the caller should not pay for the re-score at all.

    Args:
        b1: Draw 1, candidate right and baseline wrong.
        c1: Draw 1, baseline right and candidate wrong.
        fresh_correctness: The candidate's per-pair credit on its fresh
            scores (`pair_correctness`), or None when the re-score failed.
        baseline_correctness: The baseline's per-pair credit draw 1 used.
        p_threshold: The one-sided tail probability draw 1 was held to.
        simplification: The winner was accepted on the simplification rule
            Then the re-draw only has to lose no pair
            (``b2 >= c2``); there was no significance test on draw 1 to pool
            with — and at ``b1 = c1 = 0`` there is no re-draw at all (below).

    Returns:
        The confirmation record, with `confirmed` set.
    """

    if simplification and b1 == 0 and c1 == 0:
        # A simplification at b = c = 0 ordered every pair of the exam
        # exactly as the baseline did, and it is strictly smaller in every
        # complexity component. That IS the whole claim, and draw 1 already
        # observed it on all N pairs — so a second draw has nothing to
        # confirm. What it would measure is scorer noise: the true b - c is
        # exactly 0, the observed b2 - c2 is a coin flip, and a one-sided
        # `b2 >= c2` would pass about half the time. Confirming here costs
        # nothing and removes the coin flip; a simplification with b1 > c1 is
        # a real claim about pairs it changed and still re-draws below.
        return PromotionConfirmation(
            b1=b1,
            c1=c1,
            b2=None,
            c2=None,
            z_pooled=None,
            p_pooled=None,
            method="identical_ordering",
            p_threshold=p_threshold,
            confirmed=True,
            reason=(
                f"ordering identical on all {len(baseline_correctness)} selection pairs "
                "(draw 1 b/c 0/0); smaller in every complexity component; "
                "confirmed without a second draw"
            ),
        )
    if fresh_correctness is None:
        return PromotionConfirmation(
            b1=b1,
            c1=c1,
            b2=None,
            c2=None,
            z_pooled=None,
            p_pooled=None,
            method="none",
            p_threshold=p_threshold,
            confirmed=False,
            reason=(
                "promotion re-score failed; the acceptance could not be confirmed on a fresh draw"
            ),
        )
    stats = paired_stats(fresh_correctness, baseline_correctness)
    b2, c2 = stats.b, stats.c
    z_pooled, p_pooled, method = stats_from_counts(b1 + b2, c1 + c2)
    fresh_sel = (
        sum(fresh_correctness.values()) / len(fresh_correctness) if fresh_correctness else 0.0
    )
    blockers: list[str] = []
    if simplification:
        if b2 < c2:
            blockers.append(f"simplification re-draw lost pairs: b2={b2} < c2={c2}")
    else:
        if b2 <= c2:
            blockers.append(f"re-draw b2={b2} <= c2={c2}")
        if p_pooled > p_threshold:
            blockers.append(
                f"pooled p={p_pooled:.4f} > p_threshold={p_threshold:.4f} "
                f"(b={b1 + b2} c={c1 + c2}, z={z_pooled:.2f}, {method})"
            )
    if blockers:
        reason = "; ".join(blockers)
    elif simplification:
        reason = (
            f"simplification: draw 1 b/c {b1}/{c1}, re-draw {b2}/{c2} (b2 >= c2, no test required)"
        )
    else:
        reason = (
            f"draw 1 b/c {b1}/{c1}, re-draw {b2}/{c2}, pooled p={p_pooled:.4f} "
            f"<= p_threshold={p_threshold:.4f} (z={z_pooled:.2f}, {method})"
        )
    return PromotionConfirmation(
        b1=b1,
        c1=c1,
        b2=b2,
        c2=c2,
        z_pooled=z_pooled,
        p_pooled=p_pooled,
        method=method,
        p_threshold=p_threshold,
        confirmed=not blockers,
        reason=reason,
        fresh_sel_concordance=fresh_sel,
    )


def _gate_record(
    result: SelectionResult, decision: str, confirmation: PromotionConfirmation | None = None
) -> dict:
    """The paired-gate fields of a `SelectionResult`, for `selection.json`.

    `decision` is the final one — "accepted", "accepted_simplification",
    "kept", or "rejected_on_redraw" — and `confirmation` the record when a
    promotion was attempted. The b/c/z/p fields are draw 1's for the
    candidate the gate picked, whatever the re-draw then said.
    """

    return {
        "gate": result.gate,
        "decision": decision,
        "sel_concordance": result.sel_concordance,
        "baseline_sel_concordance": result.baseline_sel_concordance,
        "b": result.b,
        "c": result.c,
        "z": result.z,
        "p": result.p,
        "z_cluster": result.z_cluster,
        "threshold": result.threshold,
        "confirmation": confirmation.to_dict() if confirmation is not None else None,
    }


def _confirmation_summary_columns(confirmation: PromotionConfirmation | None) -> dict[str, object]:
    """The confirmation `summary.csv` columns: what the re-draw said.

    All four cells are empty when no promotion was attempted (the baseline
    kept on draw 1): nothing was re-drawn, so there is no
    second count to show. `gate_confirmed` is True/False otherwise, False
    also covering a re-score that failed (then `sel_b2`/`sel_c2` are empty
    and `sel_p_pooled` too).

    The three counts are also empty with `gate_confirmed` True
    for the simplification confirmed without a second draw (draw 1 read
    `sel_b`/`sel_c` 0/0). Read `gate_confirmed`, not the emptiness, to tell
    the two apart.
    """

    if confirmation is None:
        return {"sel_b2": "", "sel_c2": "", "sel_p_pooled": "", "gate_confirmed": ""}
    return {
        "sel_b2": confirmation.b2 if confirmation.b2 is not None else "",
        "sel_c2": confirmation.c2 if confirmation.c2 is not None else "",
        "sel_p_pooled": (
            round(confirmation.p_pooled, 6) if confirmation.p_pooled is not None else ""
        ),
        "gate_confirmed": confirmation.confirmed,
    }


def _selection_summary_columns(
    result: SelectionResult,
    decision: str,
    *,
    promoted_rescore_delta: float | None,
    patience: int,
    selection_pairs: int | None,
) -> dict[str, object]:
    """The selection gate's `summary.csv` columns, in their fixed order.

    Cells are empty strings rather than zeros when the gate did not run:
    a `sel_z` of 0 would read as "tested, no difference", which is not what
    happened.
    """

    on = decision != "off"
    return {
        "sel_concordance": result.sel_concordance if on else "",
        "sel_baseline_concordance": result.baseline_sel_concordance if on else "",
        "sel_b": result.b if on else "",
        "sel_c": result.c if on else "",
        "sel_z": round(result.z, 4) if on else "",
        "sel_z_cluster": round(result.z_cluster, 4) if on else "",
        "gate_decision": decision,
        "gate_threshold": round(result.threshold, 4) if on else "",
        "promoted_rescore_delta": (
            round(promoted_rescore_delta, 6) if promoted_rescore_delta is not None else ""
        ),
        "patience": patience,
        "selection_pairs": selection_pairs if selection_pairs is not None else "",
    }


def _gating_summary_columns(
    outcome: PrecheckOutcome, *, winner: str, winner_is_baseline: bool
) -> dict[str, object]:
    """The precheck gate's `summary.csv` columns: how many candidates the gate
    held back, and the winner's constraint status.

    `constraint_status` is "baseline" for a kept baseline and empty when no
    precheck ran (the status is unknown, not "ok").
    """

    if winner_is_baseline:
        status: str | None = "baseline"
    else:
        status = outcome.status_of(winner)
    return {
        "gated_out": len(outcome.gated_out),
        "constraint_status": status if status is not None else "",
    }


def _lookup_winner_notes(winner_id: str, proposer_candidates: list) -> str:
    """Find the proposer-emitted ``notes`` for the winning candidate.

    The optimizer materializes each candidate as
    ``iter{k:03d}_{i:02d}_{slug}`` where ``slug`` comes from the
    proposer JSON's ``name`` field. We suffix-match to recover the notes.
    Baseline winners have no proposer entry so we return "" — those iters
    just "kept baseline" which the selector reason already conveys.
    """

    for cand in proposer_candidates or []:
        slug = str((cand or {}).get("name") or "").strip()
        if slug and winner_id.endswith(slug):
            return str(cand.get("notes") or "")
    return ""


def _new_run_id(runs_dir: Path) -> str:
    """Timestamped run id, disambiguated when runs start the same second."""

    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if not (runs_dir / stamp).exists():
        return stamp
    for n in range(1, 100):
        alt = f"{stamp}_{n}"
        if not (runs_dir / alt).exists():
            return alt
    raise RuntimeError(f"cannot allocate a run id under {runs_dir}")


# How many mis-ordered pairs one candidate contributes to `critic_outputs/`.
# Each pair yields two records carrying a full trajectory snapshot, so the cap
# is about file size, not about signal: the list is sorted most-confidently-
# wrong first and the prompt blocks render only the top five anyway.
MAX_PAIR_MISORDER_RECORDS = 20

# `build_client` keyword per client class, for `config.yaml`'s `solver_provider`.
_PROVIDER_BY_CLIENT: dict[str, str] = {
    "MockLLMClient": "mock",
    "OpenAICompatibleClient": "openai",
    "AnthropicClient": "anthropic",
    "CodexCliClient": "codex",
    "ClaudeCodeHeadlessClient": "claude-code",
}

# Characters one pair's per-criterion line may take in the proposer prompt.
# Six criteria fit whole; past that the opposing axes are kept and the rest
# is counted, so the block the pair sits in does not grow with the rubric.
_BREAKDOWN_BUDGET = 220


def _criterion_breakdown(
    preferred: FilterDecision | None,
    other: FilterDecision | None,
    *,
    budget: int = _BREAKDOWN_BUDGET,
) -> str:
    """One line: per-criterion score difference across a misordered pair.

    `preferred` is the judge's pick, `other` the row the filter put above
    it, so a NEGATIVE difference is a criterion that argued against the
    judge — the one most negative is flagged, because that is the axis
    whose reading inverted the pair and the thing a rubric edit can act on.
    The two rows' `rubric_scores` are already in the prompt separately; the
    subtraction is what was missing, and a proposer left to do it by eye
    over six criteria and five pairs did not.
    """

    if preferred is None or other is None:
        return ""
    shared = [c for c in preferred.rubric_scores if c in other.rubric_scores]
    if not shared:
        return ""
    diffs = sorted(
        ((c, float(preferred.rubric_scores[c]) - float(other.rubric_scores[c])) for c in shared),
        key=lambda item: item[1],
    )
    worst = diffs[0][0] if diffs[0][1] < 0 else None

    def _piece(name: str, delta: float) -> str:
        flag = " (OPPOSES the judge most)" if name == worst else ""
        return f"{name} {delta:+g}{flag}"

    head = "per-criterion (judge's pick minus the row ranked above it): "
    full = head + ", ".join(_piece(c, d) for c, d in diffs)
    if len(full) <= budget:
        return full
    opposing = [item for item in diffs if item[1] < 0][:2]
    rest = len(diffs) - len(opposing)
    shown = ", ".join(_piece(c, d) for c, d in opposing) or "no criterion opposes the judge"
    return f"{head}{shown}; {rest} other criteria in rubric_scores"


# How many of the misorder slots are reserved for pairs the judge bought in
# THIS iteration. 0.4 puts 2 of the prompt's 5 rendered pairs on the freshest
# buy and 8 of the 20 saved records; see `_select_misordered_pairs` for why a
# minority and not all of them.
_FRESH_MISORDER_SHARE = 0.4

# Characters one pair's history line may take. It replaces nothing, so it is
# the only way this block can grow: five pairs at this cap is 0.7 KB, small
# beside the misorder blocks themselves.
_PAIR_HISTORY_BUDGET = 160


def _pair_history_note(
    *,
    source: str | None,
    judged_iteration: int | None,
    iteration: int | None,
    shown_at: list[int],
    budget: int = _PAIR_HISTORY_BUDGET,
) -> str:
    """One line saying where a misordered pair came from and who has seen it.

    The proposer can read the pair itself off the block. What it cannot read
    is whether this pair is the fallout of its own last move or a standing
    failure its whole lineage has been shown and has not fixed — and those
    two call for different edits. Returns "" when nothing is known, so a
    caller that passes no provenance renders exactly what it rendered before.
    """

    if source is None and judged_iteration is None and not shown_at:
        return ""
    bits: list[str] = []
    if judged_iteration is None:
        bits.append("verdict inherited from an earlier run")
    elif iteration is not None and judged_iteration == iteration:
        bits.append(f"bought this iteration (iter {judged_iteration})")
    else:
        bits.append(f"bought at iter {judged_iteration}")
    if source is not None:
        bits.append("consensus-audit stratum" if source == "audit" else "disputed stratum")
    if shown_at:
        seen = ", ".join(str(i) for i in shown_at[-3:])
        bits.append(f"already shown at iter {seen} and STILL mis-ordered")
    line = "pair history: " + "; ".join(bits)
    return line if len(line) <= budget else line[: budget - 1] + "\u2026"


def _misorder_slot_is_fresh(index: int, share: float) -> bool:
    """Whether slot `index` belongs to the fresh-buy stream.

    A Bresenham split rather than "the first N slots": it spreads the fresh
    minority through the list, so the top of the block — the only part the
    prompt renders — carries both streams at the intended ratio no matter
    what `limit` is.
    """

    return int((index + 1) * share) > int(index * share)


def _select_misordered_pairs(
    *,
    scores: dict[str, float],
    resolved_pairs: list[tuple[str, str]],
    provenance: dict[frozenset[str], dict] | None = None,
    iteration: int | None = None,
    shown_before: dict[frozenset[str], list[int]] | None = None,
    limit: int = MAX_PAIR_MISORDER_RECORDS,
    fresh_share: float = _FRESH_MISORDER_SHARE,
) -> list[tuple[str, str, float, str]]:
    """Which of the judged pairs this candidate gets wrong to show, and in what order.

    `resolved_pairs` is meant to be the ACCUMULATED truth set restricted to
    the live pool — everything the run has ever bought that applies here —
    not the iteration's fresh purchase. Handed only the fresh purchase, the
    prompts would name only pairs judged at the immediately preceding
    iteration, and most of the pairs the run paid for could never be named.

    Returned as `(winner, loser, gap, history_note)`, most useful first. The
    order is the whole selection rule, because the prompt renders only the
    head of it:

    * **A fresh minority.** `fresh_share` of the slots go to pairs bought this
      iteration. The proposer needs to see what its last move provoked, and
      those pairs are also the only ones targeted at the candidates that
      currently disagree. A minority, because they are already the pairs most
      likely to be noise: the acquisition function picked them precisely
      where the committee was split.
    * **The rest to standing failures.** Pairs bought earlier that the
      champion *still* orders wrongly. A pair the filter got wrong three
      iterations ago and gets wrong today is better evidence than a pair it
      has never been asked about, and it is the only evidence that says a
      previous fix did not take.
    * **Most confidently wrong first**, inside each stream: `missed_pairs`
      sorts on the inversion gap, so the head is where the design is not
      merely wrong but sure of itself.
    * **Disputed before audit.** Consensus-audit pairs order candidates far
      less like the hidden selection exam than disputed pairs do, so as
      *ranking* evidence they are close to noise. They are not dropped — they
      are the lineage's blind spot and they carry real information — but that
      information has its own channel
      (`--pair-evidence`), which is what lets this block deprioritise them
      without losing them.

    Repeats are deliberately NOT suppressed. A pair the champion still gets
    wrong is still the best evidence there is, and hiding it would make a
    stuck loop look like a moving one. What `shown_before` buys instead is
    the history line: the proposer is told that this pair has already been
    put in front of its lineage and the filter still inverts it.
    """

    from meta_filter.ranking import missed_pairs

    prov = provenance or {}
    seen = shown_before or {}
    fresh: list[tuple] = []
    standing: list[tuple] = []
    for winner_id, loser_id, gap in missed_pairs(scores, resolved_pairs):
        key = frozenset((winner_id, loser_id))
        record = prov.get(key) or {}
        source = record.get("source")
        bought = record.get("judged_iteration")
        note = _pair_history_note(
            source=source,
            judged_iteration=bought,
            iteration=iteration,
            shown_at=list(seen.get(key, ())),
        )
        row = (winner_id, loser_id, gap, note, source == "audit")
        is_fresh = iteration is not None and bought == iteration
        (fresh if is_fresh else standing).append(row)

    # Stable, so `missed_pairs`' most-confidently-wrong-first order survives
    # inside each stratum.
    fresh.sort(key=lambda row: row[4])
    standing.sort(key=lambda row: row[4])

    out: list[tuple[str, str, float, str]] = []
    fi = si = 0
    for slot in range(limit):
        if fi >= len(fresh) and si >= len(standing):
            break
        take_fresh = _misorder_slot_is_fresh(slot, fresh_share)
        if take_fresh and fi >= len(fresh):
            take_fresh = False
        elif not take_fresh and si >= len(standing):
            take_fresh = True
        if take_fresh:
            row = fresh[fi]
            fi += 1
        else:
            row = standing[si]
            si += 1
        out.append((row[0], row[1], row[2], row[3]))
    return out


def _audit_blind_spot_pairs(
    *,
    scores: dict[str, float],
    pairs: list[tuple[str, str]],
    provenance: dict[frozenset[str], dict],
    exclude: set[frozenset[str]] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, float, int | None]]]:
    """The consensus-audit pairs this score map gets wrong, blind spot first.

    Returns `(audit_pool, missed)` — the audit-stratum pairs that lie in
    `pairs`, and the ones the map inverts as
    `(winner, loser, gap, audited_iteration)`, most confidently wrong first.
    Both are needed by the caller: the readout is "N of M", and M is the
    audit pairs that applied, not the ones that went wrong.

    `exclude` drops pairs another channel already shows. That is what makes
    this channel worth its prompt lines: an audit pair is one every candidate
    ordered the same way and the judge reversed, so no disagreement-driven
    sample can reach it — whereas filling it from the same mis-ordered list
    the misordered-pair blocks render would repeat them exactly.
    """

    from meta_filter.ranking import missed_pairs

    skip = exclude or set()
    audit_pool = [
        pair for pair in pairs if (provenance.get(frozenset(pair)) or {}).get("source") == "audit"
    ]
    missed: list[tuple[str, str, float, int | None]] = []
    for winner_id, loser_id, gap in missed_pairs(scores, audit_pool):
        key = frozenset((winner_id, loser_id))
        if key in skip:
            continue
        missed.append(
            (winner_id, loser_id, gap, (provenance.get(key) or {}).get("judged_iteration"))
        )
    return audit_pool, missed


def _misorder_pair_key(example: CounterExample) -> frozenset[str]:
    """The unordered pair a `pair_misorder_vs:` counterexample belongs to."""

    prefix = "pair_misorder_vs:"
    subtype = example.failure_subtype or ""
    partner = subtype[len(prefix) :] if subtype.startswith(prefix) else ""
    return frozenset((example.trajectory_id, partner))


def _pair_misorder_examples(
    *,
    scores: dict[str, float],
    resolved_pairs: list[tuple[str, str]],
    traj_by_id: dict[str, Trajectory],
    decisions_by_id: dict[str, FilterDecision],
    iteration: int,
    filter_version: str,
    rationale_for=None,
    provenance: dict[frozenset[str], dict] | None = None,
    shown_before: dict[frozenset[str], list[int]] | None = None,
    limit: int = MAX_PAIR_MISORDER_RECORDS,
) -> tuple[list[CounterExample], list[CounterExample]]:
    """Judge-derived counterexamples for a ranking candidate.

    Returns `(false_negatives, false_positives)` built ONLY from pairs the
    pair judge ordered and this candidate ordered the other way round. Both
    directions of every such pair are recorded, because a mis-ordered pair is
    two mistakes at once and the proposer needs to see which row was pushed up
    and which was pushed down:

    * the row the judge dispreferred but the filter ranked higher is the
      ranking analogue of a false negative — the filter kept what the second
      opinion would drop;
    * the row the judge preferred but the filter ranked lower is the false
      positive, its mirror.

    Nothing here reads the filter's own verdict as if it were a critic's. That
    is the whole point: `BoundaryCritic` writes `critic_decision =
    predicted_decision`, and those self-copies would be the bulk of what
    `critic_outputs/` held while the prompt calls the directory "where the
    judge disagreed".

    WHICH pairs, and in what order, is `_select_misordered_pairs`; pass it the
    accumulated truth set restricted to this pool, plus `provenance` and
    `shown_before`, and the head of the list is the deliberate selection the
    prompt renders. With neither passed the order is the pure inversion-gap
    order.
    """

    false_negatives: list[CounterExample] = []
    false_positives: list[CounterExample] = []
    for winner_id, loser_id, gap, history in _select_misordered_pairs(
        scores=scores,
        resolved_pairs=resolved_pairs,
        provenance=provenance,
        iteration=iteration if provenance is not None else None,
        shown_before=shown_before,
        limit=limit,
    ):
        why = ""
        if rationale_for is not None:
            try:
                why = rationale_for(winner_id, loser_id)
            except Exception:  # noqa: BLE001 — evidence prose is never fatal
                why = ""
        judge_said = f' Judge: "{why}"' if why else ""
        breakdown = _criterion_breakdown(
            decisions_by_id.get(winner_id), decisions_by_id.get(loser_id)
        )

        for tid, partner, critic_decision, predicted, bucket in (
            (loser_id, winner_id, Decision.REJECT, Decision.ACCEPT, false_negatives),
            (winner_id, loser_id, Decision.ACCEPT, Decision.REJECT, false_positives),
        ):
            traj = traj_by_id.get(tid)
            if traj is None:
                continue
            decision = decisions_by_id.get(tid)
            over = critic_decision is Decision.REJECT
            direction = "ABOVE" if over else "BELOW"
            bucket.append(
                CounterExample(
                    trajectory_id=tid,
                    source_iteration=iteration,
                    filter_version=filter_version,
                    critic_decision=critic_decision,
                    predicted_decision=predicted,
                    failure_type="false_negative" if over else "false_positive",
                    failure_subtype=f"pair_misorder_vs:{partner}",
                    severity=2,
                    confidence=decision.confidence if decision is not None else 0.0,
                    evidence_turns=list(range(min(len(traj.turns), 8))),
                    description=(
                        f"The pair judge ranked {partner} above {tid}; this "
                        f"filter scored {tid} {direction} {partner} by "
                        f"{gap:.2f}.{judge_said}"
                    ),
                    why_filter_failed=(
                        f"scored {tid} {direction} {partner} against the "
                        f"judge's order (inversion gap {gap:.2f})"
                    ),
                    raw_trajectory=traj,
                    solver_decision=decision,
                    criterion_breakdown=breakdown,
                    pair_history=history,
                )
            )
    return false_negatives, false_positives


def _tiny_histogram(traj: Trajectory, top_n: int = 6) -> dict[str, int]:
    """Compact top-N action histogram for critic evidence packs."""

    from collections import Counter

    c: Counter[str] = Counter()
    for t in traj.turns:
        key = (t.action or t.role or "").strip()
        if key:
            c[key] += 1
    return dict(c.most_common(top_n))


def _summarize_trajectory_pool(trajectories: list[Trajectory]) -> str:
    """Produce a short textual summary of the training pool so the proposer
    understands the shape of the data it's filtering (dataset breakdown, api
    call counts, submission size distribution, exit-status mix).

    Kept intentionally short so it doesn't dominate the prompt. Distribution
    stats use min/median/max instead of full histograms.

    There is no exit-status line: the rows carry no such field, and a line
    reporting it as "(none)" would be the prompt naming it.
    """

    if not trajectories:
        return "(no trajectories)"

    from collections import Counter
    from statistics import median

    n = len(trajectories)
    api_calls: list[int] = []
    n_messages: list[int] = []
    submission_bytes: list[int] = []
    datasets: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()

    for t in trajectories:
        md = t.metadata or {}
        if isinstance(md.get("api_calls"), int):
            api_calls.append(md["api_calls"])
        if isinstance(md.get("n_messages"), int):
            n_messages.append(md["n_messages"])
        if isinstance(md.get("submission_bytes"), int):
            submission_bytes.append(md["submission_bytes"])
        iid = md.get("instance_id") or ""
        if isinstance(iid, str):
            # instance_id typically encodes the dataset via prefix (org__repo)
            # via the source path — just take a lightweight sniff.
            src = md.get("source", "")
            if src:
                datasets[str(src)] += 1
        for turn in t.turns:
            a = (turn.action or turn.role or "").strip()
            if a:
                action_counter[a] += 1

    def _stats(values: list[int]) -> str:
        if not values:
            return "(none)"
        return f"min={min(values)} median={int(median(values))} max={max(values)}"

    top_actions = ", ".join(f"{a}={c}" for a, c in action_counter.most_common(8))
    dataset_mix = ", ".join(f"{k}={v}" for k, v in datasets.most_common(3)) or "(unknown)"

    return "\n".join(
        [
            f"- pool size: {n} trajectories",
            f"- source(s): {dataset_mix}",
            f"- api_calls: {_stats(api_calls)}",
            f"- messages per trajectory: {_stats(n_messages)}",
            f"- submission_bytes: {_stats(submission_bytes)}",
            f"- top actions across pool: {top_actions}",
        ]
    )


def _consensus_from_decisions(
    decision_lists: list[list[FilterDecision]],
    trajectories: list[Trajectory],
) -> dict[str, Decision | None]:
    """For each trajectory, if all candidates gave the same decision, that
    decision is the consensus (used as pseudo-truth). Otherwise the entry is
    None, which the caller interprets as "resolver needed".

    REVIEW decisions still count as votes — a candidate that says REVIEW is
    hedging, and only if *every* candidate hedges the same way do we accept
    that as consensus. In practice REVIEW-only consensus is rare.
    """

    consensus: dict[str, Decision | None] = {}
    per_id: dict[str, set[Decision]] = {t.trajectory_id: set() for t in trajectories}
    for dlist in decision_lists:
        for d in dlist:
            if d.trajectory_id in per_id:
                per_id[d.trajectory_id].add(d.decision)
    for tid, votes in per_id.items():
        if len(votes) == 1:
            consensus[tid] = next(iter(votes))
        else:
            consensus[tid] = None
    return consensus


def _eval_set_identity(pairs: list[tuple[str, str]], *, accumulated: int) -> dict:
    """What this iteration was graded on, hashed so it can be compared.

    The count alone is not enough — two iterations can offer the same number
    of pairs and not the same pairs. The hash is over the sorted pair ids, so
    it is stable against the order they were bought in. Every pair is bought
    by the run itself, so `inherited_pairs` is always 0 and
    `bought_this_run` equals `accumulated_pairs`; both keys stay so every
    `ranking.json` has one shape.
    """

    digest = hashlib.sha256(
        "\n".join(sorted("|".join(sorted(p)) for p in pairs)).encode("utf-8")
    ).hexdigest()
    return {
        "graded_pairs": len(pairs),
        "graded_sha256": digest,
        "accumulated_pairs": accumulated,
        "inherited_pairs": 0,
        "bought_this_run": accumulated,
    }


def _median_novelty(shapes: list) -> float:
    """Median share of a candidate that is new relative to its baseline.

    Median rather than mean so one deliberate rewrite among three edits does
    not read as the whole iteration having rewritten.
    """

    if not shapes:
        return 0.0
    values = sorted(s.novelty for s in shapes)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _candidate_workers(configured: int, n_candidates: int) -> int:
    """Resolve `candidates_parallel` against how many candidates there are.

    0 means "one worker per candidate": an iteration then costs the slowest
    candidate rather than the sum of all of them.
    """

    if n_candidates < 2:
        return 1
    return min(configured or n_candidates, n_candidates)


def _stub_metrics(total: int, decisions: list[FilterDecision]):
    """A minimal `FilterMetrics` used only as a seed for the proposer prompt
    when we haven't resolved disagreements yet. Loss / FN / FP are zeroed so
    the proposer relies on the file contents and memory, not on stale metrics.
    """

    from meta_filter.schemas import FilterMetrics

    counts = {"accepted": 0, "rejected": 0, "reviewed": 0}
    for d in decisions:
        if d.decision == Decision.ACCEPT:
            counts["accepted"] += 1
        elif d.decision == Decision.REJECT:
            counts["rejected"] += 1
        else:
            counts["reviewed"] += 1
    return FilterMetrics(
        total=total,
        critic_accept_total=0,
        critic_reject_total=0,
        accepted=counts["accepted"],
        rejected=counts["rejected"],
        reviewed=counts["reviewed"],
        agree_accept=0,
        agree_reject=0,
        filter_accept_critic_reject=0,
        filter_reject_critic_accept=0,
        critic_reject_recall=0.0,
        critic_accept_retention=0.0,
        disputed_accept_rate=0.0,
        disputed_reject_rate=0.0,
        review_rate=counts["reviewed"] / total if total else 0.0,
    )


def _append_rule_pricing(line: str, loaded_filter, decisions, trajectories=None) -> str:
    """Add the rule-wiring readout to a rank-lift line, when there is one.

    Kept beside the lift line rather than in its own prompt block because the
    two are read together: rank lift says a rule does not move the ordering,
    and this says whether that is the rule's fault or its wiring's. Reporting
    the first without the second gets a rule whose price key has a typo
    deleted for being useless.
    """

    from meta_filter.criterion_lift import render_rule_pricing, rule_pricing

    try:
        pricing = render_rule_pricing(rule_pricing(loaded_filter, decisions, trajectories))
    except Exception:  # pragma: no cover - a diagnostic must never stop a run
        logger.debug("rule pricing readout failed", exc_info=True)
        return line
    return f"{line}\n- {pricing}" if pricing else line
