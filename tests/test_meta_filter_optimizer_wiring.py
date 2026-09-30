"""Unit-level checks for the optimizer wiring.

These are cheap tests that don't need to spin up the whole run loop —
they exercise the small helper surface where regressions are most likely
(config defaults, `compute_score` end-to-end, curated evidence builder).
"""

from __future__ import annotations

from meta_filter.critics import TrajectoryCritic
from meta_filter.critics.trajectory_critic import TrajectoryVerdict
from meta_filter.metrics import DEFAULT_SCORE_WEIGHTS, compute_score
from meta_filter.optimizer import (
    OptimizerConfig,
    _candidate_workers,
    _tiny_histogram,
)
from meta_filter.schemas import (
    CandidateEvaluation,
    Decision,
    FilterDecision,
    FilterMetrics,
    Trajectory,
    Turn,
)


def test_optimizer_config_gains_v2_fields():
    cfg = OptimizerConfig()
    # New fields exist with sensible defaults.
    assert cfg.curated_evidence_budget == 6
    assert cfg.rule_critic_llm_recommendations is False
    assert isinstance(cfg.score_weights, dict)
    assert cfg.score_weights == {}


def test_candidates_run_concurrently_unless_told_otherwise():
    """Serial candidate evaluation spends most of an iteration in scorer
    prefetch alone, so the default has to be one worker per candidate."""

    assert OptimizerConfig().candidates_parallel == 0
    assert _candidate_workers(0, 3) == 3
    # An explicit 1 is the escape hatch back to the serial flow.
    assert _candidate_workers(1, 3) == 1
    # Never spin up more workers than there is work.
    assert _candidate_workers(8, 3) == 3
    assert _candidate_workers(0, 1) == 1


def test_compute_score_reacts_to_quality_signals():
    m = FilterMetrics(
        total=100,
        critic_accept_total=50,
        critic_reject_total=50,
        accepted=45,
        rejected=45,
        reviewed=10,
        agree_accept=45,
        agree_reject=45,
        filter_accept_critic_reject=5,
        filter_reject_critic_accept=5,
        critic_reject_recall=0.9,
        critic_accept_retention=0.9,
        disputed_accept_rate=0.1,
        disputed_reject_rate=0.1,
        review_rate=0.1,
        critic_agreement=0.9,
        estimated_filter_cost=50.0,
        complexity_score=10.0,
        rubric_quality=0.5,  # neutral
        rule_quality=0.5,  # neutral
    )
    baseline_score = compute_score(m)

    # The two critic-quality terms are DIAGNOSTIC: both
    # weights are 0, so a perfect rubric_quality buys nothing. The critics
    # still run and still publish these numbers — they are just not selected
    # on, because they grade prose rather than behaviour and a candidate can
    # buy them with rhetoric.
    m.rubric_quality = 1.0
    assert compute_score(m) == baseline_score

    # The term is still summed, so a run that re-enables it gets the old
    # behaviour with no code change. This is the property that makes the
    # zeroing a configuration choice rather than a deletion.
    re_enabled = compute_score(m, weights={**DEFAULT_SCORE_WEIGHTS, "lambda_rubric_quality": 0.30})
    assert abs((re_enabled - baseline_score) - 0.30) < 1e-6


def test_compute_score_clamps_penalties():
    """A penalty way over budget still caps at its λ, and cost charges 0.

    `estimated_filter_cost` measures cache state, not deployment cost: it
    freezes once every later candidate re-reads the same cached cells. Its
    weight is 0, so an absurd cost is free and only
    complexity is charged.
    """

    m = FilterMetrics(
        total=1,
        critic_accept_total=1,
        critic_reject_total=0,
        accepted=1,
        rejected=0,
        reviewed=0,
        agree_accept=1,
        agree_reject=0,
        filter_accept_critic_reject=0,
        filter_reject_critic_accept=0,
        critic_reject_recall=0.0,
        critic_accept_retention=1.0,
        disputed_accept_rate=0.0,
        disputed_reject_rate=0.0,
        review_rate=0.0,
        critic_agreement=1.0,
        estimated_filter_cost=1_000_000.0,  # way over budget
        complexity_score=1_000_000.0,  # way over budget
    )
    s = compute_score(m)
    # agreement 1.0 − 0.15 * min(1, 1e6 / 120) = 1.0 − 0.15 = 0.85
    assert abs(s - 0.85) < 1e-6

    # Only complexity moved it: zeroing the cost that is already free is a
    # no-op, while zeroing complexity recovers the whole 0.15.
    assert compute_score(m, weights={"lambda_cost": 0.0}) == s
    assert abs(compute_score(m, weights={"lambda_complexity": 0.0}) - 1.0) < 1e-6


def test_tiny_histogram_counts_top_actions():
    traj = Trajectory(
        trajectory_id="t",
        task="",
        turns=[
            Turn(role="agent", action="grep", content=""),
            Turn(role="agent", action="grep", content=""),
            Turn(role="agent", action="edit", content=""),
        ],
    )
    hist = _tiny_histogram(traj, top_n=5)
    assert hist == {"grep": 2, "edit": 1}


def test_judge_verdict_survives_curated_pipeline(tmp_path):
    """The judge's misfire label must be readable by the wiring helper via
    `verdict_for`. This exercises the same path _build_curated_evidence
    uses without spinning up the whole optimizer.
    """

    class _StubClient:
        model = "stub"
        calls = 0

        def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
            self.calls += 1
            from meta_filter.llm.base import LLMResponse

            return LLMResponse(text="", prompt_tokens=0, completion_tokens=0)

    backend = TrajectoryCritic(client=_StubClient(), cache_dir=tmp_path)
    # Seed a cache entry directly.
    import json

    backend.cache_path_for("trajX").write_text(
        json.dumps(
            {
                "trajectory_id": "trajX",
                "decision": "REJECT",
                "rationale": "hallucinated file",
                "closest_candidate": "cand_1",
                "misfire": "rubric",
                "misfire_note": "verification anchor is vague",
            }
        )
    )
    v = backend.verdict_for("trajX")
    assert isinstance(v, TrajectoryVerdict)
    assert v.misfire == "rubric"
    assert v.decision is Decision.REJECT


def test_build_curated_evidence_handles_string_fn_fp(tmp_path):
    """Regression: CandidateEvaluation.false_negatives is list[str], not
    list[CounterExample]. The wiring must NOT attempt attribute access on
    those strings (`AttributeError: 'str' object has no attribute
    'trajectory_id'`).
    """

    import os
    from pathlib import Path

    from meta_filter.optimizer import Optimizer, OptimizerConfig
    from meta_filter.proposer import Proposer
    from meta_filter.selector import Selector
    from meta_filter.solver import HeuristicRubricScorer, Solver
    from meta_filter.validator import Validator

    root = Path(__file__).resolve().parents[1]
    proposer = Proposer(client=_MinimalStub(), n_candidates=1)
    opt = Optimizer(
        initial_filter_dir=root / "tests" / "fixtures" / "filters" / "five_criteria",
        train_data=root / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=proposer,
        selector=Selector(),
        solver=Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}),
        config=OptimizerConfig(
            iterations=1,
            candidates_per_iteration=1,
            selection_data=root / "data" / "demo_selection.jsonl",
        ),
    )
    eval_with_fn_fp = CandidateEvaluation(
        candidate_id="c",
        is_baseline=True,
        metrics=_stub_filter_metrics(),
        false_negatives=["trajA", "trajB"],
        false_positives=["trajC"],
    )
    evidence, ids = opt._build_curated_evidence(
        backend=None,  # non-LLMJudge path → uses fn+fp fallback
        active_train=[
            Trajectory(trajectory_id="trajA", task="t"),
            Trajectory(trajectory_id="trajB", task="t"),
            Trajectory(trajectory_id="trajC", task="t"),
        ],
        baseline_evaluation=eval_with_fn_fp,
        baseline_decisions=[],
        iteration=0,
    )
    # No crash, and the fallback path returns exactly the fn+fp trajectory ids.
    assert set(ids) == {"trajA", "trajB", "trajC"}
    assert len(evidence) == 3
    _ = os  # keep import warning-free


def _build_optimizer(tmp_path, *, agentic: bool):
    from pathlib import Path

    from meta_filter.optimizer import Optimizer, OptimizerConfig
    from meta_filter.proposer import Proposer
    from meta_filter.selector import Selector
    from meta_filter.solver import HeuristicRubricScorer, Solver
    from meta_filter.validator import Validator

    root = Path(__file__).resolve().parents[1]
    return Optimizer(
        initial_filter_dir=root / "tests" / "fixtures" / "filters" / "five_criteria",
        train_data=root / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=Proposer(client=_MinimalStub(), n_candidates=1),
        selector=Selector(),
        solver=Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}),
        config=OptimizerConfig(
            iterations=1,
            candidates_per_iteration=1,
            agentic_critics=agentic,
            selection_data=root / "data" / "demo_selection.jsonl",
        ),
        run_id="wiring",
    )


def test_agentic_run_pseudonymizes_every_proposer_surface(tmp_path):
    """One pseudonymizer must cover both surfaces the proposer can reach.

    The run directory (which it reads with tools) and the prompt (which it is
    handed). Two independent instances would give the same trajectory two
    different pseudonyms and break the cross-iteration signal the proposer
    legitimately uses.
    """

    opt = _build_optimizer(tmp_path, agentic=True)
    assert opt.pseudonymizer is not None
    assert opt.store.pseudonymizer is opt.pseudonymizer
    assert opt.proposer.pseudonymizer is opt.pseudonymizer


def test_idmap_is_written_outside_the_run_directory(tmp_path):
    """The map is the key the proposer is not allowed to have."""

    opt = _build_optimizer(tmp_path, agentic=True)
    idmap = tmp_path / "runs" / "_idmap" / "wiring.json"
    assert idmap.exists()
    assert opt.store.run_dir not in idmap.parents


def test_heldout_sink_lands_outside_the_run_directory(tmp_path):
    opt = _build_optimizer(tmp_path, agentic=True)
    sink = opt.store.heldout_sink
    assert sink == tmp_path / "runs" / "_heldout" / "wiring.jsonl"
    assert opt.store.run_dir not in sink.parents


def test_static_run_keeps_real_ids(tmp_path):
    opt = _build_optimizer(tmp_path, agentic=False)
    assert opt.pseudonymizer is None
    assert opt.store.pseudonymizer is None
    assert opt.proposer.pseudonymizer is None
    assert opt.store.heldout_sink is None
    assert not (tmp_path / "runs" / "_idmap").exists()


class _MinimalStub:
    model = "stub"

    def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        from meta_filter.llm.base import LLMResponse

        return LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0)


def _stub_filter_metrics() -> FilterMetrics:
    return FilterMetrics(
        total=1,
        critic_accept_total=1,
        critic_reject_total=0,
        accepted=1,
        rejected=0,
        reviewed=0,
        agree_accept=1,
        agree_reject=0,
        filter_accept_critic_reject=0,
        filter_reject_critic_accept=0,
        critic_reject_recall=0.0,
        critic_accept_retention=1.0,
        disputed_accept_rate=0.0,
        disputed_reject_rate=0.0,
        review_rate=0.0,
        critic_agreement=1.0,
    )


def test_candidate_evaluation_carries_v2_fields():
    """CandidateEvaluation must round-trip the v2 quality signals."""

    metrics = FilterMetrics(
        total=1,
        critic_accept_total=1,
        critic_reject_total=0,
        accepted=1,
        rejected=0,
        reviewed=0,
        agree_accept=1,
        agree_reject=0,
        filter_accept_critic_reject=0,
        filter_reject_critic_accept=0,
        critic_reject_recall=0.0,
        critic_accept_retention=1.0,
        disputed_accept_rate=0.0,
        disputed_reject_rate=0.0,
        review_rate=0.0,
        critic_agreement=1.0,
        rubric_quality=0.8,
        rule_quality=0.7,
        score=0.99,
    )
    _ = FilterDecision(
        trajectory_id="x",
        decision=Decision.ACCEPT,
    )
    ev = CandidateEvaluation(candidate_id="c", is_baseline=True, metrics=metrics)
    payload = ev.model_dump()
    assert payload["metrics"]["rubric_quality"] == 0.8
    assert payload["metrics"]["rule_quality"] == 0.7
    assert payload["metrics"]["score"] == 0.99
