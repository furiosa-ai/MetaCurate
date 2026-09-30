"""Shared fixture: renders a proposer prompt from fully fixed inputs."""

from pathlib import Path

from meta_filter.filter_loader import load_filter, materialize_candidate
from meta_filter.proposer import Proposer
from meta_filter.schemas import (
    CounterExample,
    Decision,
    FilterDecision,
    FilterMetrics,
    Trajectory,
    Turn,
)

RUBRIC = "# Rubric\n\n## Criterion: Task Grounding\n+2 names the defect\n"
RULES = "def rule_one(t):\n    return None\n"
POLICY = (
    "RUBRIC_CRITERIA = ['Task Grounding']\n\n\n"
    "def decide(t, s, r):\n    return {'decision': 'REVIEW', 'score': 50.0}\n"
)


def a_filter(tmp: Path):
    d = tmp / "filter"
    materialize_candidate(d, rubric_md=RUBRIC, rules_py=RULES, policy_py=POLICY)
    return load_filter(d)


def a_trajectory(tid: str) -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task=f"fix the parser in {tid}",
        turns=[
            Turn(role="system", action=None, content="you are a shell agent"),
            Turn(role="assistant", action="grep", content="grep -rn parse ."),
            Turn(role="assistant", action="sed", content="sed -i s/a/b/ p.py"),
        ],
        metadata={
            "api_calls": 12,
            "n_messages": 30,
            "submission_bytes": 900,
            "exit_status": "Submitted",
            "instance_id": tid,
            "resolved": True,
            "f2p_pass": True,
            "p2p_pass": True,
        },
    )


def a_decision(tid: str) -> FilterDecision:
    return FilterDecision(
        trajectory_id=tid,
        decision=Decision.ACCEPT,
        confidence=0.8,
        rubric_scores={"Task Grounding": 2.0},
        triggered_rules=["rule_one"],
        reasoning_summary="score 71.0 from rubric mean +2.0",
    )


def a_pair(*, over: str, under: str, gap: float, judge: str):
    """The two records `_pair_misorder_examples` writes for one inversion."""

    fn = CounterExample(
        trajectory_id=over,
        source_iteration=1,
        filter_version="v1",
        critic_decision=Decision.REJECT,
        predicted_decision=Decision.ACCEPT,
        failure_type="false_negative",
        failure_subtype=f"pair_misorder_vs:{under}",
        severity=2,
        confidence=0.9,
        description=(
            f"The pair judge ranked {under} above {over}; this filter scored "
            f'{over} ABOVE {under} by {gap:.2f}. Judge: "{judge}"'
        ),
        why_filter_failed=(
            f"scored {over} ABOVE {under} against the judge's order (inversion gap {gap:.2f})"
        ),
        raw_trajectory=a_trajectory(over),
        solver_decision=a_decision(over),
    )
    fp = CounterExample(
        trajectory_id=under,
        source_iteration=1,
        filter_version="v1",
        critic_decision=Decision.ACCEPT,
        predicted_decision=Decision.REJECT,
        failure_type="false_positive",
        failure_subtype=f"pair_misorder_vs:{over}",
        severity=2,
        confidence=0.7,
        description=(
            f"The pair judge ranked {under} above {over}; this filter scored "
            f'{under} BELOW {over} by {gap:.2f}. Judge: "{judge}"'
        ),
        why_filter_failed=(
            f"scored {under} BELOW {over} against the judge's order (inversion gap {gap:.2f})"
        ),
        raw_trajectory=a_trajectory(under),
        solver_decision=a_decision(under),
    )
    return fn, fp


def stub_metrics() -> FilterMetrics:
    return FilterMetrics(
        total=50,
        critic_accept_total=0,
        critic_reject_total=0,
        accepted=15,
        rejected=35,
        reviewed=0,
        agree_accept=0,
        agree_reject=0,
        filter_accept_critic_reject=0,
        filter_reject_critic_accept=0,
        critic_reject_recall=0.0,
        critic_accept_retention=0.0,
        disputed_accept_rate=0.0,
        disputed_reject_rate=0.0,
        review_rate=0.0,
        complexity_score=17.5,
    )


REPORTS = {
    "rubric_critic_report": {
        "rubric_quality": 0.42,
        "issues": [
            {
                "criterion": "Task Grounding",
                "problem": "anchor is vague",
                "suggested_edit": "require a named symbol",
            }
        ],
        "missing_criteria": ["regression risk"],
    },
    "past_diagnoses": [
        {
            "iteration": 0,
            "diagnosis": ["verification is unscored"],
            "candidate_notes": ["added a verification criterion"],
        }
    ],
    "quality_trajectory": [
        {
            "iteration": 0,
            "score": 0.715,
            "rubric_quality": 0.78,
            "rule_quality": 0.0,
            "critic_agreement": 0.742,
            "review_rate": 1.0,
        }
    ],
    "past_winners": [
        {
            "iteration": 0,
            "winner": "iter000_00_gold_alignment",
            "reason": "selected iter000_00_gold_alignment — score 0.715 > "
            "baseline 0.620 (loss 26.076 vs 2.280). FN=0, FP=0, "
            "review=50, complexity=21.51, cost=0.00",
            "winner_notes": "scored gold alignment",
        }
    ],
}


def render(
    tmp: Path,
    *,
    iteration: int = 1,
    pairs: bool = True,
    proposer_attrs: dict | None = None,
    **extra,
) -> str:
    """The prompt a bare `Proposer` renders; `proposer_attrs` are what the
    optimizer would have set on it (`candidates_scored`, `initial_complexity`)."""

    proposer = Proposer(n_candidates=3)
    for name, value in (proposer_attrs or {}).items():
        setattr(proposer, name, value)
    fns: list = []
    fps: list = []
    if pairs:
        for over, under, gap, why in (
            ("traj_over_a", "traj_under_a", 19.0, "B verified the fix and A did not"),
            ("traj_over_b", "traj_under_b", 9.0, "B's patch engages the root cause"),
        ):
            fn, fp = a_pair(over=over, under=under, gap=gap, judge=why)
            fns.append(fn)
            fps.append(fp)
    kwargs = dict(
        loaded_filter=a_filter(tmp),
        iteration=iteration,
        current_metrics=stub_metrics(),
        recent_false_negatives=fns,
        recent_false_positives=fps,
        **REPORTS,
    )
    kwargs.update(extra)
    return proposer._build_prompt(**kwargs)
