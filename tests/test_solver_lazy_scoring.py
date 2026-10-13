"""Lazy rubric scoring: skip the LLM where the rules already decided.

For cascade-style policies most rows' decisions do not depend on rubric
scores at all — a fired rule decides. The solver probes `decide()` with
adversarial score dicts (+2 everywhere, -2 everywhere, and no scores); if all
three agree and none raises, the decision is provably score-independent and
LLM scoring is skipped. The probe must never change what a full run would
have decided — only what it costs.
"""

from __future__ import annotations

import logging

import pytest

from meta_filter.core.filter_loader import load_filter
from meta_filter.core.schemas import Decision, Trajectory, Turn
from meta_filter.core.solver import ScorerCost, Solver

RUBRIC = "# Rubric\n\n## Task Grounding\n\n## Error Recovery\n"

RULES = '''\
def rule_flagged(trajectory):
    """Fires on trajectories whose task carries the marker word."""

    if "bad" in (trajectory.task or ""):
        return {"name": "flagged", "verdict": "REJECT", "evidence": ["task flagged"]}
    return None
'''

# Never reads rubric_scores: every row is rule-decided.
RULE_ONLY_POLICY = """\
RUBRIC_CRITERIA = ["Task Grounding", "Error Recovery"]


def decide(trajectory, rubric_scores, triggered_rules):
    if any(r.get("verdict") == "REJECT" for r in triggered_rules or []):
        return {
            "decision": "REJECT",
            "confidence": 0.9,
            "reasoning_summary": "a deterministic rule fired",
            "evidence": ["rule fired"],
        }
    return {
        "decision": "ACCEPT",
        "confidence": 0.8,
        "reasoning_summary": "no rule fired",
        "evidence": [],
    }
"""

# Reads rubric_scores on every row: nothing is score-independent.
RUBRIC_POLICY = """\
RUBRIC_CRITERIA = ["Task Grounding", "Error Recovery"]


def decide(trajectory, rubric_scores, triggered_rules):
    total = sum(rubric_scores.values()) if rubric_scores else 0.0
    return {
        "decision": "ACCEPT" if total > 0 else "REJECT",
        "confidence": 0.5,
        "reasoning_summary": "rubric decided",
        "evidence": [],
    }
"""

# Cascade: fired rules decide immediately, the rubric decides the rest.
CASCADE_POLICY = """\
RUBRIC_CRITERIA = ["Task Grounding", "Error Recovery"]


def decide(trajectory, rubric_scores, triggered_rules):
    if any(r.get("verdict") == "REJECT" for r in triggered_rules or []):
        return {
            "decision": "REJECT",
            "confidence": 0.9,
            "reasoning_summary": "a deterministic rule fired",
            "evidence": ["rule fired"],
        }
    total = sum(rubric_scores.values()) if rubric_scores else 0.0
    return {
        "decision": "ACCEPT" if total > 0 else "REVIEW",
        "confidence": 0.5,
        "reasoning_summary": "rubric decided",
        "evidence": [],
    }
"""

# Indexes into rubric_scores directly: the empty-dict probe raises KeyError.
FRAGILE_POLICY = """\
RUBRIC_CRITERIA = ["Task Grounding", "Error Recovery"]


def decide(trajectory, rubric_scores, triggered_rules):
    grounding = rubric_scores["Task Grounding"]
    return {
        "decision": "ACCEPT" if grounding > 0 else "REJECT",
        "confidence": 0.5,
        "reasoning_summary": "read one criterion directly",
        "evidence": [],
    }
"""


class CountingScorer:
    """Deterministic scorer that records every score and prefetch call."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.scored: list[str] = []
        self.prefetched: list[list[str]] = []

    def score(self, trajectory, criteria, rubric_md):
        self.scored.append(trajectory.trajectory_id)
        return (
            {c: self.value for c in criteria},
            ["scored"],
            ScorerCost(llm_calls=1, llm_tokens=10),
        )

    def prefetch_parallel(self, trajectories, criteria, rubric_md):
        self.prefetched.append([t.trajectory_id for t in trajectories])
        return 0


class RefusingScorer:
    """Refuses everything, the way a safeguard-tripping rubric does."""

    def score(self, trajectory, criteria, rubric_md):
        raise RuntimeError('{"stop_reason":"refusal"}')


def _write_filter(root, policy, rules=RULES, rubric=RUBRIC):
    root.mkdir(parents=True, exist_ok=True)
    (root / "rubric.md").write_text(rubric, encoding="utf-8")
    (root / "rules.py").write_text(rules, encoding="utf-8")
    (root / "policy.py").write_text(policy, encoding="utf-8")
    return load_filter(root)


def _traj(tid: str, task: str = "fix the bug") -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task=task,
        turns=[Turn(role="assistant", content="patched it")],
    )


def test_rule_decided_rows_skip_the_scorer_and_keep_their_decisions(tmp_path):
    """A policy that never reads scores: no row needs the LLM, and the
    decisions are byte-for-byte what a full (lazy-off) run produces."""

    loaded = _write_filter(tmp_path / "f", RULE_ONLY_POLICY)
    pool = [_traj("clean_a"), _traj("flagged", task="bad task"), _traj("clean_b")]

    lazy_scorer = CountingScorer()
    lazy = Solver(rubric_scorer=lazy_scorer).run(loaded, pool)

    full_scorer = CountingScorer()
    full = Solver(rubric_scorer=full_scorer, lazy_scoring=False).run(loaded, pool)

    assert lazy_scorer.scored == [], "no row's decision depends on scores"
    assert lazy_scorer.prefetched == [], "nothing to score, nothing to prefetch"
    assert [(d.trajectory_id, d.decision) for d in lazy] == [
        (d.trajectory_id, d.decision) for d in full
    ]
    assert [d.decision for d in lazy] == [Decision.ACCEPT, Decision.REJECT, Decision.ACCEPT]


def test_skipped_rows_carry_empty_scores_a_marker_and_zero_cost(tmp_path):
    loaded = _write_filter(tmp_path / "f", RULE_ONLY_POLICY)

    decisions = Solver(rubric_scorer=CountingScorer()).run(loaded, [_traj("t0")])

    (record,) = decisions
    assert record.rubric_scores == {}
    assert any("lazy scoring" in e for e in record.evidence)
    assert record.llm_calls == 0
    assert record.llm_tokens == 0
    assert record.triggered_rules == [], "rule bookkeeping is unchanged"


def test_a_rubric_driven_policy_scores_every_row(tmp_path):
    loaded = _write_filter(tmp_path / "f", RUBRIC_POLICY)
    pool = [_traj("t0"), _traj("t1"), _traj("t2")]
    scorer = CountingScorer()

    decisions = Solver(rubric_scorer=scorer).run(loaded, pool)

    assert scorer.scored == ["t0", "t1", "t2"]
    assert all(d.decision is Decision.ACCEPT for d in decisions)
    assert all(d.rubric_scores for d in decisions)
    assert not any("lazy scoring" in e for d in decisions for e in d.evidence)


def test_a_policy_that_raises_on_empty_scores_falls_back_to_scoring(tmp_path):
    """The empty-dict probe raising is contained: the row scores normally."""

    loaded = _write_filter(tmp_path / "f", FRAGILE_POLICY)
    scorer = CountingScorer(value=2.0)

    decisions = Solver(rubric_scorer=scorer).run(loaded, [_traj("t0"), _traj("t1")])

    assert scorer.scored == ["t0", "t1"]
    assert [d.decision for d in decisions] == [Decision.ACCEPT, Decision.ACCEPT]


def test_the_flag_off_scores_every_row(tmp_path):
    loaded = _write_filter(tmp_path / "f", RULE_ONLY_POLICY)
    scorer = CountingScorer()

    Solver(rubric_scorer=scorer, lazy_scoring=False).run(loaded, [_traj("t0"), _traj("t1")])

    assert scorer.scored == ["t0", "t1"]


def test_prefetch_receives_only_the_rows_that_need_scoring(tmp_path):
    """Cascade policy: flagged rows are rule-decided, the rest go through the
    rubric — and only the rest reach prefetch and the scorer."""

    loaded = _write_filter(tmp_path / "f", CASCADE_POLICY)
    pool = [
        _traj("flagged_a", task="bad task"),
        _traj("clean_a"),
        _traj("flagged_b", task="another bad one"),
        _traj("clean_b"),
    ]
    scorer = CountingScorer()

    decisions = Solver(rubric_scorer=scorer).run(loaded, pool)

    assert scorer.prefetched == [["clean_a", "clean_b"]]
    assert scorer.scored == ["clean_a", "clean_b"]
    by_id = {d.trajectory_id: d for d in decisions}
    assert by_id["flagged_a"].decision is Decision.REJECT
    assert by_id["clean_a"].decision is Decision.ACCEPT


def test_the_run_logs_how_many_rows_skipped_the_rubric(tmp_path, caplog):
    loaded = _write_filter(tmp_path / "f", CASCADE_POLICY)
    pool = [_traj("flagged", task="bad task"), _traj("clean")]

    with caplog.at_level(logging.INFO):
        Solver(rubric_scorer=CountingScorer()).run(loaded, pool)

    assert any(
        "lazy scoring: 1 of 2 rows decided without the rubric" in r.getMessage()
        for r in caplog.records
    )


def test_unscoreable_fraction_is_over_rows_that_needed_scoring(tmp_path):
    """10 rule-decided rows must not dilute a total scoring outage on the one
    row that actually needed the rubric: 1/1 refused is fatal, not 1/11."""

    loaded = _write_filter(tmp_path / "f", CASCADE_POLICY)
    pool = [_traj(f"flagged_{i}", task="bad task") for i in range(10)] + [_traj("clean")]

    with pytest.raises(RuntimeError, match="do not describe its quality"):
        Solver(rubric_scorer=RefusingScorer()).run(loaded, pool)
