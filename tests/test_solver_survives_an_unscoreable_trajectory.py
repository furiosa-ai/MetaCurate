"""One trajectory the model refuses must not end the run.

A trajectory can contain a test fixture holding a literal `BEGIN RSA PRIVATE
KEY`, and a model's safety layer answers it with `stop_reason: "refusal"`,
every time — this is not a flake and a retry cannot help. Uncaught, it ends
the run in iteration 0.

What makes it hard to see: the parallel prefetch catches the same exception
and logs `scorer failed for <id>`, so the run looks like it is tolerating the
problem — while the sequential pass scores the same trajectory again and
lets the exception out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meta_filter.core.filter_loader import load_filter
from meta_filter.core.schemas import Decision, Trajectory, Turn
from meta_filter.core.solver import Solver

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "tests" / "fixtures" / "filters" / "five_criteria"


class _RefusingScorer:
    """Refuses one trajectory by id and scores every other one."""

    def __init__(self, refuse: str) -> None:
        self.refuse = refuse
        self.seen: list[str] = []

    def score(self, trajectory, criteria, rubric_md):
        self.seen.append(trajectory.trajectory_id)
        if trajectory.trajectory_id == self.refuse:
            raise RuntimeError(
                'claude headless exit 1: {"stop_reason":"refusal","result":'
                '"API Error: Sonnet 5\'s safeguards flagged this message."}'
            )
        from meta_filter.core.solver import ScorerCost

        return {c: 3.0 for c in criteria}, ["ok"], ScorerCost(llm_calls=1, llm_tokens=10)


def _trajectories() -> list[Trajectory]:
    return [
        Trajectory(
            trajectory_id=tid,
            task="fix the bug",
            turns=[Turn(role="assistant", content="patched it")],
        )
        for tid in ("fine_a", "bridgecrewio__checkov-755::att0", "fine_b")
    ]


def test_a_refused_trajectory_is_reviewed_and_the_rest_are_still_decided():
    scorer = _RefusingScorer("bridgecrewio__checkov-755::att0")
    solver = Solver(rubric_scorer=scorer)

    decisions = solver.run(load_filter(SEED), _trajectories())

    assert len(decisions) == 3, "the run continues past the refusal"
    assert [d.trajectory_id for d in decisions] == [
        "fine_a",
        "bridgecrewio__checkov-755::att0",
        "fine_b",
    ]

    refused = decisions[1]
    assert refused.decision is Decision.REVIEW
    assert refused.rubric_scores == {}, (
        "a default score would make an unreadable trajectory look like a bad one"
    )
    assert "scorer unavailable" in refused.evidence[0]
    assert "refusal" in refused.evidence[0]


def test_the_refusal_costs_nothing_it_did_not_spend():
    """A call that returned no scores must not be billed as if it had."""

    solver = Solver(rubric_scorer=_RefusingScorer("bridgecrewio__checkov-755::att0"))
    decisions = solver.run(load_filter(SEED), _trajectories())

    refused = next(d for d in decisions if d.decision is Decision.REVIEW)
    assert refused.llm_calls == 0
    assert refused.llm_tokens == 0


class _RefusingAll:
    """A candidate whose own rubric trips the safeguard: nothing scores."""

    def __init__(self, refuse_all: bool = True) -> None:
        self.refuse_all = refuse_all

    def score(self, trajectory, criteria, rubric_md):
        raise RuntimeError('{"stop_reason":"refusal"}')


def _pool(n: int) -> list[Trajectory]:
    return [
        Trajectory(
            trajectory_id=f"t{i}", task="fix it", turns=[Turn(role="assistant", content="patched")]
        )
        for i in range(n)
    ]


def test_a_filter_that_scores_nothing_is_an_error_not_a_bad_score():
    """The proposer writes the rubric, so a candidate whose rubric trips the
    safeguard has every call refused. Routing all of them to review would give
    it a terrible score for a reason unrelated to its quality, and nothing in
    the metrics would say why."""

    solver = Solver(rubric_scorer=_RefusingAll())

    with pytest.raises(RuntimeError, match="do not describe its quality"):
        solver.run(load_filter(SEED), _pool(20))


def test_one_refusal_in_a_large_pool_is_reported_quietly():
    """A single refused trajectory is a constant offset across candidates —
    it cannot change which one wins, so it is not worth an ERROR."""

    solver = Solver(rubric_scorer=_RefusingScorer("t7"))
    decisions = solver.run(load_filter(SEED), _pool(100))

    assert len(decisions) == 100
    assert sum(1 for d in decisions if d.decision is Decision.REVIEW) == 1


def test_the_warning_threshold_is_where_the_score_starts_to_move(caplog):
    """`lambda_review` is 2.0, so 5% unscoreable shifts the score by 0.10 —
    bigger than the gaps that usually separate candidates."""

    import logging

    class _RefuseFirstN:
        def __init__(self, n):
            self.n, self.seen = n, 0

        def score(self, trajectory, criteria, rubric_md):
            self.seen += 1
            if self.seen <= self.n:
                raise RuntimeError('{"stop_reason":"refusal"}')
            from meta_filter.core.solver import ScorerCost

            return {c: 1.0 for c in criteria}, [], ScorerCost()

    with caplog.at_level(logging.INFO):
        Solver(rubric_scorer=_RefuseFirstN(6)).run(load_filter(SEED), _pool(100))

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "6 of 100 is past the 5% threshold and has to be loud"
    assert "unscoreable" in errors[0].getMessage()
