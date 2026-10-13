"""A pairwise verdict that depends on which slot a trajectory sat in.

The pair judge decides the truth every candidate is scored against, and it
was asked exactly once per pair. An LLM comparing two long transcripts has a
position preference, so a single-order verdict is partly a measurement of
presentation order — and that noise entered the objective as signal, with
nothing anywhere recording which slot each side had been shown in.

Every pair is now asked in both orders. Agreement resolves it; anything else
is a TIE that leaves the truth set rather than scoring half a point.
"""

from __future__ import annotations

import json

from meta_filter.core.schemas import Trajectory, Turn
from meta_filter.gate.pair_judge import (
    ORDER_FORWARD,
    ORDER_REVERSED,
    PairJudge,
)
from meta_filter.gate.ranking import concordance_detail
from meta_filter.visibility.view_policy import pair_view_signature


class _PositionBiasedClient:
    """Always names slot A the winner — the pure position-bias judge."""

    model = "stub"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt, **_kw):
        from meta_filter.llm.base import LLMResponse

        self.calls.append(prompt)
        return LLMResponse(
            text='{"winner": "A", "rationale": "whichever came first"}',
            prompt_tokens=0,
            completion_tokens=0,
        )


class _ConsistentClient:
    """Always names trajectory `winner`, whichever slot it is in."""

    model = "stub"

    def __init__(self, winner: str) -> None:
        self.winner = winner
        self.calls: list[str] = []

    def generate(self, prompt, **_kw):
        from meta_filter.llm.base import LLMResponse

        self.calls.append(prompt)
        # The prompt names trajectory A on its `=== TRAJECTORY A (id) ===` line.
        first = prompt.split("=== TRAJECTORY A (", 1)[1].split(")", 1)[0]
        letter = "A" if first == self.winner else "B"
        return LLMResponse(
            text=f'{{"winner": "{letter}", "rationale": "it verified the fix"}}',
            prompt_tokens=0,
            completion_tokens=0,
        )


def _traj(tid: str) -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task=f"task for {tid}",
        turns=[Turn(role="agent", action="edit", content="patch")],
        metadata={},
    )


def _pair():
    return _traj("t::a"), _traj("t::b")


def test_a_verdict_that_flips_with_the_order_is_a_tie(tmp_path):
    a, b = _pair()
    judge = PairJudge(client=_PositionBiasedClient(), cache_dir=tmp_path)

    winner, status = judge.judge_both_orders(a, b)
    assert winner is None
    assert status == "tie"
    # And the compatibility surface agrees: no winner means no truth pair.
    assert judge.judge(a, b) is None


def test_an_order_independent_verdict_resolves(tmp_path):
    a, b = _pair()
    judge = PairJudge(client=_ConsistentClient("t::b"), cache_dir=tmp_path)

    winner, status = judge.judge_both_orders(a, b)
    assert (winner, status) == ("t::b", "resolved")


def test_each_order_is_cached_under_its_own_key(tmp_path):
    a, b = _pair()
    client = _ConsistentClient("t::b")
    judge = PairJudge(client=client, cache_dir=tmp_path)
    judge.judge_both_orders(a, b)

    assert len(client.calls) == 2, "both presentations have to be paid for"
    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert len(files) == 2, files
    assert any(f"__ord_{ORDER_FORWARD}" in name for name in files)
    assert any(f"__ord_{ORDER_REVERSED}" in name for name in files)

    # Each record says which trajectory sat in which slot — the fact whose
    # absence made the old cache unauditable.
    slots = {json.loads(p.read_text())["shown_as_a"] for p in tmp_path.glob("*.json")}
    assert slots == {"t::a", "t::b"}

    # Re-asking is free.
    judge.judge_both_orders(a, b)
    assert len(client.calls) == 2


def _write_single_order(judge: PairJudge, a, b, winner: str) -> None:
    """A single-order cache file: a winner, and no slot."""

    judge._cache_path(a.trajectory_id, b.trajectory_id, pair_view_signature(a, b)).write_text(
        json.dumps(
            {
                "pair": sorted((a.trajectory_id, b.trajectory_id)),
                "winner_id": winner,
                "letter": "B" if winner == b.trajectory_id else "A",
                "raw": '{"winner": "B", "rationale": "it verified the fix"}',
            }
        ),
        encoding="utf-8",
    )


def test_a_single_order_record_cannot_manufacture_agreement(tmp_path):
    """The regression: a one-order verdict standing in for the forward pass.

    Nothing records which slot a single-order verdict was asked in, and it is not
    recoverable — consensus and audit pairs went to the judge in the
    candidates' score order (`ranking.truth_pairs`), so a single-order record is as
    likely to be a reverse presentation as a forward one. Seeding the forward
    slot with one lets a position-biased judge agree with ITSELF across two
    identical presentations and report a resolved winner, which is precisely
    the measurement the two-order rule exists to refuse.

    Here the single-order record names `t::b`, and a judge that always picks slot A
    names `t::b` again when `t::b` is shown first. Reusing the record gives
    "resolved"; asking both orders gives TIE, which is the truth.
    """

    a, b = _pair()
    client = _PositionBiasedClient()
    judge = PairJudge(client=client, cache_dir=tmp_path)
    _write_single_order(judge, a, b, winner="t::b")

    winner, status = judge.judge_both_orders(a, b)
    assert (winner, status) == (None, "tie")
    assert len(client.calls) == 2, "both orders have to be asked afresh"


def test_a_single_order_record_still_yields_its_prose(tmp_path):
    """Re-judging the verdict does not throw away the judge's rationale."""

    a, b = _pair()
    judge = PairJudge(client=_ConsistentClient("t::b"), cache_dir=tmp_path)
    _write_single_order(judge, a, b, winner="t::b")

    assert judge.rationale_for("t::a", "t::b", pair_view_signature(a, b)) == "it verified the fix"


def test_a_record_from_the_other_presentation_is_not_reused(tmp_path):
    """The general rule behind it: slot A has to match what is being asked."""

    a, b = _pair()
    client = _ConsistentClient("t::b")
    judge = PairJudge(client=client, cache_dir=tmp_path)
    forward = judge._cache_path("t::a", "t::b", pair_view_signature(a, b), ORDER_FORWARD)
    forward.write_text(
        json.dumps({"shown_as_a": "t::b", "winner_id": "t::b", "letter": "A"}),
        encoding="utf-8",
    )

    judge.judge_both_orders(a, b)
    assert len(client.calls) == 2, "a mislabelled record must not be trusted"


# --- an unparseable answer is a failure, not a verdict -----------------------


class _UnparseableClient:
    """Answers, but never in the contract's shape."""

    model = "stub"

    def __init__(self, good_after: int = 10**6) -> None:
        self.good_after = good_after
        self.calls = 0

    def generate(self, prompt, **_kw):
        from meta_filter.llm.base import LLMResponse

        self.calls += 1
        if self.calls <= self.good_after:
            text = "I am afraid I cannot compare these."
        else:
            # Order-independent once it answers at all, so a resolved verdict
            # here is evidence about the retry and not about position bias.
            first = prompt.split("=== TRAJECTORY A (", 1)[1].split(")", 1)[0]
            text = '{"winner": "%s", "rationale": "fine"}' % ("A" if first == "t::b" else "B")
        return LLMResponse(text=text, prompt_tokens=0, completion_tokens=0)


def test_an_unparseable_answer_is_retried_once(tmp_path):
    a, b = _pair()
    client = _UnparseableClient(good_after=1)
    judge = PairJudge(client=client, cache_dir=tmp_path)

    winner, status = judge.judge_both_orders(a, b)
    # Order one needed two attempts, order two needed one.
    assert client.calls == 3
    assert (winner, status) == ("t::b", "resolved")
    assert judge.parse_failures == 0


def test_a_persistently_unparseable_pair_is_not_cached(tmp_path):
    """Caching it would freeze a transport failure into a permanent verdict.

    Every later run sharing the cache would read the record, find no letter,
    and call the pair unresolved without ever asking again.
    """

    a, b = _pair()
    client = _UnparseableClient()
    judge = PairJudge(client=client, cache_dir=tmp_path)

    winner, status = judge.judge_both_orders(a, b)
    assert (winner, status) == (None, "unresolved")
    assert list(tmp_path.glob("*.json")) == []
    assert judge.parse_failures >= 1


def test_resolve_pairs_reports_ties_separately(tmp_path):
    a, b = _pair()
    judge = PairJudge(client=_PositionBiasedClient(), cache_dir=tmp_path)

    detail = judge.resolve_pairs_detail([("t::a", "t::b")], {"t::a": a, "t::b": b}, max_workers=1)
    assert detail.resolved == []
    assert detail.tied == [("t::a", "t::b")]
    assert detail.tie_share == 1.0
    # The list form drops everything that is not a resolved pair.
    assert judge.resolve_pairs([("t::a", "t::b")], {"t::a": a, "t::b": b}) == []


def test_concordance_drops_tied_pairs_rather_than_halving_them():
    """A pair with no verdict is not a pair the candidate half-answered."""

    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    truth = [("a", "b"), ("b", "c")]

    both = concordance_detail(scores, truth, unscored="loss")
    assert (both.concordance, both.n_offered, both.n_tied) == (1.0, 2, 0)

    # The judge tied (b, c). It leaves the exam entirely: one pair offered,
    # not two pairs with one scored at 0.5.
    one = concordance_detail(scores, truth, unscored="loss", ties=[("c", "b")])
    assert (one.concordance, one.n_offered, one.n_tied) == (1.0, 1, 1)

    # And a candidate that gets the surviving pair wrong is not rescued by it.
    inverted = concordance_detail(
        {"a": 1.0, "b": 2.0, "c": 3.0}, truth, unscored="loss", ties=[("c", "b")]
    )
    assert inverted.concordance == 0.0
