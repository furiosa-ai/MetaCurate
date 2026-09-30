"""The cuts a ranking filter can make (`scripts.build_ranking_subset.choose_rows`).

These tests pin the parts where getting it wrong is silent rather than loud: a
tie-break that is not reproducible, and an accept-all subset that quietly
becomes a top-1000.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.build_ranking_subset import choose_rows


@dataclass
class Scored:
    rank_score: float
    decision: str = "REJECT"


def pool(*rows: tuple[str, float, str]) -> dict[str, Scored]:
    return {tid: Scored(score, decision) for tid, score, decision in rows}


def test_top_cut_takes_the_highest_scores_in_order():
    scored = pool(("a", 10.0, "ACCEPT"), ("b", 30.0, "REJECT"), ("c", 20.0, "REVIEW"))

    order, kept, lowest, bucket_kept, bucket_total = choose_rows(
        scored, seed=42, top=2, accept_all=False
    )

    assert kept == ["b", "c"]
    assert order == ["b", "c", "a"]
    assert lowest == 20.0
    assert (bucket_kept, bucket_total) == (1, 1)


def test_a_tie_at_the_boundary_is_broken_by_the_seed_and_reported():
    """Four rows tied at the cut, two seats. Which two is arbitrary; that it is
    the SAME two on a re-run is not, and the manifest has to say a coin was
    flipped at all."""

    scored = pool(*[(f"t{i}", 50.0, "REJECT") for i in range(4)], ("high", 99.0, "REJECT"))

    _, kept, lowest, bucket_kept, bucket_total = choose_rows(
        scored, seed=7, top=3, accept_all=False
    )
    _, again, _, _, _ = choose_rows(scored, seed=7, top=3, accept_all=False)

    assert kept[0] == "high"
    assert lowest == 50.0
    assert (bucket_kept, bucket_total) == (2, 4), "2 of 4 tied rows kept — say so"
    assert kept == again, "the same seed must produce the same subset"
    assert len(set(kept)) == 3, "no row is kept twice"


def test_accept_all_keeps_every_accept_and_only_accepts():
    scored = pool(
        ("hi", 90.0, "ACCEPT"),
        ("mid", 70.0, "REVIEW"),
        ("also", 65.0, "ACCEPT"),
        ("lo", 10.0, "REJECT"),
    )

    _, kept, lowest, bucket_kept, bucket_total = choose_rows(
        scored, seed=42, top=1000, accept_all=True
    )

    assert kept == ["hi", "also"], "score order, ACCEPTs only"
    assert lowest == 65.0
    # The seed decides nothing here: membership is the policy's decision, not a
    # rank, so reporting a boundary bucket would invent a coin flip.
    assert (bucket_kept, bucket_total) == (0, 0)


def test_accept_all_ignores_the_top_default():
    """The failure this guards: --accept-all silently truncating at 1000.

    `--top` carries a default, so an accept-all subset larger than it would come
    out at exactly 1,000 rows and look like a deliberate budget.
    """

    scored = pool(*[(f"t{i}", float(i), "ACCEPT") for i in range(1500)])

    _, kept, _, _, _ = choose_rows(scored, seed=42, top=1000, accept_all=True)

    assert len(kept) == 1500


def test_accept_all_on_a_filter_that_accepts_nothing_refuses():
    """Not an empty subset: a full-pool scan costs tens of thousands of calls,
    and shipping a zero-row train.jsonl would fail much later, at training."""

    scored = pool(("a", 5.0, "REJECT"), ("b", 4.0, "REVIEW"))

    with pytest.raises(SystemExit, match="accepted 0 of 2"):
        choose_rows(scored, seed=42, top=1000, accept_all=True, whose="filters/x")


def test_a_top_cut_larger_than_the_pool_refuses():
    scored = pool(("a", 5.0, "ACCEPT"))

    with pytest.raises(SystemExit, match="only 1 scored rows"):
        choose_rows(scored, seed=42, top=1000, accept_all=False)
