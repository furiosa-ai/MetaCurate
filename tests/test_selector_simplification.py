"""The simplification lane of `Selector.choose_paired`, the no-op
and deletion exemptions in `gate_candidates`, and the simplification re-draw
confirmation.

Same cell-code fixtures as `test_selector_paired_gate.py`: the baseline is
right on the first half of forty pairs, and a candidate is described by how
many baseline-wrong pairs it fixes and baseline-right pairs it breaks.

What reaches the lane is any candidate smaller than its baseline in every
complexity component, not only a dead-axis removal recognised by a diff test.

The confirmation is split in two: a simplification at `b = c = 0` ordered
every pair as the baseline did and is confirmed with no second draw, while one
at `b > c` claims pairs it changed and still has to survive `b2 >= c2`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from meta_filter.optimizer import (
    NOOP_REASON,
    PromotionConfirmation,
    _confirmation_summary_columns,
    _settled_without_a_redraw,
    confirm_promotion,
    gate_candidates,
)
from meta_filter.selector import SelectionResult, Selector
from tests.test_selector_paired_gate import BASE, PAIRS, _cand, _eval

# --- choose_paired ----------------------------------------------------------------


def _choose(cands: dict[str, dict], *, simpler=frozenset(), strong=frozenset(), **kw):
    evaluations = [_eval("base", is_baseline=True)] + [_eval(cid) for cid in cands]
    correctness = {"base": BASE, **cands}
    return Selector().choose_paired(
        evaluations, correctness, simplification=simpler, strong_evidence=strong, **kw
    )


def test_a_prune_that_changes_no_pair_is_accepted_as_a_simplification():
    result = _choose({"prune": dict(BASE)}, simpler={"prune"})
    assert result.winner == "prune"
    assert result.simplification_winner is True
    assert result.gate == "mcnemar"
    record = result.per_candidate["prune"]
    assert record["simplification"] is True
    assert record["eligible"] is True
    assert (record["b"], record["c"]) == (0, 0)
    assert "SIMPLIFICATION" in result.reason and "b=0 >= c=0" in result.reason
    # A regular candidate with the same vector is a no-op and is refused.
    regular = _choose({"same": dict(BASE)})
    assert regular.winner == "base"
    assert regular.simplification_winner is False


def test_a_simplification_that_loses_pairs_is_not_accepted():
    result = _choose({"prune": _cand(fixes=1, breaks=2)}, simpler={"prune"})
    assert result.winner == "base"
    assert result.per_candidate["prune"]["blockers"] == ["simplification lost pairs: b=1 < c=2"]


def test_a_simplification_that_also_fixes_pairs_needs_no_significance_test():
    # 2-0 is far below any threshold (exact p = 0.25) and below min_discordant.
    result = _choose({"prune": _cand(fixes=2, breaks=0)}, simpler={"prune"})
    assert result.winner == "prune" and result.simplification_winner


def test_a_regular_candidate_that_passes_outranks_a_simplification():
    result = _choose({"prune": dict(BASE), "strong": _cand(fixes=12, breaks=0)}, simpler={"prune"})
    assert result.winner == "strong"
    assert result.simplification_winner is False
    assert result.per_candidate["prune"]["eligible"] is True


def test_a_regular_candidate_that_fails_yields_to_the_simplification():
    result = _choose({"weak": _cand(fixes=3, breaks=1), "prune": dict(BASE)}, simpler={"prune"})
    assert result.winner == "prune" and result.simplification_winner
    assert result.per_candidate["weak"]["eligible"] is False


def test_between_two_simplifications_the_one_keeping_more_pairs_wins():
    result = _choose(
        {"s_flat": dict(BASE), "s_better": _cand(fixes=1, breaks=0)},
        simpler={"s_flat", "s_better"},
    )
    assert result.winner == "s_better"


def test_two_equally_good_simplifications_are_separated_by_complexity():
    result = _choose(
        {"s_big": dict(BASE), "s_small": dict(BASE)},
        simpler={"s_big", "s_small"},
        complexity_by_cand={"s_big": 9.0, "s_small": 4.0},
    )
    assert result.winner == "s_small"
    # Without the map the tie falls back to proposal order, not to a zero.
    assert (
        _choose({"s_big": dict(BASE), "s_small": dict(BASE)}, simpler={"s_big", "s_small"}).winner
        == "s_big"
    )


def test_the_flag_is_written_per_candidate_and_strong_evidence_is_untouched():
    result = _choose(
        {"prune": dict(BASE), "grown": _cand(fixes=8, breaks=0)},
        simpler={"prune"},
        strong={"grown"},
    )
    assert result.per_candidate["grown"]["simplification"] is False
    assert result.per_candidate["grown"]["strong_evidence"] is True
    assert result.per_candidate["prune"]["threshold"] == pytest.approx(
        result.per_candidate["grown"]["threshold"] - 1.0
    )


# --- gate_candidates ---------------------------------------------------------------


def _result(name: str, *, credit, simpler: bool, deletion: bool = True, complexity: float = 1.0):
    constraint = SimpleNamespace(
        status="ok",
        reasons=[],
        deletion=deletion,
        simplification=simpler,
        deleted=["Dead"] if deletion else [],
    )
    return SimpleNamespace(
        candidate_id=name,
        score=0.7,
        error="",
        complexity=complexity,
        constraint=constraint,
        correctness_key=lambda: tuple(sorted(credit.items())),
    )


def test_a_simplification_identical_to_the_parent_on_judged_pairs_is_not_held_as_a_no_op():
    parent = _result("parent", credit=BASE, simpler=False, deletion=False)
    results = {
        "prune": _result("prune", credit=BASE, simpler=True),
        "rewrite": _result("rewrite", credit=BASE, simpler=False, deletion=False),
    }
    kept, gated = gate_candidates(
        [Path("prune"), Path("rewrite")], results, candidates_scored=None, parent=parent
    )
    assert [d.name for d in kept] == ["prune"]
    assert gated["rewrite"]["reason"] == NOOP_REASON


def test_two_identical_simplifications_keep_one_copy():
    parent = _result("parent", credit=BASE, simpler=False, deletion=False)
    results = {
        "s_big": _result("s_big", credit=BASE, simpler=True, complexity=5.0),
        "s_small": _result("s_small", credit=BASE, simpler=True, complexity=2.0),
    }
    kept, gated = gate_candidates(
        [Path("s_big"), Path("s_small")], results, candidates_scored=None, parent=parent
    )
    assert [d.name for d in kept] == ["s_small"]
    assert gated["s_big"]["reason"].startswith("duplicate of s_small")


def test_a_simplification_does_not_use_the_iterations_one_deletion_slot():
    other = _cand(fixes=1, breaks=0)
    results = {
        "prune": _result("prune", credit=other, simpler=True),
        "cut": _result("cut", credit=_cand(fixes=2, breaks=0), simpler=False),
    }
    kept, gated = gate_candidates([Path("prune"), Path("cut")], results, candidates_scored=None)
    assert [d.name for d in kept] == ["prune", "cut"]
    assert gated == {}


# --- confirm_promotion ----------------------------------------------------------------


def test_a_simplification_that_changed_no_pair_is_confirmed_without_a_second_draw():
    """b = c = 0 is the whole claim already observed on every pair, so
    the re-draw has nothing to confirm and is not taken."""

    out = confirm_promotion(
        b1=0,
        c1=0,
        fresh_correctness=None,  # the caller does not buy the re-score at all
        baseline_correctness=BASE,
        p_threshold=0.02,
        simplification=True,
    )
    assert out.confirmed is True
    assert (out.b2, out.c2, out.p_pooled, out.z_pooled) == (None, None, None, None)
    assert out.method == "identical_ordering"
    assert out.reason == (
        f"ordering identical on all {len(BASE)} selection pairs (draw 1 b/c 0/0); "
        "smaller in every complexity component; confirmed without a second draw"
    )
    # A fresh draw handed in anyway is ignored, not tested: it can only
    # disagree by scorer noise, here with b2 < c2.
    noisy = dict(BASE)
    noisy[PAIRS[0]] = 0.0
    assert (
        confirm_promotion(
            b1=0,
            c1=0,
            fresh_correctness=noisy,
            baseline_correctness=BASE,
            p_threshold=0.02,
            simplification=True,
        ).confirmed
        is True
    )
    # The same counts on a REGULAR promotion are still not a confirmation.
    assert not confirm_promotion(
        b1=0, c1=0, fresh_correctness=dict(BASE), baseline_correctness=BASE, p_threshold=0.02
    ).confirmed


def test_a_simplification_that_also_fixed_pairs_still_re_draws_and_can_still_lose():
    """b1 > c1 is a claim about pairs the candidate changed, so it keeps its
    second draw and the `b2 >= c2` requirement."""

    fresh = dict(BASE)
    fresh[PAIRS[0]] = 0.0  # a baseline-right pair now wrong
    out = confirm_promotion(
        b1=3,
        c1=0,
        fresh_correctness=fresh,
        baseline_correctness=BASE,
        p_threshold=0.02,
        simplification=True,
    )
    assert out.confirmed is False
    assert out.method != "identical_ordering"
    assert out.reason == "simplification re-draw lost pairs: b2=0 < c2=1"
    # It is confirmed when the re-draw loses nothing.
    kept = confirm_promotion(
        b1=3,
        c1=0,
        fresh_correctness=dict(BASE),
        baseline_correctness=BASE,
        p_threshold=0.02,
        simplification=True,
    )
    assert kept.confirmed is True
    assert kept.reason.startswith("simplification: draw 1 b/c 3/0, re-draw 0/0")


def test_the_no_second_draw_record_round_trips_through_json():
    out = confirm_promotion(
        b1=0,
        c1=0,
        fresh_correctness=None,
        baseline_correctness=BASE,
        p_threshold=0.02,
        simplification=True,
    )
    payload = json.loads(json.dumps(out.to_dict()))
    assert payload["confirmed"] is True and payload["b2"] is None
    assert isinstance(PromotionConfirmation(**payload), PromotionConfirmation)
    # `summary.csv` leaves the three re-draw counts empty and says True.
    assert _confirmation_summary_columns(out) == {
        "sel_b2": "",
        "sel_c2": "",
        "sel_p_pooled": "",
        "gate_confirmed": True,
    }


def test_only_a_b_equals_c_equals_zero_simplification_skips_the_re_score():
    """`_settled_without_a_redraw` is what the optimizer branches on."""

    def _result(**kw) -> SelectionResult:
        return SelectionResult(winner="prune", reason="", ranked=["prune"], **kw)

    assert _settled_without_a_redraw(_result(simplification_winner=True, b=0, c=0))
    assert not _settled_without_a_redraw(_result(simplification_winner=True, b=2, c=0))
    assert not _settled_without_a_redraw(_result(simplification_winner=True, b=0, c=2))
    # A regular winner never has b = c = 0 (min_discordant blocks it), and
    # would not skip its re-score if it somehow did.
    assert not _settled_without_a_redraw(_result(simplification_winner=False, b=0, c=0))
