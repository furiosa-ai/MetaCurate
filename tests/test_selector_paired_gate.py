"""`Selector.choose_paired`: the McNemar acceptance gate.

The fixtures build each candidate's per-pair credit from a string of cell
codes against the baseline, so b and c are visible in the test itself and the
gate's decisions can be checked by hand.
"""

from __future__ import annotations

import math

import pytest

from meta_filter.paired_test import one_sided_p_from_z, z_threshold_for
from meta_filter.schemas import CandidateEvaluation, FilterMetrics
from meta_filter.selector import Selector


def _eval(
    cid: str,
    *,
    is_baseline: bool = False,
    loss: float = 1.0,
    complexity: float = 0.0,
    score: float = 0.0,
    fn_regressions: int = 0,
    fp_regressions: int = 0,
    invalid: str | None = None,
) -> CandidateEvaluation:
    m = FilterMetrics(
        total=10,
        critic_accept_total=5,
        critic_reject_total=5,
        accepted=5,
        rejected=5,
        reviewed=0,
        agree_accept=5,
        agree_reject=5,
        filter_accept_critic_reject=0,
        filter_reject_critic_accept=0,
        critic_reject_recall=1.0,
        critic_accept_retention=1.0,
        disputed_accept_rate=0.0,
        disputed_reject_rate=0.0,
        review_rate=0.0,
        regression_must_accept_violations=fp_regressions,
        regression_must_reject_violations=fn_regressions,
        complexity_score=complexity,
        loss=loss,
        score=score,
    )
    return CandidateEvaluation(
        candidate_id=cid, is_baseline=is_baseline, metrics=m, invalid_reason=invalid
    )


# The baseline is right on the first half of the pair set and wrong on the
# second half; a candidate is described relative to that. Each pair gets its
# own two rows so the pairs are independent and the plain z is the honest one.
N_PAIRS = 40
PAIRS = [(f"h{i}", f"l{i}") for i in range(N_PAIRS)]
BASE = {p: (1.0 if i < N_PAIRS // 2 else 0.0) for i, p in enumerate(PAIRS)}


def _cand(fixes: int, breaks: int, *, ties: int = 0) -> dict:
    """A candidate that fixes `fixes` baseline-wrong pairs, breaks `breaks`
    baseline-right pairs, and ties `ties` of the remaining baseline-right ones."""

    credit = dict(BASE)
    wrong = [p for p in PAIRS if BASE[p] == 0.0]
    right = [p for p in PAIRS if BASE[p] == 1.0]
    for p in wrong[:fixes]:
        credit[p] = 1.0
    for p in right[:breaks]:
        credit[p] = 0.0
    for p in right[breaks : breaks + ties]:
        credit[p] = 0.5
    return credit


def _choose(cands: dict[str, dict], evals: list[CandidateEvaluation] | None = None, **kw):
    evals = evals or [_eval("baseline", is_baseline=True)] + [_eval(cid) for cid in cands]
    maps = {"baseline": BASE, **cands}
    return Selector().choose_paired(evals, maps, **kw)


def test_arithmetic_and_reason_are_reported():
    # b=12, c=2: z = 10 / sqrt(14) = 2.67, over the default 3-way threshold.
    res = _choose({"c1": _cand(12, 2)})
    assert res.winner == "c1"
    assert res.gate == "mcnemar"
    assert (res.b, res.c) == (12, 2)
    assert res.z == pytest.approx(10 / math.sqrt(14))
    assert res.threshold == pytest.approx(z_threshold_for(1))
    assert res.p < 0.01
    assert math.isfinite(res.z_cluster)
    assert res.sel_concordance == pytest.approx(30 / 40)
    assert res.baseline_sel_concordance == pytest.approx(0.5)
    for token in ("b=12", "c=2", "z=2.67", "p=", "threshold"):
        assert token in res.reason
    stats = res.per_candidate["c1"]
    assert stats["eligible"] is True
    assert stats["method"] == "exact"


def test_threshold_defaults_to_family_wise_for_the_candidate_count():
    res = _choose({"c1": _cand(1, 0), "c2": _cand(1, 0), "c3": _cand(1, 0)})
    assert res.winner == "baseline"
    assert res.threshold == pytest.approx(z_threshold_for(3))
    assert 2.38 <= res.threshold <= 2.45


def test_min_discordant_blocks_a_perfect_but_tiny_gain():
    # b=4, c=0: z=2.0 which the explicit threshold allows, but 4 < 5 pairs.
    res = _choose({"c1": _cand(4, 0)}, z_threshold=1.5, min_discordant=5)
    assert res.winner == "baseline"
    assert "min_discordant" in res.per_candidate["c1"]["blockers"][0]
    assert _choose({"c1": _cand(4, 0)}, z_threshold=1.5, min_discordant=4).winner == "c1"


def test_z_below_threshold_keeps_baseline_and_names_the_closest():
    # b=7, c=3: z = 4/sqrt(10) = 1.26.
    res = _choose({"c1": _cand(7, 3)}, z_threshold=2.0)
    assert res.winner == "baseline"
    assert res.sel_concordance == res.baseline_sel_concordance == pytest.approx(0.5)
    # The refused challenger's numbers are what the row reports.
    assert (res.b, res.c) == (7, 3)
    assert res.z == pytest.approx(4 / math.sqrt(10))
    assert "no candidate passed" in res.reason and "c1" in res.reason


def test_strong_evidence_raises_the_threshold_by_one_z():
    # b=12, c=2: z=2.67 clears 2.0 but not 3.0 (exact p 0.0065 vs 0.0013).
    plain = _choose({"c1": _cand(12, 2)}, z_threshold=2.0)
    assert plain.winner == "c1"
    strong = _choose({"c1": _cand(12, 2)}, z_threshold=2.0, strong_evidence={"c1"})
    assert strong.winner == "baseline"
    assert strong.per_candidate["c1"]["threshold"] == pytest.approx(3.0)
    assert strong.per_candidate["c1"]["strong_evidence"] is True
    # A big enough gain still gets through the raised gate: b=20, c=0 → z=4.47.
    assert _choose({"c1": _cand(20, 0)}, z_threshold=2.0, strong_evidence={"c1"}).winner == "c1"
    # And so does one the old doubling (4.0, α 3.2e-5) refused: 14-0 is exact
    # p 6.1e-5, under 3.0's 0.00135 and over 4.0's.
    assert _choose({"c1": _cand(14, 0)}, z_threshold=2.0, strong_evidence={"c1"}).winner == "c1"
    # The increment is a parameter; 0 makes strong evidence ordinary evidence.
    assert (
        _choose(
            {"c1": _cand(12, 2)}, z_threshold=2.0, strong_evidence={"c1"}, strong_extra_z=0.0
        ).winner
        == "c1"
    )


def test_the_kept_baseline_reason_says_what_b_was_needed():
    # b=7, c=3 at z 2.0: exact p at b+c=10 falls under 0.0228 first at b=9.
    res = _choose({"c1": _cand(7, 3)}, z_threshold=2.0)
    assert res.winner == "baseline"
    assert res.per_candidate["c1"]["min_b_required"] == 9
    assert "Closest: c1 b=7 c=3, needed b ≥ 9;" in res.reason
    # Too few discordant pairs for any b to pass: the line says so instead.
    tiny = _choose({"c1": _cand(3, 1)}, z_threshold=2.0, min_discordant=1)
    assert tiny.per_candidate["c1"]["min_b_required"] is None
    assert "no b reaches the threshold at b+c=4" in tiny.reason
    # A winner's record carries the number too.
    assert (
        _choose({"c1": _cand(12, 2)}, z_threshold=2.0).per_candidate["c1"]["min_b_required"] == 12
    )


def test_exact_tie_between_eligible_candidates_keeps_baseline():
    a = _cand(12, 2)
    # Same counts on different pairs: identical selection concordance.
    b = dict(BASE)
    wrong = [p for p in PAIRS if BASE[p] == 0.0]
    right = [p for p in PAIRS if BASE[p] == 1.0]
    for p in wrong[-12:]:
        b[p] = 1.0
    for p in right[-2:]:
        b[p] = 0.0
    res = _choose({"c1": a, "c2": b}, z_threshold=2.0)
    assert res.per_candidate["c1"]["eligible"] and res.per_candidate["c2"]["eligible"]
    assert res.winner == "baseline"
    assert "none strictly ahead" in res.reason


def test_none_eligible_keeps_baseline():
    res = _choose({"c1": _cand(2, 3), "c2": _cand(0, 0), "c3": _cand(6, 6)}, z_threshold=2.0)
    assert res.winner == "baseline"
    assert not any(v["eligible"] for v in res.per_candidate.values())
    assert res.ranked[0] == "baseline"


def test_highest_selection_concordance_no_longer_wins_on_its_own():
    """ "Concordance decides once both are in" would be an argmax on noise. c1
    is b=12 c=0 (conc 0.80) and c2 is b=16 c=2 (conc 0.85), but head to head
    that is b=4 c=2 — z_cluster 0.62, nowhere near a lead.

    With nothing measured to break the tie the baseline stands, which
    `test_exact_tie_between_eligible_candidates_keeps_baseline` also pins.
    Holding is reached only where the loop genuinely measured nothing — which
    is exactly where promoting on proposal order would let a filter grow with
    no measured benefit, the failure the paired gate exists to stop."""

    res = _choose({"c1": _cand(12, 0), "c2": _cand(16, 2)}, z_threshold=2.0)
    assert res.winner == "baseline"
    assert res.among_tie_break == "hold"
    # The ranking is still by concordance behind the winner; only the CHOICE
    # stopped using it.
    assert res.ranked[0] == "baseline"
    assert res.ranked[1:] == ["c2", "c1"]

    # Given a size to compare, the tie goes to the smaller filter — the rule
    # the simplification lane already runs on — not to the higher concordance.
    sized = _choose(
        {"c1": _cand(12, 0), "c2": _cand(16, 2)},
        z_threshold=2.0,
        complexity_by_cand={"c1": 4.0, "c2": 20.0},
    )
    assert sized.winner == "c1"
    assert sized.among_tie_break == "complexity"


def test_no_loss_or_complexity_tie_break():
    same = _cand(12, 2)
    evals = [
        _eval("baseline", is_baseline=True, loss=5.0, complexity=50.0),
        _eval("lean", loss=0.1, complexity=1.0, score=0.9),
        _eval("fat", loss=9.0, complexity=99.0, score=0.1),
    ]
    res = _choose({"lean": same, "fat": dict(same)}, evals, z_threshold=2.0)
    # A loss or complexity sort would hand this to "lean"; the paired gate
    # must not.
    assert res.winner == "baseline"


def test_regression_filter_is_respected():
    evals = [
        _eval("baseline", is_baseline=True),
        _eval("regressed", fn_regressions=1),
        _eval("clean"),
    ]
    res = _choose({"regressed": _cand(20, 0), "clean": _cand(12, 2)}, evals, z_threshold=2.0)
    assert res.winner == "clean"
    assert res.per_candidate["regressed"]["eligible"] is False
    assert res.per_candidate["regressed"]["regression_ok"] is False


def test_regression_filter_lifts_only_when_the_baseline_regresses_too():
    evals = [
        _eval("baseline", is_baseline=True, fn_regressions=2),
        _eval("regressed", fn_regressions=1),
    ]
    res = _choose({"regressed": _cand(20, 0)}, evals, z_threshold=2.0)
    assert res.winner == "regressed"
    assert "NOTE: every candidate regressed" in res.reason


def test_invalid_candidates_are_skipped_and_need_no_map():
    evals = [
        _eval("baseline", is_baseline=True),
        _eval("broken", invalid="did not compile"),
        _eval("c1"),
    ]
    res = Selector().choose_paired(evals, {"baseline": BASE, "c1": _cand(12, 2)}, z_threshold=2.0)
    assert res.winner == "c1"
    assert "broken" not in res.per_candidate
    assert "broken" not in res.ranked


def test_baseline_alone_is_retained():
    res = Selector().choose_paired([_eval("baseline", is_baseline=True)], {"baseline": BASE})
    assert res.winner == "baseline"
    assert res.per_candidate == {}
    assert res.gate == "mcnemar"


def test_missing_correctness_map_is_an_error():
    evals = [_eval("baseline", is_baseline=True), _eval("c1")]
    with pytest.raises(ValueError, match="c1"):
        Selector().choose_paired(evals, {"baseline": BASE})
    with pytest.raises(ValueError, match="baseline"):
        Selector().choose_paired(evals, {"c1": _cand(1, 0)})


def test_exactly_one_baseline_is_required():
    with pytest.raises(ValueError, match="exactly one baseline"):
        Selector().choose_paired([_eval("c1")], {"c1": BASE})


def test_pareto_fields_are_still_populated():
    evals = [
        _eval("baseline", is_baseline=True, loss=1.0, complexity=10.0),
        _eval("c1", loss=0.5, complexity=5.0),
    ]
    res = _choose({"c1": _cand(12, 2)}, evals, z_threshold=2.0)
    assert res.winner == "c1"
    assert res.pareto_ids_complexity == ["c1"]
    assert res.winner_pareto_safe is True


# --- the gate is on p, so the exact-binomial branch binds ----------------------


def test_six_to_nothing_clears_z_but_not_the_exact_p_and_is_refused():
    """b=6, c=0: z = 2.449 passes a 2.43 threshold, but the exact chance of 6-0
    under no effect is 1/64 = 0.0156, twice the 0.0076 that z = 2.43 stands
    for. Gating on z let it through; gating on p does not."""

    threshold = 2.43
    res = _choose({"c1": _cand(6, 0)}, z_threshold=threshold, min_discordant=5)
    stats = res.per_candidate["c1"]

    assert stats["z"] == pytest.approx(math.sqrt(6))
    assert stats["z"] > threshold, "the probe is only meaningful if z alone would pass"
    assert stats["method"] == "exact"
    assert stats["p"] == pytest.approx(1 / 64)
    assert stats["p_threshold"] == pytest.approx(0.00755, abs=2e-4)
    assert stats["eligible"] is False
    assert any(b.startswith("p=0.0156 > p_threshold=") for b in stats["blockers"])
    assert res.winner == "baseline"
    assert "p=0.0156" in res.reason


def test_eight_to_nothing_clears_the_exact_p():
    res = _choose({"c1": _cand(8, 0)}, z_threshold=2.43, min_discordant=5)
    stats = res.per_candidate["c1"]
    assert stats["method"] == "exact"
    assert stats["p"] == pytest.approx(1 / 256)
    assert stats["eligible"] is True
    assert res.winner == "c1"
    assert "p=0.0039 <= p_threshold" in res.reason


def test_in_the_normal_branch_p_gating_is_z_gating():
    """Above 25 discordant pairs `p` is derived from `z`, so the two gates
    agree: 19-7 (z 2.35) fails 2.43, 20-6 (z 2.75) passes it. b + c = 26 here."""

    below = _choose({"c1": _cand(19, 7)}, z_threshold=2.43)
    above = _choose({"c1": _cand(20, 6)}, z_threshold=2.43)
    for res in (below, above):
        stats = res.per_candidate["c1"]
        assert stats["method"] == "normal"
        assert stats["b"] + stats["c"] == 26
    assert below.per_candidate["c1"]["z"] == pytest.approx(12 / math.sqrt(26))
    assert below.per_candidate["c1"]["eligible"] is False
    assert above.per_candidate["c1"]["z"] == pytest.approx(14 / math.sqrt(26))
    assert above.per_candidate["c1"]["eligible"] is True
    assert above.winner == "c1"


def test_the_p_threshold_is_one_z_higher_for_strong_evidence():
    """3.43 in z is about 3e-4 in p; 8-0 (0.0039) does not reach it, 12-0
    (0.00024) does. The doubled 4.86 (6e-7) that this replaced could not be
    reached by any exact-branch count under 25 discordant pairs."""

    res = _choose({"c1": _cand(8, 0)}, z_threshold=2.43, strong_evidence={"c1"})
    stats = res.per_candidate["c1"]
    assert stats["threshold"] == pytest.approx(3.43)
    assert stats["p_threshold"] == pytest.approx(3.0e-4, abs=2e-5)
    assert stats["eligible"] is False
    assert stats["min_b_required"] is None, "no b out of 8 reaches 3e-4"

    twelve = _choose({"c1": _cand(12, 0)}, z_threshold=2.43, strong_evidence={"c1"})
    assert twelve.per_candidate["c1"]["p"] == pytest.approx(1 / 4096)
    assert twelve.winner == "c1"


def test_a_z_353_candidate_clears_the_additive_strong_gate():
    """A strong-evidence candidate at z 3.53 would be blocked by a doubled
    threshold of 4.55. 24-5 is z 3.53 in the normal branch; at 2.43 + 1 it
    passes, at 2 x 2.43 it would not."""

    # The shared fixture has 20 fixable pairs; this one needs 24, so it
    # builds its own: baseline right on 10 pairs, wrong on 30.
    pairs = [(f"H{i}", f"L{i}") for i in range(40)]
    base = {p: (1.0 if i < 10 else 0.0) for i, p in enumerate(pairs)}
    cand = dict(base)
    for p in pairs[10:34]:
        cand[p] = 1.0
    for p in pairs[:5]:
        cand[p] = 0.0
    res = Selector().choose_paired(
        [_eval("baseline", is_baseline=True), _eval("c1")],
        {"baseline": base, "c1": cand},
        z_threshold=2.43,
        strong_evidence={"c1"},
        min_discordant=5,
    )
    stats = res.per_candidate["c1"]
    assert (stats["b"], stats["c"]) == (24, 5)
    assert stats["method"] == "normal"
    assert stats["z"] == pytest.approx(19 / math.sqrt(29))
    assert stats["eligible"] is True
    assert res.winner == "c1"
    assert stats["p"] > one_sided_p_from_z(4.86), "the old doubled gate would have refused it"
