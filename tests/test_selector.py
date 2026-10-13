"""The choice *among* eligible candidates is a test, not an argmax.

Taking `max(sel_concordance)` over whoever cleared the baseline gate, with
only an exact tie counted as a tie, reads margins well inside the noise of a
concordance difference as wins. These tests pin the replacement: the leader is
compared to every other eligible candidate on the same pairs, and a lead that
does not survive that comparison hands the decision to a tie-break that is not
the metric, or to the baseline.

Fixtures come from `test_selector_paired_gate.py` — forty pairs, each with its
own two rows, the baseline right on the first twenty — so b and c are readable
off the test and the rows really are independent.
"""

from __future__ import annotations

import pytest

from meta_filter.gate.selector import AMONG_CANDIDATES_Z, Selector
from tests.test_selector_paired_gate import BASE, PAIRS, _cand, _eval


def _evals(*ids: str) -> list:
    """Evaluations in proposal order; the one called "baseline" is the baseline."""

    return [_eval(cid, is_baseline=cid == "baseline") for cid in ids]


def _choose(cands: dict[str, dict], *, order: list[str] | None = None, **kw):
    """Run the paired gate over `cands`, proposed in `order` (default: key order)."""

    evaluations = _evals("baseline", *(order or cands))
    return Selector().choose_paired(evaluations, {"baseline": BASE, **cands}, **kw)


# --- a lead that survives the test ------------------------------------------------


def test_a_leader_that_separates_from_the_runner_up_is_promoted():
    """20-0 against the baseline versus 8-0: the leader is right on twelve
    pairs the runner-up gets wrong and wrong on none of its, which is exact p
    0.00024 and z_cluster 2.81. That is a lead, and it is promoted."""

    res = _choose({"strong": _cand(20, 0), "weak": _cand(8, 0)}, z_threshold=2.0)

    assert res.winner == "strong"
    assert res.among_tie_break == "separated"
    vs = res.per_candidate["weak"]["vs_leader"]
    assert vs["leader"] == "strong"
    assert (vs["b"], vs["c"]) == (12, 0)
    assert vs["separated"] is True
    assert vs["z_cluster"] > AMONG_CANDIDATES_Z
    among = res.per_candidate["strong"]["among_candidates"]
    assert among == {
        "leader": True,
        "threshold": AMONG_CANDIDATES_Z,
        "tested": ["weak"],
        "tied_with": [],
        "tie_break": "separated",
        "chosen": "strong",
    }
    assert "Leader strong separated from every other eligible candidate" in res.reason


def test_one_eligible_candidate_is_decided_exactly_as_before():
    """Nothing to compare the leader against, so no comparison is run and no
    among-candidates telemetry is written."""

    res = _choose({"solo": _cand(12, 2), "failed": _cand(2, 3)}, z_threshold=2.0)

    assert res.winner == "solo"
    assert res.per_candidate["failed"]["eligible"] is False
    assert res.among_tie_break == ""
    assert "among_candidates" not in res.per_candidate["solo"]
    assert "vs_leader" not in res.per_candidate["failed"]
    assert "Leader" not in res.reason


# --- a lead that does not ---------------------------------------------------------


def test_an_inseparable_leader_falls_through_to_complexity():
    """12-0 and 11-0 differ on a single pair. The margin is 0.025 of
    concordance and the old code would have promoted it; one discordant pair
    is not a measurement, so the smaller filter takes the iteration instead —
    the rule the simplification lane already runs on."""

    res = _choose(
        {"big": _cand(12, 0), "small": _cand(11, 0)},
        z_threshold=2.0,
        complexity_by_cand={"big": 20.0, "small": 4.0},
    )

    assert res.winner == "small"
    assert res.among_tie_break == "complexity"
    assert res.per_candidate["big"]["among_candidates"]["tied_with"] == ["small"]
    vs = res.per_candidate["small"]["vs_leader"]
    assert (vs["b"], vs["c"]) == (1, 0)
    assert vs["separated"] is False
    assert "Leader big was not separable from small" in res.reason
    assert "fell through to complexity, which chose small" in res.reason
    # The promoted candidate's own numbers against the baseline are still what
    # the summary row reports.
    assert (res.b, res.c) == (11, 0)


def test_complexity_ties_fall_through_to_the_smaller_change():
    res = _choose(
        {"rewrite": _cand(12, 0), "nudge": _cand(11, 0)},
        z_threshold=2.0,
        complexity_by_cand={"rewrite": 7.0, "nudge": 7.0},
        change_size_by_cand={"rewrite": 31.0, "nudge": 3.0},
    )

    assert res.winner == "nudge"
    assert res.among_tie_break == "change"
    assert "fell through to change, which chose nudge" in res.reason


def test_complexity_and_change_tied_fall_through_to_proposal_order():
    """Same size, same edit size, indistinguishable on the exam: the loop has
    nothing left to prefer with, and the first proposal takes it. Note which
    one wins — `second` leads on concordance and loses anyway, which is the
    whole point of taking the decision off the metric."""

    res = _choose(
        {"second": _cand(12, 0), "first": _cand(11, 0)},
        order=["first", "second"],
        z_threshold=2.0,
        complexity_by_cand={"second": 7.0, "first": 7.0},
        change_size_by_cand={"second": 5.0, "first": 5.0},
    )

    assert res.winner == "first"
    assert res.among_tie_break == "order"
    assert "fell through to order, which chose first" in res.reason


def test_nothing_measured_keeps_the_baseline_rather_than_flipping_a_coin():
    """No complexity, no change size, and a lead the test refuses: the loop
    knows nothing that distinguishes these two filters. Proposal order would
    still name one, and naming one is exactly the failure being fixed, so the
    baseline stands.

    The other side of this was argued in `_tie_break_among` and is not silly —
    both candidates did clear the gate — but the tie-break chain ends at
    "keep the baseline", and a filter chosen by nothing is what the
    proposer reads and edits for the rest of the run."""

    res = _choose({"big": _cand(12, 0), "small": _cand(11, 0)}, z_threshold=2.0)

    assert res.winner == "baseline"
    assert res.among_tie_break == "hold"
    assert res.per_candidate["big"]["among_candidates"]["chosen"] is None
    assert "no tie-break separated them either" in res.reason
    assert "the baseline stands rather than a coin flip" in res.reason


def test_an_unmeasured_candidate_cannot_win_the_complexity_tie_break():
    """A missing complexity sorts last, not at zero — the simplification
    lane's rule, for the same reason. With only one of the two measured the
    axis says nothing about the pair, so it is skipped rather than read as a
    win for whoever happens to be in the map."""

    res = _choose(
        {"big": _cand(12, 0), "small": _cand(11, 0)},
        z_threshold=2.0,
        complexity_by_cand={"small": 4.0},
    )

    assert res.winner == "baseline"
    assert res.among_tie_break == "hold"


def test_the_leader_is_tested_against_every_other_eligible_candidate():
    """Three eligible: the leader separates from the weakest and not from the
    middle one, so the tie-break runs over the pair it could not separate and
    the one it beat stays out of it."""

    res = _choose(
        {"lead": _cand(20, 0), "near": _cand(19, 0), "far": _cand(8, 0)},
        z_threshold=2.0,
        complexity_by_cand={"lead": 30.0, "near": 5.0, "far": 1.0},
    )

    assert res.per_candidate["near"]["vs_leader"]["separated"] is False
    assert res.per_candidate["far"]["vs_leader"]["separated"] is True
    among = res.per_candidate["lead"]["among_candidates"]
    assert among["tested"] == ["near", "far"]
    assert among["tied_with"] == ["near"]
    # `far` is the smallest filter of the three and still does not win the
    # tie-break: it lost the comparison, so it is not in the tied group.
    assert res.winner == "near"
    assert res.among_tie_break == "complexity"


# --- the threshold ----------------------------------------------------------------


def test_the_among_candidates_threshold_is_its_own_parameter():
    """It is not the baseline gate's threshold and is not derived from it: the
    two nulls differ (candidates share the cached cells of every criterion
    neither edited) and only the baseline one has been simulated. 12-0 against
    6-0 is z_cluster 1.79 — a lead at 1.5, not one at the conservative
    default, and `z_threshold` moving does not move it."""

    cands = {"a": _cand(12, 0), "b": _cand(6, 0)}

    assert AMONG_CANDIDATES_Z > 2.0, "the default must be stricter than the baseline gate"
    default = _choose(cands, z_threshold=2.0, complexity_by_cand={"a": 9.0, "b": 2.0})
    assert default.winner == "b" and default.among_tie_break == "complexity"

    loosened = _choose(
        cands, z_threshold=2.0, among_z_threshold=1.5, complexity_by_cand={"a": 9.0, "b": 2.0}
    )
    assert loosened.winner == "a" and loosened.among_tie_break == "separated"

    # The baseline gate's threshold is reported per candidate; the
    # among-candidates one is reported on the comparison, and they are
    # independent knobs.
    assert default.per_candidate["a"]["threshold"] == pytest.approx(2.0)
    assert default.per_candidate["b"]["vs_leader"]["threshold"] == pytest.approx(AMONG_CANDIDATES_Z)


def test_too_few_discordant_pairs_is_not_a_separation():
    """`min_discordant` binds here as it does against the baseline: two
    candidates that differ on four pairs of 2,696 have not been compared, they
    have been sampled."""

    res = _choose(
        {"a": _cand(12, 0), "b": _cand(8, 0)},
        z_threshold=2.0,
        among_z_threshold=0.0,
        complexity_by_cand={"a": 9.0, "b": 2.0},
    )

    vs = res.per_candidate["b"]["vs_leader"]
    assert (vs["b"], vs["c"]) == (4, 0)
    assert vs["p"] <= vs["p_threshold"], "p alone would call this separated at threshold 0"
    assert vs["separated"] is False
    assert res.among_tie_break == "complexity"


# --- the guards it inherited ------------------------------------------------------


def test_a_tie_break_winner_still_has_to_beat_the_baseline():
    """The pre-2.6 `top > baseline_sel` guard, moved onto whoever the
    tie-break picked. A candidate can clear the paired gate on b − c and still
    sit at or below the baseline's mean credit, because its own ties are paid
    at half here and count as losses there. The baseline wins those."""

    base = {p: (1.0 if i < 10 else 0.5 if i < 30 else 0.0) for i, p in enumerate(PAIRS)}

    def _dropper(fixes: int) -> dict:
        # Keeps the ten pairs the baseline orders, abandons the twenty it ties
        # (no `c`: a baseline tie is not a baseline win), fixes `fixes` of the
        # ten it gets wrong.
        credit = {p: (1.0 if i < 10 else 0.0) for i, p in enumerate(PAIRS)}
        for p in PAIRS[30 : 30 + fixes]:
            credit[p] = 1.0
        return credit

    res = Selector().choose_paired(
        [_eval("baseline", is_baseline=True), _eval("a"), _eval("b")],
        {"baseline": base, "a": _dropper(8), "b": _dropper(7)},
        z_threshold=2.0,
        complexity_by_cand={"a": 9.0, "b": 3.0},
    )

    assert res.per_candidate["a"]["eligible"] and res.per_candidate["b"]["eligible"]
    assert res.winner == "baseline"
    assert res.among_tie_break == "below_baseline"
    assert res.per_candidate["a"]["among_candidates"]["chosen"] is None
    assert "not ahead of the baseline on selection concordance" in res.reason


def test_the_simplification_lane_is_untouched():
    """The among-candidates test compares regular candidates with each other. Simplifications are
    chosen by their own rule — most pairs kept, then complexity, then proposal
    order — and that rule ran before the among-candidates test existed and
    still runs unchanged, including when two regular candidates deadlock and
    hand the iteration to it."""

    flat, tidier = dict(BASE), dict(BASE)
    res = Selector().choose_paired(
        _evals("baseline", "big", "small", "prune"),
        {"baseline": BASE, "big": _cand(12, 0), "small": _cand(11, 0), "prune": flat},
        z_threshold=2.0,
        simplification={"prune"},
    )
    # The two regular candidates are eligible and deadlock (no complexity map),
    # so the simplification takes the iteration exactly as it would have if
    # they had failed the gate outright.
    assert res.among_tie_break == "hold"
    assert res.winner == "prune"
    assert res.simplification_winner is True
    assert "SIMPLIFICATION" in res.reason
    assert "no tie-break separated them either" in res.reason

    # Two simplifications are still separated by their own lane, never by the
    # among-candidates test: `simplification_eligible` is a separate list.
    two = Selector().choose_paired(
        _evals("baseline", "fat", "lean"),
        {"baseline": BASE, "fat": flat, "lean": tidier},
        simplification={"fat", "lean"},
        complexity_by_cand={"fat": 9.0, "lean": 4.0},
    )
    assert two.winner == "lean"
    assert two.simplification_winner is True
    assert two.among_tie_break == ""
    assert all("vs_leader" not in rec for rec in two.per_candidate.values())
