"""At most one deletion candidate per iteration.

A deletion — a criterion, a `rule_` function or a policy branch removed —
is the edit whose damage the train-side number is worst at showing: it
frees complexity on every row and loses ordering only on the rows the
removed piece used to separate. So the loop takes deletions one at a time:
among the candidates whose constraint report says `deletion`, the highest
precheck score is scored and the rest are held back with
`invalid_reason="second deletion candidate this iteration"`.

Same fixture as `test_candidate_gating.py`. The deletion here is
`rule_noop` removed from rules.py (rule_count 1 -> 0, which the deletion
floor of initial - 1 = 0 still allows), paired with a policy that decides
the candidate's precheck rank.

Each deletion policy carries one line the baseline's does not, so
`policy_loc` rises and the candidate is a deletion but NOT a
simplification. A candidate smaller in every component skips this limit by
design (`tests/test_selector_simplification.py`), and the limit is what this
file is about. `policy_loc` is in neither the soft nor the hard cap, so the
extra line fires nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

from meta_filter.loop.optimizer import SECOND_DELETION_REASON, gate_candidates
from tests.selection_gate_fixture import TRUTH_POLICY, prepare
from tests.test_candidate_gating import (
    INVERTED_NOISY_POLICY,
    INVERTED_TRUTH_POLICY,
    LOSING_FIRST,
    RUBRIC,
    _build,
    _precheck,
    _validation,
)

NO_RULES = "# every rule removed\n"


def _weighted(policy: str) -> str:
    """The same policy plus a `WEIGHT` line: one more `policy_loc` than the
    baseline, which is what keeps these deletions out of the simplification
    lane."""

    return policy.replace("def decide(", "WEIGHT = 1.0\n\n\ndef decide(").replace(
        "scores['", "WEIGHT * scores['"
    )


INVERTED_DELETION = ("inverted_deletion", _weighted(INVERTED_TRUTH_POLICY), RUBRIC, NO_RULES)
TRUTH_DELETION = ("truth_deletion", _weighted(TRUTH_POLICY), RUBRIC, NO_RULES)


def test_two_deletion_candidates_the_lower_precheck_one_is_held_back(tmp_path):
    """Proposal order puts the weaker deletion first; the score decides, not the order."""

    run = prepare(tmp_path)
    run_dir = _build(
        run,
        script={
            0: LOSING_FIRST,
            1: [INVERTED_DELETION, TRUTH_DELETION, ("inverted_noisy", INVERTED_NOISY_POLICY)],
        },
        iterations=2,
        run_id="two-deletions",
    ).run()

    validation = _validation(run_dir, 1)
    held = _precheck(run_dir, 1)["gated_out"]

    reason = validation["iter001_00_inverted_deletion"]["invalid_reason"]
    assert reason.startswith(SECOND_DELETION_REASON)
    assert "kept iter001_01_truth_deletion" in reason
    assert validation["iter001_01_truth_deletion"]["invalid_reason"] is None
    assert validation["iter001_02_inverted_noisy"]["invalid_reason"] is None, (
        "a non-deletion candidate is untouched by the limit"
    )
    assert set(held) == {"iter001_00_inverted_deletion"}
    assert held["iter001_00_inverted_deletion"]["constraint_status"] == "ok"

    constraints = _precheck(run_dir, 1)["constraints"]
    for cid in ("iter001_00_inverted_deletion", "iter001_01_truth_deletion"):
        assert constraints[cid]["deletion"] is True
        assert constraints[cid]["deleted"] == ["rule_noop"]
    assert constraints["iter001_02_inverted_noisy"]["deletion"] is False

    # Held back, not removed: the next proposer may read it.
    assert (run_dir / "iterations" / "001" / "candidates" / "iter001_00_inverted_deletion").is_dir()
    solved = {p.stem for p in (run_dir / "iterations" / "001" / "solver_outputs").iterdir()}
    assert "iter001_00_inverted_deletion" not in solved
    assert "iter001_01_truth_deletion" in solved


def test_a_single_deletion_candidate_passes(tmp_path):
    run = prepare(tmp_path)
    run_dir = _build(
        run,
        script={0: LOSING_FIRST, 1: [TRUTH_DELETION, ("inverted_noisy", INVERTED_NOISY_POLICY)]},
        iterations=2,
        run_id="one-deletion",
    ).run()

    validation = _validation(run_dir, 1)
    assert all(entry["invalid_reason"] is None for entry in validation.values())
    assert _precheck(run_dir, 1)["gated_out"] == {}
    assert _precheck(run_dir, 1)["constraints"]["iter001_00_truth_deletion"]["deletion"] is True


def test_the_limit_holds_at_iteration_zero_where_nothing_is_judged_yet(tmp_path):
    """No pairs means no precheck score, so proposal order breaks the tie —
    but the cap still applies: two deletions never both get scored."""

    run = prepare(tmp_path)
    run_dir = _build(
        run,
        script={0: [TRUTH_DELETION, INVERTED_DELETION]},
        iterations=1,
        run_id="zero-deletions",
    ).run()

    payload = _precheck(run_dir, 0)
    assert payload["before"] == {} and payload["after"] == {}, "nothing to measure at iteration 0"
    assert set(payload["gated_out"]) == {"iter000_01_inverted_deletion"}
    held = payload["gated_out"]["iter000_01_inverted_deletion"]
    assert held["reason"].startswith(SECOND_DELETION_REASON)
    assert held["score"] is None
    assert _validation(run_dir, 0)["iter000_01_inverted_deletion"]["invalid_reason"].startswith(
        SECOND_DELETION_REASON
    )
    assert _validation(run_dir, 0)["iter000_00_truth_deletion"]["invalid_reason"] is None


# --- the pure rule, without a run ---------------------------------------------


class _Constraint:
    def __init__(self, status="ok", deletion=False, reasons=()):
        self.status = status
        self.deletion = deletion
        self.reasons = list(reasons)


class _Result:
    def __init__(self, score, constraint=None, error=""):
        self.score = score
        self.constraint = constraint
        self.error = error


def _dirs(*names: str) -> list[Path]:
    return [Path(f"/cands/{n}") for n in names]


def test_gate_candidates_applies_invalid_then_deletion_then_rank():
    dirs = _dirs("a", "b", "c", "d", "e")
    results = {
        "a": _Result(0.9, _Constraint("invalid", reasons=["rubric_criteria 9 > 1.5 x initial 4"])),
        "b": _Result(0.8, _Constraint(deletion=True)),
        "c": _Result(0.7, _Constraint(deletion=True)),
        "d": _Result(0.6, _Constraint()),
        "e": _Result(0.5, _Constraint("needs_strong_evidence")),
    }

    kept, held = gate_candidates(dirs, results, candidates_scored=2)

    assert [d.name for d in kept] == ["b", "d"]
    assert held["a"]["reason"] == "constraint invalid: rubric_criteria 9 > 1.5 x initial 4"
    assert held["a"]["constraint_status"] == "invalid"
    assert held["c"]["reason"] == f"{SECOND_DELETION_REASON} (kept b)"
    # Rank is over the survivors of the first two rules: b, d, e -> 3.
    assert held["e"]["reason"] == "precheck rank 3 of 3"
    assert held["e"]["constraint_status"] == "needs_strong_evidence"
    assert held["e"]["score"] == 0.5


def test_gate_candidates_ranks_an_unscoreable_candidate_last_and_keeps_proposal_order_on_ties():
    dirs = _dirs("a", "b", "c")
    results = {
        "a": _Result(0.0, error="ImportError: boom"),
        "b": _Result(0.7),
        "c": _Result(0.7),
    }

    kept, held = gate_candidates(dirs, results, candidates_scored=1)

    assert [d.name for d in kept] == ["b"]
    assert held["a"] == {"reason": "precheck rank 3 of 3", "score": None, "constraint_status": None}
    assert held["c"]["reason"] == "precheck rank 2 of 3"


def test_gate_candidates_without_reports_or_a_limit_keeps_everything():
    dirs = _dirs("a", "b")
    kept, held = gate_candidates(
        dirs, {"a": _Result(0.1), "b": _Result(0.2)}, candidates_scored=None
    )
    assert kept == dirs and held == {}


def test_gate_candidates_survives_a_candidate_with_no_precheck_result():
    """A directory the proposer added after the precheck ran has no readout;
    it ranks last rather than raising."""

    dirs = _dirs("a", "b")
    kept, held = gate_candidates(dirs, {"a": _Result(0.3)}, candidates_scored=1)
    assert [d.name for d in kept] == ["a"]
    assert held["b"]["reason"] == "precheck rank 2 of 2"
    assert json.dumps(held)  # serialisable as written into precheck.json
