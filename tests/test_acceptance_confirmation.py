"""An acceptance is confirmed on its own re-draw.

The gate accepts on draw 1 (candidate cells vs baseline cells). The winner is
then re-scored fresh — the promotion re-score — and the fresh draw has a say:
the promotion stands only if the re-draw still favours the candidate
(`b2 > c2`) and the two draws pooled clear the threshold draw 1 was held to.
A borderline gate — b/c 26/7 (z 3.31) on draw 1, 28/14 (z 2.16) on the
re-drawn cells — is the case this exists for.

Two scorers here. The fixture's `MetaScorer` is deterministic, so a fresh
draw agrees with draw 1 exactly and the promotion is confirmed. `Reversing`
negates every score inside `fresh_draw()`, so the candidate that won 8-0 on
draw 1 reads 0-8 on the re-draw — the worst case of that pattern — and the
promotion is undone with the baseline's scores left as they were.
"""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from meta_filter.core.filter_loader import load_filter
from meta_filter.core.solver import Solver
from meta_filter.gate.paired_test import (
    one_sided_p_from_z,
    pair_correctness,
    paired_stats,
    stats_from_counts,
)
from meta_filter.gate.selection_set import baseline_scores_path
from meta_filter.loop.optimizer import PromotionConfirmation, confirm_promotion
from tests.selection_gate_fixture import (
    TRUTH_POLICY,
    MetaScorer,
    build_optimizer,
    prepare,
)


def _summary(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selection(run_dir: Path, k: int) -> dict:
    return json.loads((run_dir / "iterations" / f"{k:03d}" / "selection.json").read_text("utf-8"))


# --- the pooled arithmetic, on hand-built maps ---------------------------------


def _maps(b: int, c: int, both_right: int = 0, both_wrong: int = 0):
    """Candidate and baseline credit maps with exactly these discordant counts."""

    cand: dict[tuple[str, str], float] = {}
    base: dict[tuple[str, str], float] = {}
    n = 0
    for credit_c, credit_b, count in (
        (1.0, 0.0, b),
        (0.0, 1.0, c),
        (1.0, 1.0, both_right),
        (0.0, 0.0, both_wrong),
    ):
        for _ in range(count):
            key = (f"h{n}", f"l{n}")
            cand[key], base[key] = credit_c, credit_b
            n += 1
    return cand, base


def test_stats_from_counts_is_the_arithmetic_paired_stats_runs():
    for b, c in ((8, 0), (5, 3), (26, 7), (28, 14), (0, 0), (12, 12)):
        cand, base = _maps(b, c, both_right=3)
        stats = paired_stats(cand, base)
        z, p, method = stats_from_counts(b, c)
        assert (z, p, method) == (stats.z, stats.p_one_sided, stats.method), (b, c)
    assert stats_from_counts(0, 0) == (0.0, 1.0, "none")
    assert stats_from_counts(8, 0)[2] == "exact" and stats_from_counts(20, 5)[2] == "normal"
    with pytest.raises(ValueError):
        stats_from_counts(-1, 0)


def test_pooled_counts_are_summed_over_both_draws_and_tested_once():
    """26/7 then 28/14 pools to 54/21 — one test on 75."""

    fresh, base = _maps(28, 14, both_right=10)
    out = confirm_promotion(
        b1=26, c1=7, fresh_correctness=fresh, baseline_correctness=base, p_threshold=0.0076
    )
    z, p, method = stats_from_counts(54, 21)
    assert (out.b1, out.c1, out.b2, out.c2) == (26, 7, 28, 14)
    assert out.z_pooled == pytest.approx(z) and out.p_pooled == pytest.approx(p)
    assert out.method == method == "normal"
    assert out.z_pooled == pytest.approx(33 / 75**0.5)
    assert out.confirmed is True, "pooled 54/21 clears the three-candidate threshold"
    assert out.fresh_sel_concordance == pytest.approx(38 / 52)


def test_a_reversed_re_draw_is_rejected_whatever_the_pooled_statistic_says():
    fresh, base = _maps(b=3, c=9)
    out = confirm_promotion(
        b1=30, c1=0, fresh_correctness=fresh, baseline_correctness=base, p_threshold=0.0076
    )
    assert out.confirmed is False
    assert "re-draw b2=3 <= c2=9" in out.reason
    # b2 == c2 is not "still favours the candidate" either.
    fresh, base = _maps(b=4, c=4)
    assert not confirm_promotion(
        b1=30, c1=0, fresh_correctness=fresh, baseline_correctness=base, p_threshold=0.0076
    ).confirmed


def test_a_re_draw_that_agrees_but_pools_short_of_the_threshold_is_rejected():
    """Draw 1: 5-0, exact p 0.03125 <= 0.05. Re-draw 3-2 agrees in sign, but
    8-2 over both draws has exact p 56/1024 = 0.0547 > 0.05."""

    fresh, base = _maps(b=3, c=2)
    out = confirm_promotion(
        b1=5, c1=0, fresh_correctness=fresh, baseline_correctness=base, p_threshold=0.05
    )
    assert out.method == "exact"
    assert out.p_pooled == pytest.approx(56 / 1024)
    assert out.confirmed is False
    assert "pooled p=0.0547 > p_threshold=0.0500" in out.reason
    # One more fixed pair on the re-draw and it pools to 9-2: p 67/2048.
    fresh, base = _maps(b=4, c=2)
    again = confirm_promotion(
        b1=5, c1=0, fresh_correctness=fresh, baseline_correctness=base, p_threshold=0.05
    )
    assert again.p_pooled == pytest.approx(67 / 2048) and again.confirmed is True


def test_the_threshold_is_the_one_draw_one_was_held_to():
    """The same pooled 20/6 (z 2.75, normal branch, p 0.0030) confirms at the
    plain threshold and fails the one-z-higher one a strong-evidence
    candidate carries (3.43, p 0.0003)."""

    fresh, base = _maps(b=10, c=3)
    plain = one_sided_p_from_z(2.43)
    strong = one_sided_p_from_z(3.43)
    assert confirm_promotion(
        b1=10, c1=3, fresh_correctness=fresh, baseline_correctness=base, p_threshold=plain
    ).confirmed
    assert not confirm_promotion(
        b1=10, c1=3, fresh_correctness=fresh, baseline_correctness=base, p_threshold=strong
    ).confirmed


def test_a_failed_re_score_is_an_unconfirmed_promotion():
    out = confirm_promotion(
        b1=8, c1=0, fresh_correctness=None, baseline_correctness={}, p_threshold=0.0076
    )
    assert out.confirmed is False
    assert out.b2 is None and out.c2 is None and out.p_pooled is None
    assert out.method == "none"
    assert json.loads(json.dumps(out.to_dict()))["confirmed"] is False


def test_the_record_round_trips_through_json():
    fresh, base = _maps(b=6, c=1)
    out = confirm_promotion(
        b1=7, c1=1, fresh_correctness=fresh, baseline_correctness=base, p_threshold=0.0076
    )
    payload = json.loads(json.dumps(out.to_dict()))
    assert payload["b1"] == 7 and payload["b2"] == 6 and payload["c2"] == 1
    assert payload["confirmed"] is True
    assert isinstance(PromotionConfirmation(**out.to_dict()), PromotionConfirmation)


# --- run level: confirmed on both draws -> promoted ------------------------------


@pytest.fixture(scope="module")
def confirmed_run(tmp_path_factory):
    run = prepare(tmp_path_factory.mktemp("confirmed"))
    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="confirmed"
    )
    return optimizer.run(), optimizer, run


def test_a_candidate_confirmed_on_both_draws_is_promoted(confirmed_run):
    run_dir, optimizer, run = confirmed_run
    (row,) = _summary(run_dir)

    assert row["gate_decision"] == "accepted"
    assert row["selected"] == "iter000_00_truth"
    assert row["gate_confirmed"] == "True"
    # A deterministic scorer re-draws the same cells: b2 = b1, c2 = c1 = 0.
    assert row["sel_b2"] == row["sel_b"] and row["sel_c2"] == row["sel_c"] == "0"
    b = int(row["sel_b"])
    assert float(row["sel_p_pooled"]) == pytest.approx(0.5 ** (2 * b), abs=1e-6), "6 dp in the CSV"
    assert row["promoted_rescore_delta"] != "", "still recorded on a confirmed promotion"
    assert float(row["promoted_rescore_delta"]) == pytest.approx(0.0)
    assert row["patience"] == "0"

    gate = _selection(run_dir, 0)["gate"]
    assert gate["decision"] == "accepted"
    confirmation = gate["confirmation"]
    assert confirmation["confirmed"] is True
    assert confirmation["b1"] == confirmation["b2"] == b
    assert confirmation["p_threshold"] == pytest.approx(
        _selection(run_dir, 0)["per_candidate"]["iter000_00_truth"]["p_threshold"]
    )
    # The promoted filter's scores are the baseline's now, and were persisted
    # under the promoted candidate's fingerprint.
    saved = json.loads(baseline_scores_path(run_dir).read_text("utf-8"))
    assert saved["iteration"] == 0
    promoted = load_filter(run_dir / "iterations" / "000" / "candidates" / "iter000_00_truth")
    assert optimizer._filter_fingerprint(promoted) == saved["filter_fingerprint"]
    assert optimizer._filter_fingerprint(load_filter(run.initial)) != saved["filter_fingerprint"]


def test_the_columns_are_empty_when_the_baseline_was_kept_on_draw_one(tmp_path):
    from tests.selection_gate_fixture import TRAIN_ONLY_POLICY

    run = prepare(tmp_path)
    run_dir = build_optimizer(
        run, script={0: [("trainonly", TRAIN_ONLY_POLICY)]}, iterations=1, run_id="kept"
    ).run()
    (row,) = _summary(run_dir)
    assert row["gate_decision"] == "kept"
    assert row["gate_confirmed"] == "" and row["sel_b2"] == "" and row["sel_p_pooled"] == ""
    assert _selection(run_dir, 0)["gate"]["confirmation"] is None


# --- run level: the re-draw reverses -> rejected, baseline stays ------------------


class Reversing(MetaScorer):
    """Deterministic on draw 1; every score negated inside `fresh_draw()`."""

    def __init__(self) -> None:
        self.reversed = False
        self.fresh_draws = 0

    @contextmanager
    def fresh_draw(self):
        self.reversed = True
        self.fresh_draws += 1
        try:
            yield
        finally:
            self.reversed = False

    def score(self, trajectory, criteria, rubric_md=""):
        scores, evidence, cost = super().score(trajectory, criteria, rubric_md)
        if self.reversed:
            scores = {name: -value for name, value in scores.items()}
        return scores, evidence, cost


@pytest.fixture(scope="module")
def reversed_run(tmp_path_factory):
    run = prepare(tmp_path_factory.mktemp("reversed"))
    optimizer = build_optimizer(
        run,
        script={0: [("truth", TRUTH_POLICY)], 1: [("truth", TRUTH_POLICY)]},
        iterations=2,
        run_id="reversed",
        patience=3,
        min_iterations=1,
    )
    scorer = Reversing()
    optimizer.solver = Solver(rubric_scorer=scorer)
    return optimizer.run(), optimizer, run, scorer


def test_a_candidate_whose_re_draw_reverses_is_rejected_and_the_baseline_stays(reversed_run):
    run_dir, optimizer, run, scorer = reversed_run
    rows = _summary(run_dir)

    assert scorer.fresh_draws == 2, "one re-score per draw-1 acceptance"
    for k, row in enumerate(rows):
        assert row["gate_decision"] == "rejected_on_redraw", k
        assert row["selected"] == f"iter{k:03d}_baseline"
        assert row["is_baseline"] == "True"
        # Draw 1 passed 8-0; negated, the re-draw loses every pair the
        # baseline orders (the 8 it fixed and every pair both had right).
        assert int(row["sel_b"]) > int(row["sel_c"]) == 0
        assert row["sel_b2"] == "0" and int(row["sel_c2"]) > int(row["sel_b"])
        assert row["gate_confirmed"] == "False"
        assert float(row["sel_p_pooled"]) > 0.99, "pooled 8 vs 52: no evidence for the candidate"
        assert row["promoted_rescore_delta"] == "", "nothing was promoted"
    assert [r["patience"] for r in rows] == ["1", "2"], "a rejected re-draw counts as kept"


def test_the_rejection_is_recorded_with_both_draws(reversed_run):
    run_dir, *_ = reversed_run
    selection = _selection(run_dir, 0)
    gate = selection["gate"]

    assert selection["selected_candidate"] == "iter000_baseline"
    assert "REJECTED ON RE-DRAW of iter000_00_truth" in selection["reason"]
    assert gate["decision"] == "rejected_on_redraw"
    assert gate["b"] == 8 and gate["c"] == 0
    confirmation = gate["confirmation"]
    assert confirmation["confirmed"] is False
    assert (confirmation["b1"], confirmation["c1"], confirmation["b2"]) == (8, 0, 0)
    # Every pair the baseline had right is one the negated candidate has wrong.
    baseline_right = round(gate["baseline_sel_concordance"] * 60)
    assert confirmation["c2"] == baseline_right > 8
    assert f"re-draw b2=0 <= c2={baseline_right}" in confirmation["reason"]
    assert confirmation["fresh_sel_concordance"] == pytest.approx(0.0)
    # The challenger's draw-1 statistics are still on record.
    assert selection["per_candidate"]["iter000_00_truth"]["eligible"] is True


def test_the_rejected_winners_scores_are_adopted_nowhere(reversed_run):
    run_dir, optimizer, run, _ = reversed_run

    # The filter iteration 1 started from is still the initial one: iteration
    # 0's draw-1 winner was never promoted.
    current = load_filter(run_dir / "iterations" / "001" / "current_filter")
    assert current.rubric_md == load_filter(run.initial).rubric_md
    assert Path(current.directory, "policy.py").read_text("utf-8") == Path(
        run.initial, "policy.py"
    ).read_text("utf-8")
    assert optimizer._filter_fingerprint(current) == optimizer._filter_fingerprint(
        load_filter(run.initial)
    )
    # The persisted baseline scores describe the initial filter, and are the
    # initial filter's own (non-reversed) scores on the selection rows.
    saved = json.loads(baseline_scores_path(run_dir).read_text("utf-8"))
    assert saved["filter_fingerprint"] == optimizer._filter_fingerprint(load_filter(run.initial))
    initial_scores = Solver(rubric_scorer=MetaScorer()).run(
        load_filter(run.initial), list(optimizer.selection_rows)
    )
    expected = {d.trajectory_id: float(d.rank_score) for d in initial_scores}
    assert saved["scores"] == pytest.approx(expected)
    assert optimizer._baseline_selection_scores == pytest.approx(expected)
    # And they are the correctness the draw-1 comparison used: the baseline
    # concordance the gate reports is the initial filter's.
    base_corr = pair_correctness(
        expected, optimizer.selection_set.pairs, ties=optimizer.selection_set.tied
    )
    assert _selection(run_dir, 1)["gate"]["baseline_sel_concordance"] == pytest.approx(
        sum(base_corr.values()) / len(base_corr)
    )
