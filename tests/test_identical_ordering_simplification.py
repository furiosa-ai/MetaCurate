"""A simplification that orders every hidden pair as the baseline does is
promoted without a second draw.

The shape is a proposer removing rule functions that fire on no row. That
reads `b = 0, c = 0` at the gate — the only thing a zero-fire prune can read.
The true `b - c` is exactly 0, so a re-draw's `b2 - c2` would be pure scorer
noise and a one-sided `b2 >= c2` a coin flip: identical legitimate prunes
would be kept or refused at random.

So the re-score is not bought at all. The scorer here reverses every score
inside `fresh_draw()`, which is the flip forced to land tails: if the
promotion re-score ran, the candidate would read `b2 = 0` against a large
`c2` and be refused. `fresh_draws == 0` is the assertion that matters.

The pool differs from `test_simplification_acceptance_end_to_end` in one way
that the adoption choice turns on: `Dead` is a nonzero CONSTANT, so removing
it shifts every score by the same -5 and changes no ordering anywhere. The
baseline's score vector and the winner's are therefore different numbers with
identical per-pair credit — which is exactly what `b = c = 0` asserts, and
what tells the two candidate adoption rules apart.
"""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from meta_filter.core.filter_loader import load_filter
from meta_filter.core.solver import Solver
from meta_filter.gate.selection_set import baseline_scores_path
from tests.selection_gate_fixture import RULES, prepare
from tests.test_candidate_gating import INVERTED_NOISY_POLICY
from tests.test_simplification_acceptance_end_to_end import (
    BASELINE_WITH_DEAD,
    DEAD_RUBRIC,
    RUBRIC,
    SMALLER_POLICY,
    DeadAwareScorer,
    _build,
)

DEAD_CONSTANT = 5.0


class ReversingOnFreshDraw(DeadAwareScorer):
    """Deterministic on draw 1; every score negated inside `fresh_draw()`.

    A promotion re-score taken through this scorer reverses the candidate's
    order on every pair, so it reads `b2 = 0` and the old rule refused the
    promotion. Nothing here should ever enter the block.
    """

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


def _with_constant_dead(path: Path) -> None:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["task"] += f" dead={DEAD_CONSTANT}"
            rows.append(json.dumps(row))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _summary(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(run_dir: Path, k: int, name: str):
    return json.loads((run_dir / "iterations" / f"{k:03d}" / name).read_text("utf-8"))


@pytest.fixture(scope="module")
def identical_ordering_run(tmp_path_factory):
    run = prepare(tmp_path_factory.mktemp("identical_ordering"))
    (run.initial / "rubric.md").write_text(DEAD_RUBRIC, encoding="utf-8")
    (run.initial / "policy.py").write_text(BASELINE_WITH_DEAD, encoding="utf-8")
    _with_constant_dead(run.train)
    _with_constant_dead(run.selection)
    script = {
        # Iteration 0 disputes every pair, so iteration 1 has judged pairs and
        # a precheck; it loses, so nothing is promoted and nothing re-scored.
        0: [("inverted", INVERTED_NOISY_POLICY, DEAD_RUBRIC, RULES)],
        1: [("drop_dead", SMALLER_POLICY, RUBRIC, RULES)],
    }
    optimizer = _build(run, script, run_id="identical_ordering")
    scorer = ReversingOnFreshDraw()
    optimizer.solver = Solver(rubric_scorer=scorer)
    return optimizer.run(), optimizer, run, scorer


def test_a_zero_fire_prune_is_promoted_and_never_re_scored(identical_ordering_run):
    run_dir, _optimizer, _run, scorer = identical_ordering_run
    rows = _summary(run_dir)

    assert [r["gate_decision"] for r in rows] == ["kept", "accepted_simplification"]
    assert rows[1]["selected"] == "iter001_00_drop_dead"
    assert rows[1]["is_baseline"] == "False"
    assert (rows[1]["sel_b"], rows[1]["sel_c"]) == ("0", "0"), "identical ordering on every pair"
    assert rows[1]["gate_confirmed"] == "True"
    # No second draw was taken, so there are no re-draw counts to print and no
    # delta between two draws. `gate_confirmed`, not the emptiness, is what
    # separates this from a re-score that failed.
    assert rows[1]["sel_b2"] == "" and rows[1]["sel_c2"] == "" and rows[1]["sel_p_pooled"] == ""
    assert rows[1]["promoted_rescore_delta"] == ""
    assert scorer.fresh_draws == 0, "the promotion re-score must not be bought at all"


def test_the_recorded_reason_says_why_no_second_draw_was_needed(identical_ordering_run):
    run_dir, *_ = identical_ordering_run
    selection = _json(run_dir, 1, "selection.json")

    assert selection["selected_candidate"] == "iter001_00_drop_dead"
    assert "PROMOTION of iter001_00_drop_dead CONFIRMED" in selection["reason"]
    assert "ordering identical on all 60 selection pairs" in selection["reason"]
    assert "smaller in every complexity component" in selection["reason"]
    assert "confirmed without a second draw" in selection["reason"]
    # `record.json` carries the same sentence, which is what a human reads.
    assert _json(run_dir, 1, "record.json")["selected_reason"] == selection["reason"]

    confirmation = selection["gate"]["confirmation"]
    assert confirmation["confirmed"] is True
    assert confirmation["method"] == "identical_ordering"
    assert (confirmation["b1"], confirmation["c1"]) == (0, 0)
    assert confirmation["b2"] is confirmation["c2"] is None
    assert confirmation["p_pooled"] is confirmation["z_pooled"] is None
    assert confirmation["fresh_sel_concordance"] is None
    assert confirmation["reason"] == (
        "ordering identical on all 60 selection pairs (draw 1 b/c 0/0); "
        "smaller in every complexity component; confirmed without a second draw"
    )


def test_the_winner_becomes_the_baseline_carrying_its_own_selection_scores(
    identical_ordering_run,
):
    """The stored scores are the WINNER's, not the outgoing baseline's.

    They order every pair the same way — that is what `b = c = 0` says — but
    they are not the same numbers: dropping the constant `Dead` axis shifts
    every score by -5. What the next iteration needs is that the carried
    scores equal what the baseline FILTER scores, so a rubric-identical
    challenger reads `b = c = 0` instead of noise; only the winner's own
    numbers do that.
    """

    run_dir, optimizer, run, scorer = identical_ordering_run
    promoted = load_filter(run_dir / "iterations" / "001" / "candidates" / "iter001_00_drop_dead")

    saved = json.loads(baseline_scores_path(run_dir).read_text("utf-8"))
    assert saved["iteration"] == 1
    assert optimizer._filter_fingerprint(promoted) == saved["filter_fingerprint"]
    assert optimizer._filter_fingerprint(load_filter(run.initial)) != saved["filter_fingerprint"]

    scores = {tid: float(v) for tid, v in saved["scores"].items()}
    assert scores == optimizer._score_on_selection_rows(promoted)
    assert scorer.fresh_draws == 0, "and that check bought no fresh draw either"
    # The outgoing baseline sat exactly DEAD_CONSTANT higher on every row.
    baseline_scores = optimizer._score_on_selection_rows(load_filter(run.initial))
    assert scores == {tid: pytest.approx(v - DEAD_CONSTANT) for tid, v in baseline_scores.items()}
    assert optimizer.selection_set.concordance(scores) == pytest.approx(
        optimizer.selection_set.concordance(baseline_scores)
    ), "identical ordering, different numbers — which is the whole point"
