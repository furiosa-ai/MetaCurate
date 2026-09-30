"""The promotion re-score is the ONE draw everyone reads afterwards.

The loop re-scores a promoted candidate so the baseline carries a fresh draw
rather than the draw that won selection. The fresh draw must overwrite the
cached cells under their normal key, so that

* a next-iteration candidate whose rubric.md is byte-identical (a rules or
  policy edit) reads exactly the cells the baseline carries — `b = c = 0`,
  not scorer noise;
* the adopted scores are persisted to `runs/_selection/<run_id>.baseline_scores.json`
  and a resumed run reads them back;
* a re-score that fails outright cannot confirm the acceptance:
  the baseline is kept, its scores untouched, and the run goes on.

The scorer here is the real `LLMRubricScorer` over a client that answers
from the row's task line plus a per-call jitter, so every fresh draw is a
different number with the same ordering — the shape of real scorer noise,
made deterministic.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from meta_filter.filter_loader import load_filter
from meta_filter.llm.base import LLMResponse
from meta_filter.llm_scorer import LLMRubricScorer
from meta_filter.selection_set import baseline_scores_path, score_on_selection
from meta_filter.solver import Solver
from tests.selection_gate_fixture import (
    TRUTH_POLICY,
    MetaScorer,
    build_optimizer,
    prepare,
)

# Rows are `task <prefix> <i> ...`; `truth` rises with i, `noisy` inverts
# every even-indexed adjacent pair, exactly as the fixture's task numbers do
# but on the scorer's [-2, 2] scale. The jitter is far below every gap.
STEP = 0.25
FLIP = 0.18
JITTER = 1e-5


@dataclass
class JitteredClient:
    """Answers per criterion from the task line; every call moves by JITTER."""

    model: str = "jitter"
    calls: int = 0
    fresh_calls_by_row: dict[str, int] = field(default_factory=dict)

    def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        self.calls += 1
        task = re.search(r"^Task:\n(.+)$", prompt, flags=re.MULTILINE).group(1)
        prefix, index = task.split()[1], int(task.split()[2])
        criterion = next(
            line.removeprefix("Criteria to score (in order): ")
            for line in prompt.splitlines()
            if line.startswith("Criteria to score (in order): ")
        )
        truth = -2.0 + STEP * index
        noisy = truth + (FLIP if index % 2 == 0 else -FLIP)
        base = {
            "Truth": truth,
            "Noisy": noisy,
            "TrainBoost": (truth - noisy) if prefix == "train" else 0.0,
        }[criterion]
        value = max(-2.0, min(2.0, base + JITTER * self.calls))
        row = " ".join(task.split()[:3])
        self.fresh_calls_by_row[row] = self.fresh_calls_by_row.get(row, 0) + 1
        body = json.dumps(
            {"scores": {criterion: {"score": value, "evidence": f"call {self.calls}"}}}
        )
        return LLMResponse(text=body, prompt_tokens=1, completion_tokens=1)


# TRUTH_POLICY with a comment: a different candidate directory whose rubric
# and criteria are byte-identical to the promoted filter's, so every one of
# its selection cells is a cache hit.
TRUTH_POLICY_RESTATED = TRUTH_POLICY.replace(
    "def decide(", "# same mechanism, restated\ndef decide(", 1
)


def _summary(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selection(run_dir: Path, k: int) -> dict:
    return json.loads((run_dir / "iterations" / f"{k:03d}" / "selection.json").read_text("utf-8"))


def _llm_optimizer(tmp_path: Path, *, script, iterations: int, run_id: str, **kwargs):
    run = prepare(tmp_path)
    optimizer = build_optimizer(run, script=script, iterations=iterations, run_id=run_id, **kwargs)
    client = JitteredClient()
    optimizer.solver = Solver(
        rubric_scorer=LLMRubricScorer(client=client, cache_dir=tmp_path / "scorer_cache"),
        lazy_scoring=False,
    )
    return optimizer, run, client


@pytest.fixture(scope="module")
def promoted_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("promoted")
    optimizer, run, client = _llm_optimizer(
        tmp_path,
        script={0: [("truth", TRUTH_POLICY)], 1: [("restated", TRUTH_POLICY_RESTATED)]},
        iterations=2,
        run_id="canonical",
    )
    run_dir = optimizer.run()
    return run_dir, optimizer, run, client, tmp_path


def test_the_winner_is_promoted_and_re_scored_on_a_noisy_scorer(promoted_run):
    run_dir, optimizer, run, client, _ = promoted_run
    rows = _summary(run_dir)
    assert rows[0]["gate_decision"] == "accepted"
    assert rows[0]["selected"] == "iter000_00_truth"
    # Every selection row was asked again at promotion: the fresh draw is a
    # different number (the jitter moved) with the same ordering, so the
    # concordance delta is 0 while the cells themselves changed.
    assert rows[0]["promoted_rescore_delta"] != ""
    assert float(rows[0]["promoted_rescore_delta"]) == pytest.approx(0.0)
    for tid in run.selection_ids:
        task = f"task sel {int(tid.split('-')[1].split('::')[0])}"
        assert client.fresh_calls_by_row[task] >= 2, task


def test_the_fresh_draw_is_what_the_cache_now_holds(promoted_run):
    """A new scorer over the same cache directory — a resumed run, a later
    candidate — reads back exactly the persisted baseline scores."""

    run_dir, optimizer, run, client, tmp_path = promoted_run
    saved = json.loads(baseline_scores_path(run_dir).read_text("utf-8"))
    assert saved["run_id"] == "canonical"
    assert saved["iteration"] == 0, "written at the promotion, not only at set build"
    assert set(saved["scores"]) == set(run.selection_ids)

    promoted = load_filter(run_dir / "iterations" / "001" / "current_filter")
    reader = LLMRubricScorer(client=JitteredClient(), cache_dir=tmp_path / "scorer_cache")
    from_cache = score_on_selection(
        promoted, Solver(rubric_scorer=reader, lazy_scoring=False), list(optimizer.selection_rows)
    )
    assert reader.client.calls == 0, "every cell is a cache hit"
    assert from_cache == pytest.approx(saved["scores"])
    assert optimizer._baseline_selection_scores == pytest.approx(saved["scores"])
    assert not list((tmp_path / "scorer_cache" / "per_criterion").glob("*__salt_*"))


def test_a_rubric_identical_candidate_next_iteration_reads_the_baselines_cells(promoted_run):
    """With the fresh draw filed beside the winning one, this candidate would
    read the lucky cells while the baseline carried the fresh ones, and its
    b - c would be pure scorer noise. Filed under the normal key, it reads the
    baseline's own cells: the precheck holds it out as a no-op, and scoring it
    on the selection rows from the cache reproduces the baseline exactly."""

    run_dir, optimizer, _run, _client, tmp_path = promoted_run
    rows = _summary(run_dir)
    assert rows[1]["gate_decision"] == "kept"
    precheck = json.loads((run_dir / "iterations" / "001" / "precheck.json").read_text("utf-8"))
    assert "no-op" in precheck["gated_out"]["iter001_00_restated"]["reason"]

    candidate = load_filter(run_dir / "iterations" / "001" / "candidates" / "iter001_00_restated")
    reader = LLMRubricScorer(client=JitteredClient(), cache_dir=tmp_path / "scorer_cache")
    from_cache = score_on_selection(
        candidate, Solver(rubric_scorer=reader, lazy_scoring=False), list(optimizer.selection_rows)
    )
    assert reader.client.calls == 0, "every cell is a cache hit"
    saved = json.loads(baseline_scores_path(run_dir).read_text("utf-8"))["scores"]
    assert from_cache == pytest.approx(saved)


def test_a_resumed_run_loads_the_persisted_baseline_scores(promoted_run):
    run_dir, optimizer, run, _, tmp_path = promoted_run
    saved = json.loads(baseline_scores_path(run_dir).read_text("utf-8"))["scores"]

    promoted = load_filter(run_dir / "iterations" / "001" / "current_filter")
    assert optimizer._load_baseline_selection_scores(promoted) == pytest.approx(saved)
    # The file fingerprints the filter it describes; another filter re-scores.
    assert optimizer._load_baseline_selection_scores(load_filter(run.initial)) is None

    # End to end: `_ensure_selection_set` on a resumed optimizer picks them up
    # before iteration 0 spends anything.
    resumed = build_optimizer(
        run,
        script={0: [("truth", TRUTH_POLICY)], 1: [("restated", TRUTH_POLICY_RESTATED)]},
        iterations=2,
        run_id="canonical",
        resume=True,
    )
    resumed._ensure_selection_set(promoted)
    assert resumed._baseline_selection_scores == pytest.approx(saved)


def test_set_build_persists_the_initial_filters_scores(tmp_path):
    optimizer, run, _ = _llm_optimizer(tmp_path, script={0: []}, iterations=1, run_id="initial")
    run_dir = optimizer.run()
    saved = json.loads(baseline_scores_path(run_dir).read_text("utf-8"))
    assert saved["iteration"] is None, "set build, before any iteration"
    assert set(saved["scores"]) == set(run.selection_ids)


# --- a failed re-score cannot confirm: the baseline is kept, the run goes on --


class _RefusingScorer(MetaScorer):
    """Deterministic scores, but the fresh-draw block dies like a transport."""

    def fresh_draw(self):
        raise RuntimeError("upstream 503 during the promotion re-score")


def test_a_failed_re_score_rejects_the_promotion_and_does_not_abort(tmp_path, caplog):
    """An acceptance is confirmed on the re-draw or not at all. A
    re-score that dies leaves nothing to confirm on, so the baseline stays —
    with the scores it had, not the winner's — and the iteration completes."""

    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="refused"
    )
    optimizer.solver = Solver(rubric_scorer=_RefusingScorer())

    run_dir = optimizer.run()

    (row,) = _summary(run_dir)
    assert row["gate_decision"] == "rejected_on_redraw"
    assert row["selected"] == "iter000_baseline"
    assert row["is_baseline"] == "True"
    assert row["promoted_rescore_delta"] == "", "not measured, not zero"
    assert row["gate_confirmed"] == "False"
    assert row["sel_b2"] == "" and row["sel_c2"] == "" and row["sel_p_pooled"] == ""
    assert "promotion re-score of iter000_00_truth failed" in caplog.text
    assert "REJECTED on the re-draw" in caplog.text
    # Draw 1's acceptance is still on record, with the reason it was undone.
    gate = _selection(run_dir, 0)["gate"]
    assert gate["decision"] == "rejected_on_redraw"
    assert gate["b"] > gate["c"] == 0, "draw 1 passed"
    assert gate["confirmation"]["confirmed"] is False
    assert gate["confirmation"]["b2"] is None
    assert "re-score failed" in gate["confirmation"]["reason"]
    # The baseline's scores are the initial filter's own; the winner's were
    # adopted nowhere. (They are re-read from the cache and re-saved after the
    # rejection, so the file names iteration 0 but fingerprints the initial.)
    saved = json.loads(baseline_scores_path(run_dir).read_text("utf-8"))
    assert saved["filter_fingerprint"] == optimizer._filter_fingerprint(load_filter(run.initial))
    initial_scores = Solver(rubric_scorer=MetaScorer()).run(
        load_filter(run.initial), list(optimizer.selection_rows)
    )
    expected = {d.trajectory_id: float(d.rank_score) for d in initial_scores}
    assert saved["scores"] == pytest.approx(expected)
    assert optimizer._baseline_selection_scores == pytest.approx(expected)
    assert set(saved["scores"]) == set(run.selection_ids)
