"""The optimizer promotes on the hidden selection set, not on the train score.

The selection set, the paired gate, patience and the summary columns wired
together. The fixture's three filters make each branch of the gate reachable with a
deterministic scorer and judge: a candidate that is better on the selection
rows is accepted with `b > c`, one that is better on train but identical on
the selection rows is kept out, a promotion is re-scored, and patience ends
the run only after `min_iterations` and never on a proposer failure.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from meta_filter.optimizer import RUN_COMPLETE_FILE
from meta_filter.selection_set import SelectionSet, selection_set_path
from tests.selection_gate_fixture import (
    BASELINE_POLICY,
    TRAIN_ONLY_POLICY,
    TRUTH_POLICY,
    build_optimizer,
    prepare,
)


def _summary(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selection(run_dir: Path, k: int) -> dict:
    return json.loads((run_dir / "iterations" / f"{k:03d}" / "selection.json").read_text("utf-8"))


def _ranking(run_dir: Path, k: int) -> dict:
    return json.loads((run_dir / "iterations" / f"{k:03d}" / "ranking.json").read_text("utf-8"))


NEW_COLUMNS = [
    "sel_concordance",
    "sel_baseline_concordance",
    "sel_b",
    "sel_c",
    "sel_z",
    "sel_z_cluster",
    "gate_decision",
    "gate_threshold",
    "promoted_rescore_delta",
    "patience",
    "selection_pairs",
    # Sections C and E (`test_candidate_gating.py`), appended after these.
    "gated_out",
    "constraint_status",
    # The re-draw confirmation (`test_acceptance_confirmation.py`),
    # appended after those.
    "sel_b2",
    "sel_c2",
    "sel_p_pooled",
    "gate_confirmed",
]


# --- (b) + (d): a candidate better on the selection rows is accepted ----------


@pytest.fixture(scope="module")
def accepted_run(tmp_path_factory) -> tuple[Path, object]:
    run = prepare(tmp_path_factory.mktemp("accepted"))
    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="accepted"
    )
    return optimizer.run(), run


def test_a_candidate_better_on_selection_rows_is_accepted_with_b_over_c(accepted_run):
    run_dir, _ = accepted_run
    (row,) = _summary(run_dir)

    assert row["gate_decision"] == "accepted"
    assert row["selected"] == "iter000_00_truth"
    assert row["is_baseline"] == "False"
    assert int(row["sel_b"]) > int(row["sel_c"]) == 0
    assert float(row["sel_z"]) >= float(row["gate_threshold"])
    assert float(row["sel_concordance"]) > float(row["sel_baseline_concordance"])
    # Truth orders every judged pair the judge's way.
    assert float(row["sel_concordance"]) == pytest.approx(1.0)
    assert row["selection_pairs"] == "60"


def test_the_summary_columns_are_appended_after_the_train_ones(accepted_run):
    run_dir, _ = accepted_run
    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header[-len(NEW_COLUMNS) :] == NEW_COLUMNS
    assert header[0] == "iteration"
    assert "cost_critic" in header[: -len(NEW_COLUMNS)]


def test_selection_json_carries_the_gate_and_every_challenger(accepted_run):
    run_dir, _ = accepted_run
    selection = _selection(run_dir, 0)
    gate = selection["gate"]

    assert gate["gate"] == "mcnemar"
    assert gate["decision"] == "accepted"
    assert gate["b"] > gate["c"] == 0
    assert gate["z"] >= gate["threshold"]
    per_candidate = selection["per_candidate"]
    assert list(per_candidate) == ["iter000_00_truth"]
    stats = per_candidate["iter000_00_truth"]
    assert stats["eligible"] is True
    assert stats["blockers"] == []
    assert {"b", "c", "z", "p", "z_cluster", "sel_concordance", "threshold"} <= set(stats)


def test_ranking_json_diag_carries_the_paired_stats_per_candidate(accepted_run):
    run_dir, _ = accepted_run
    ranking = _ranking(run_dir, 0)
    assert ranking["selection_pairs"] == 60
    truth = ranking["candidates"]["iter000_00_truth"]
    baseline = ranking["candidates"]["iter000_baseline"]
    assert truth["sel_b"] > 0 and truth["sel_c"] == 0 and truth["sel_z"] > 0
    assert truth["sel_concordance"] == pytest.approx(1.0)
    assert baseline["sel_b"] == baseline["sel_c"] == 0
    assert baseline["sel_concordance"] == pytest.approx(
        _selection(run_dir, 0)["gate"]["baseline_sel_concordance"]
    )


def test_a_promotion_records_its_rescore_delta(accepted_run):
    """Deterministic scorer, so the fresh draw equals the winning one: 0.0 —
    but the column is filled, which is what an LLM run's non-zero reads
    depend on."""

    run_dir, _ = accepted_run
    (row,) = _summary(run_dir)
    assert row["promoted_rescore_delta"] != ""
    assert float(row["promoted_rescore_delta"]) == pytest.approx(0.0)
    assert row["patience"] == "0"


def test_the_run_config_records_the_selection_arm(accepted_run):
    import yaml

    run_dir, run = accepted_run
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text("utf-8"))
    assert cfg["selection_data"] == str(run.selection)
    assert cfg["selection_pairs"] == 60
    assert cfg["selection_seed"] == 4242
    assert cfg["gate_z"] is None
    assert cfg["gate_min_discordant"] == 5
    assert cfg["patience"] is None


# --- (c): better on train, identical on selection rows → kept -----------------


def test_a_candidate_better_only_on_train_is_kept_out(tmp_path):
    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run, script={0: [("trainonly", TRAIN_ONLY_POLICY)]}, iterations=1, run_id="kept"
    )
    run_dir = optimizer.run()

    (row,) = _summary(run_dir)
    validation = json.loads((run_dir / "iterations" / "000" / "validation.json").read_text("utf-8"))
    by_id = {e["candidate_id"]: e for e in validation}
    # The candidate really is better on the train side...
    assert (
        by_id["iter000_00_trainonly"]["metrics"]["critic_agreement"]
        > by_id["iter000_baseline"]["metrics"]["critic_agreement"]
    )
    assert (
        by_id["iter000_00_trainonly"]["metrics"]["score"]
        > by_id["iter000_baseline"]["metrics"]["score"]
    )
    # ...and the gate still keeps the baseline: identical on the exam.
    assert row["gate_decision"] == "kept"
    assert row["selected"] == "iter000_baseline"
    assert row["sel_b"] == "0" and row["sel_c"] == "0"
    assert row["promoted_rescore_delta"] == ""
    assert row["patience"] == "1"
    stats = _selection(run_dir, 0)["per_candidate"]["iter000_00_trainonly"]
    assert stats["eligible"] is False
    assert any("min_discordant" in b for b in stats["blockers"])


def test_selection_set_is_built_once_and_reused_on_resume(tmp_path):
    run = prepare(tmp_path)
    script = {0: [("trainonly", TRAIN_ONLY_POLICY)], 1: [("truth", TRUTH_POLICY)]}

    first = build_optimizer(run, script=script, iterations=1, run_id="resumed")
    run_dir = first.run()
    path = selection_set_path(run_dir)
    before = SelectionSet.load(path)
    seen_selection = {tid for tid in run.judge.pair_ids_seen if tid in run.selection_ids}
    assert seen_selection, "the judge adjudicated selection pairs during the build"

    second = build_optimizer(run, script=script, iterations=2, run_id="resumed", resume=True)
    assert second.run() == run_dir

    after = SelectionSet.load(path)
    assert after.to_dict() == before.to_dict(), "the exam did not move under the run"
    assert after.provenance["built_at"] == before.provenance["built_at"]
    # No selection pair was asked again: the judge only saw train pairs.
    new_ids = run.judge.pair_ids_seen - set(run.train_ids)
    assert new_ids == seen_selection
    rows = _summary(run_dir)
    assert [r["iteration"] for r in rows] == ["0", "1"]
    assert [r["gate_decision"] for r in rows] == ["kept", "accepted"]


# --- (e): early stop ------------------------------------------------------------


def test_patience_stops_after_min_iterations_and_proposer_failures_do_not_count(tmp_path):
    run = prepare(tmp_path)
    script = {
        0: [("trainonly", TRAIN_ONLY_POLICY)],  # kept → patience 1
        1: RuntimeError("provider 503"),  # failure → unchanged (would be 2 if counted)
        2: [("trainonly", TRAIN_ONLY_POLICY)],  # kept → patience 2 → stop
        3: [("truth", TRUTH_POLICY)],  # never reached
    }
    optimizer = build_optimizer(
        run, script=script, iterations=6, run_id="patience", patience=2, min_iterations=2
    )
    run_dir = optimizer.run()

    rows = _summary(run_dir)
    assert [r["iteration"] for r in rows] == ["0", "2"]
    assert [r["patience"] for r in rows] == ["1", "2"]
    assert not (run_dir / "iterations" / "003").exists()
    complete = json.loads((run_dir / RUN_COMPLETE_FILE).read_text("utf-8"))
    assert complete["stop_reason"] == "early_stop"
    assert complete["last_iteration"] == 2
    assert complete["iterations_run"] == 3
    assert complete["iterations_planned"] == 6
    assert complete["exit_status"] == 0


def test_min_iterations_holds_the_stop_and_an_acceptance_resets_patience(tmp_path):
    run = prepare(tmp_path)
    script = {
        0: [("trainonly", TRAIN_ONLY_POLICY)],  # kept → 1
        1: [("trainonly", TRAIN_ONLY_POLICY)],  # kept → 2, but k+1=2 < min_iterations=4
        2: [("truth", TRUTH_POLICY)],  # accepted → 0
        # Truth is now the baseline, and on the judged train pairs trainonly
        # orders every pair as it does (a no-op the precheck holds back), so
        # the kept iterations below score the noisy filter instead.
        3: [("noisy", BASELINE_POLICY)],  # kept → 1
        4: [("noisy", BASELINE_POLICY)],  # kept → 2 → stop (k+1=5 ≥ 4)
        5: [("noisy", BASELINE_POLICY)],
    }
    optimizer = build_optimizer(
        run, script=script, iterations=6, run_id="floor", patience=2, min_iterations=4
    )
    run_dir = optimizer.run()

    rows = _summary(run_dir)
    assert [r["iteration"] for r in rows] == ["0", "1", "2", "3", "4"]
    assert [r["patience"] for r in rows] == ["1", "2", "0", "1", "2"]
    assert [r["gate_decision"] for r in rows] == ["kept", "kept", "accepted", "kept", "kept"]
    complete = json.loads((run_dir / RUN_COMPLETE_FILE).read_text("utf-8"))
    assert complete["stop_reason"] == "early_stop"
    assert complete["last_iteration"] == 4


def test_an_iteration_with_no_loadable_candidate_leaves_patience_alone(tmp_path):
    run = prepare(tmp_path)
    script = {
        0: [("trainonly", TRAIN_ONLY_POLICY)],  # kept → 1
        1: [],  # nothing proposed: no candidate scored → unchanged
        2: [("trainonly", TRAIN_ONLY_POLICY)],  # kept → 2 → stop
    }
    optimizer = build_optimizer(
        run, script=script, iterations=5, run_id="empty", patience=2, min_iterations=1
    )
    run_dir = optimizer.run()
    rows = _summary(run_dir)
    assert [r["patience"] for r in rows] == ["1", "1", "2"]
    assert json.loads((run_dir / RUN_COMPLETE_FILE).read_text("utf-8"))["last_iteration"] == 2


def test_without_patience_the_run_uses_every_iteration(tmp_path):
    run = prepare(tmp_path)
    script = {k: [("trainonly", TRAIN_ONLY_POLICY)] for k in range(3)}
    run_dir = build_optimizer(run, script=script, iterations=3, run_id="full").run()
    assert [r["iteration"] for r in _summary(run_dir)] == ["0", "1", "2"]
    complete = json.loads((run_dir / RUN_COMPLETE_FILE).read_text("utf-8"))
    assert complete["stop_reason"] == "iterations_exhausted"
    assert complete["last_iteration"] == 2


# --- a pinned gate_z is honoured --------------------------------------------------


def test_a_pinned_gate_z_replaces_the_derived_threshold(tmp_path):
    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="pinned", gate_z=99.0
    )
    run_dir = optimizer.run()
    (row,) = _summary(run_dir)
    assert float(row["gate_threshold"]) == 99.0
    assert row["gate_decision"] == "kept", "no candidate clears z=99"
