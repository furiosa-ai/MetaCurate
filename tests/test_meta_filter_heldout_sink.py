"""Held-out metrics must not be readable from the proposer's workspace.

The train↔held-out gap is the only thing that would actually detect a
candidate memorizing the pool. It stops being a detector the moment the
proposer can read the held-out side and optimize against it.
"""

from __future__ import annotations

import json

import pytest

from meta_filter.experience_store import ExperienceStore
from meta_filter.schemas import CandidateEvaluation, FilterMetrics


def _metrics(accuracy: float) -> FilterMetrics:
    return FilterMetrics(
        total=1,
        critic_accept_total=1,
        critic_reject_total=0,
        accepted=1,
        rejected=0,
        reviewed=0,
        agree_accept=1,
        agree_reject=0,
        filter_accept_critic_reject=0,
        filter_reject_critic_accept=0,
        critic_reject_recall=0.0,
        critic_accept_retention=1.0,
        disputed_accept_rate=0.0,
        disputed_reject_rate=0.0,
        review_rate=0.0,
        critic_agreement=accuracy,
    )


def _evaluation(with_heldout: bool) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id="cand_00",
        is_baseline=True,
        metrics=_metrics(0.9),
        heldout_metrics=_metrics(0.42) if with_heldout else None,
    )


@pytest.fixture
def sink(tmp_path):
    return tmp_path / "_heldout" / "run.jsonl"


def test_heldout_metrics_leave_the_run_tree(tmp_path, sink):
    store = ExperienceStore(run_dir=tmp_path / "run", heldout_sink=sink)
    store.save_validation(0, [_evaluation(with_heldout=True)])

    written = json.loads((tmp_path / "run" / "iterations" / "000" / "validation.json").read_text())
    assert "heldout_metrics" not in written[0]
    # Train metrics stay — the proposer is supposed to see those.
    assert written[0]["metrics"]["critic_agreement"] == 0.9


def test_sink_records_enough_to_plot_the_gap(tmp_path, sink):
    store = ExperienceStore(run_dir=tmp_path / "run", heldout_sink=sink)
    store.save_validation(3, [_evaluation(with_heldout=True)])

    row = json.loads(sink.read_text().strip())
    assert row["iteration"] == 3
    assert row["candidate_id"] == "cand_00"
    assert row["is_baseline"] is True
    assert row["heldout_metrics"]["critic_agreement"] == 0.42


def test_sink_appends_across_iterations(tmp_path, sink):
    """One file per run, not per iteration — the gap is a time series."""

    store = ExperienceStore(run_dir=tmp_path / "run", heldout_sink=sink)
    for k in range(3):
        store.save_validation(k, [_evaluation(with_heldout=True)])

    rows = [json.loads(line) for line in sink.read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [0, 1, 2]


def test_no_sink_file_when_there_is_no_heldout_set(tmp_path, sink):
    """A run without validation data must not leave an empty sink behind."""

    store = ExperienceStore(run_dir=tmp_path / "run", heldout_sink=sink)
    store.save_validation(0, [_evaluation(with_heldout=False)])
    assert not sink.exists()


def test_without_a_sink_heldout_stays_inline(tmp_path):
    """Static runs keep the old layout — nothing reads the run dir there."""

    store = ExperienceStore(run_dir=tmp_path / "run")
    store.save_validation(0, [_evaluation(with_heldout=True)])

    written = json.loads((tmp_path / "run" / "iterations" / "000" / "validation.json").read_text())
    assert written[0]["heldout_metrics"]["critic_agreement"] == 0.42
