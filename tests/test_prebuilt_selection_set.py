"""A run graded on a prebuilt selection set, and `--selection-pairs all`.

Two runs launched in parallel would plan different pairs from the same
rows (`OptimizerConfig.selection_pairs`); with `selection_set` both copy one
file, and a resume must find the same pairs in the run's copy.
"""

from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
import typer
import yaml

from meta_filter.pair_judge import PairJudge
from meta_filter.selection_set import SelectionSet, selection_set_path
from scripts import build_selection_set as bss
from tests.selection_gate_fixture import (
    TRUTH_POLICY,
    TruthJudgeClient,
    build_optimizer,
    prepare,
    truth_by_id,
)


def _prebuild(run, root: Path, *, pairs="all", name="shared.json") -> Path:
    """Build a set with its OWN judge and cache, so the run's judge never sees it."""

    judge = PairJudge(
        client=TruthJudgeClient(truth_by_id(run.selection)), cache_dir=root / "prebuild_cache"
    )
    out = root / name
    bss.build(run.selection, judge, pairs=pairs, seed=1, exclude=[run.train], out=out)
    return out


def test_the_run_copies_the_prebuilt_set_and_judges_no_selection_pair(tmp_path):
    run = prepare(tmp_path)
    shared = _prebuild(run, tmp_path)
    optimizer = build_optimizer(
        run,
        script={0: [("truth", TRUTH_POLICY)]},
        iterations=1,
        run_id="prebuilt",
        selection_set=shared,
    )
    run_dir = optimizer.run()

    copy = selection_set_path(run_dir)
    assert filecmp.cmp(shared, copy, shallow=False), "byte-for-byte copy of the shared file"
    assert not (set(run.judge.pair_ids_seen) & set(run.selection_ids)), (
        "the run's judge adjudicated no selection pair — the set came prebuilt"
    )
    assert len(optimizer.selection_set) == 16 * 15 // 2
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text("utf-8"))
    assert cfg["selection_set"] == str(shared)
    assert cfg["selection_pairs"] is None
    assert cfg["selection_seed"] is None
    assert cfg["selection_data"] == str(run.selection)
    # The gate ran on it: the truth candidate is accepted with b > c.
    import csv

    (row,) = list(csv.DictReader((run_dir / "summary.csv").open(encoding="utf-8")))
    assert row["gate_decision"] == "accepted"
    assert int(row["sel_b"]) > int(row["sel_c"]) == 0
    assert row["selection_pairs"] == "120"


def test_a_resume_verifies_the_run_copy_against_the_prebuilt_set(tmp_path):
    run = prepare(tmp_path)
    shared = _prebuild(run, tmp_path)
    script = {0: [("truth", TRUTH_POLICY)]}
    first = build_optimizer(run, script=script, iterations=1, run_id="verify", selection_set=shared)
    run_dir = first.run()
    before = SelectionSet.load(selection_set_path(run_dir))

    same = build_optimizer(
        run, script=script, iterations=2, run_id="verify", selection_set=shared, resume=True
    )
    assert same.run() == run_dir
    assert SelectionSet.load(selection_set_path(run_dir)).to_dict() == before.to_dict()

    other = _prebuild(run, tmp_path, pairs=20, name="other.json")
    assert SelectionSet.load(other).pair_signature() != before.pair_signature()
    swapped = build_optimizer(
        run, script=script, iterations=3, run_id="verify", selection_set=other, resume=True
    )
    with pytest.raises(RuntimeError, match="different pairs"):
        swapped.run()
    assert SelectionSet.load(selection_set_path(run_dir)).to_dict() == before.to_dict()


def test_a_set_over_rows_missing_from_the_data_file_is_rejected(tmp_path):
    run = prepare(tmp_path)
    shared = _prebuild(run, tmp_path)
    foreign = SelectionSet.load(shared)
    foreign.row_ids = foreign.row_ids + ["elsewhere-000"]
    foreign.save(tmp_path / "foreign.json")

    optimizer = build_optimizer(
        run,
        script={0: [("truth", TRUTH_POLICY)]},
        iterations=1,
        run_id="foreign",
        selection_set=tmp_path / "foreign.json",
    )
    with pytest.raises(ValueError, match="elsewhere-000"):
        optimizer.run()
    assert not selection_set_path(run.runs_dir / "foreign").exists()


def test_selection_pairs_all_plans_every_pair_in_the_run(tmp_path):
    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run,
        script={0: [("truth", TRUTH_POLICY)]},
        iterations=1,
        run_id="allpairs",
        selection_pairs="all",
    )
    run_dir = optimizer.run()

    built = SelectionSet.load(selection_set_path(run_dir))
    assert built.provenance["pairs_mode"] == "all"
    assert built.provenance["n_planned"] == 120
    assert built.provenance["stratified"] is False
    assert len(built) == 120
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text("utf-8"))
    assert cfg["selection_pairs"] == "all"
    assert cfg["selection_set"] is None


def test_the_cli_parses_selection_pairs():
    from meta_filter.cli import _parse_selection_pairs

    assert _parse_selection_pairs(None) == 500
    assert _parse_selection_pairs("all") == "all"
    assert _parse_selection_pairs(" All ") == "all"
    assert _parse_selection_pairs("800") == 800
    for bad in ("0", "-1", "many"):
        with pytest.raises(typer.BadParameter):
            _parse_selection_pairs(bad)
