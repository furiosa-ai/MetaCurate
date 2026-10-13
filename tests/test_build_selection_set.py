"""`scripts.build_selection_set`: one judged set, built outside any run.

The fake judge is the fixture's `TruthJudgeClient` under the REAL `PairJudge`
and a temporary cache, so the idempotency test exercises the cache path a
second build actually takes rather than a stub of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meta_filter.gate.pair_judge import PairJudge
from meta_filter.gate.selection_set import SelectionSet
from scripts import build_selection_set as bss
from tests.selection_gate_fixture import TruthJudgeClient, truth_by_id, write_pool


def _judge(root: Path, *rows: Path) -> tuple[PairJudge, TruthJudgeClient]:
    client = TruthJudgeClient(truth_by_id(*rows))
    return PairJudge(client=client, cache_dir=root / "pair_cache"), client


@pytest.fixture
def pools(tmp_path: Path) -> dict[str, Path]:
    rows = tmp_path / "rows.jsonl"
    train = tmp_path / "train.jsonl"
    write_pool(rows, "sel", 6, train=False)
    write_pool(train, "train", 4, train=True)
    return {"rows": rows, "train": train}


def test_all_pairs_is_the_whole_universe_with_provenance(pools, tmp_path):
    judge, client = _judge(tmp_path, pools["rows"])
    out = tmp_path / "set.json"

    built = bss.build(
        pools["rows"],
        judge,
        pairs="all",
        seed=99,
        exclude=[pools["train"]],
        out=out,
        judge_model="fake/judge",
        extra_provenance={"judge_cache_dir": "x"},
    )

    assert len(built) == 15, "C(6, 2) pairs, all resolved by the truth judge"
    assert client.pair_calls == 30, "two orders per pair"
    on_disk = SelectionSet.load(out)
    assert on_disk.pair_signature() == built.pair_signature()
    prov = on_disk.provenance
    assert prov["pairs_mode"] == "all"
    assert prov["seed"] is None
    assert prov["stratified"] is False
    assert prov["n_planned"] == 15
    assert prov["n_resolved"] == 15
    assert prov["n_tied"] == 0
    assert prov["judge_model"] == "fake/judge"
    assert prov["rows_file"] == str(pools["rows"])
    assert prov["rows_sha256"] == bss.file_sha256(pools["rows"])
    assert prov["n_rows"] == 6
    assert prov["exclude"] == [str(pools["train"])]
    assert prov["judge_cache_dir"] == "x"
    assert prov["built_at"]
    assert on_disk.row_ids == sorted(bss.trajectory_ids(pools["rows"]))


def test_a_pair_count_is_a_uniform_sample_without_stratification(pools, tmp_path):
    judge, _ = _judge(tmp_path, pools["rows"])
    built = bss.build(pools["rows"], judge, pairs=7, seed=5, exclude=[], out=tmp_path / "s.json")
    assert len(built) == 7
    assert built.provenance["pairs_mode"] == "sample"
    assert built.provenance["seed"] == 5
    assert built.provenance["stratified"] is False
    assert built.provenance["n_hard"] == 0


def test_a_row_shared_with_an_excluded_pool_raises_before_any_judging(pools, tmp_path):
    judge, client = _judge(tmp_path, pools["rows"])
    with pytest.raises(ValueError, match="shared with") as exc:
        bss.build(
            pools["rows"], judge, pairs="all", seed=0, exclude=[pools["rows"]], out=tmp_path / "x"
        )
    assert str(pools["rows"]) in str(exc.value)
    assert client.pair_calls == 0
    assert not (tmp_path / "x").exists()


def test_a_second_build_hits_the_cache_and_leaves_the_file_alone(pools, tmp_path):
    out = tmp_path / "set.json"
    judge, client = _judge(tmp_path, pools["rows"])
    first = bss.build(pools["rows"], judge, pairs="all", seed=0, exclude=[], out=out)
    calls_after_first = client.pair_calls
    stamp = out.stat().st_mtime_ns

    again_judge, again_client = _judge(tmp_path, pools["rows"])
    second = bss.build(pools["rows"], again_judge, pairs="all", seed=0, exclude=[], out=out)

    assert again_client.pair_calls == 0, "every verdict came from the pair-judge cache"
    assert calls_after_first == 30
    assert second.pair_signature() == first.pair_signature()
    assert second.provenance["built_at"] == first.provenance["built_at"]
    assert out.stat().st_mtime_ns == stamp


def test_parse_pairs_accepts_all_or_a_positive_int():
    import argparse

    assert bss.parse_pairs("all") == "all"
    assert bss.parse_pairs("ALL") == "all"
    assert bss.parse_pairs("12") == 12
    for bad in ("0", "-3", "some"):
        with pytest.raises(argparse.ArgumentTypeError):
            bss.parse_pairs(bad)
