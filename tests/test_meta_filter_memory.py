"""Regression memory insertion + deduplication."""

from __future__ import annotations

from meta_filter.memory import RegressionMemory
from meta_filter.schemas import CounterExample, Decision, Trajectory


def _ce(id_: str, kind: str, gold: Decision) -> CounterExample:
    return CounterExample(
        trajectory_id=id_,
        source_iteration=0,
        filter_version="v0",
        critic_decision=gold,
        predicted_decision=Decision.ACCEPT if gold == Decision.REJECT else Decision.REJECT,
        failure_type=kind,
        raw_trajectory=Trajectory(trajectory_id=id_, task="t", turns=[], critic_decision=gold),
    )


def test_fn_lands_in_must_reject_and_fp_lands_in_must_accept(tmp_path):
    mem = RegressionMemory(base_dir=tmp_path)
    mem.add(_ce("bad_1", "false_negative", Decision.REJECT))
    mem.add(_ce("good_1", "false_positive", Decision.ACCEPT))
    mem.add(_ce("boundary_1", "boundary", Decision.ACCEPT))

    assert [c.trajectory_id for c in mem.must_reject()] == ["bad_1"]
    assert [c.trajectory_id for c in mem.must_accept()] == ["good_1"]
    assert [c.trajectory_id for c in mem.boundary()] == ["boundary_1"]


def test_memory_dedups_by_trajectory_id(tmp_path):
    mem = RegressionMemory(base_dir=tmp_path)
    assert mem.add(_ce("bad_1", "false_negative", Decision.REJECT)) is True
    assert mem.add(_ce("bad_1", "false_negative", Decision.REJECT)) is False
    assert len(mem.must_reject()) == 1
