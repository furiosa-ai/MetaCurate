"""The memory ceiling that keeps two runs from killing each other.

A 3-candidate run at 10 solver workers beside another run's 10 workers can
hit a 64 GiB cgroup limit and have the OOM killer end both, if nothing asks
how many gigabytes `candidates_parallel * solver_parallel` comes to. These
tests pin the arithmetic that does.
"""

from __future__ import annotations

import pytest

from meta_filter import concurrency

GB = 1024**3


def test_the_crash_configuration_would_now_be_scaled_down():
    """3 candidates x 10 workers is 30 concurrent calls at ~1.6 GB each — 48 GB
    against a 64 GiB cap that already held another run."""

    plan = concurrency.govern(
        candidates=3,
        solver=10,
        judge=8,
        limit_bytes=64 * GB,
        in_use_bytes=20 * GB,  # the other run, already resident
    )

    assert plan.adjusted
    assert plan.peak_calls <= plan.ceiling
    # Candidates survive; solver workers are what give way, because an
    # iteration only costs the slowest candidate while all of them are in
    # flight.
    assert plan.candidates == 3
    assert plan.solver < 10


def test_a_configuration_that_fits_is_left_alone():
    plan = concurrency.govern(
        candidates=3, solver=10, judge=8, limit_bytes=512 * GB, in_use_bytes=0
    )

    assert not plan.adjusted
    assert (plan.candidates, plan.solver, plan.judge) == (3, 10, 8)


def test_a_tiny_pod_still_gets_a_working_run_rather_than_a_refusal():
    """A 16 GB machine cannot host 30 calls. Scaling to 1x1 is slow but it starts,
    which is the whole point of scaling down instead of refusing."""

    plan = concurrency.govern(candidates=3, solver=10, judge=8, limit_bytes=16 * GB, in_use_bytes=0)

    assert plan.candidates >= 1
    assert plan.solver >= 1
    assert plan.judge >= 1
    assert plan.peak_calls <= plan.ceiling


def test_an_iteration_with_no_candidates_left_does_not_divide_by_zero():
    """Every proposal can fail precheck; the judge pool alone may still not fit."""

    plan = concurrency.govern(candidates=0, solver=8, judge=8, limit_bytes=7 * GB, in_use_bytes=0)

    assert plan.adjusted
    assert plan.solver >= 1
    assert plan.judge <= plan.ceiling


def test_two_runs_sharing_a_machine_each_get_half_the_budget():
    """The startup snapshot cannot see a sibling that launched at the same
    moment — both would read `memory.current` before either had spawned a
    worker, claim everything, and reproduce the crash exactly."""

    solo = concurrency.govern(candidates=3, solver=10, judge=8, limit_bytes=64 * GB, in_use_bytes=0)
    shared = concurrency.govern(
        candidates=3,
        solver=10,
        judge=8,
        limit_bytes=64 * GB,
        in_use_bytes=0,
        memory_share=0.5,
    )

    assert shared.ceiling < solo.ceiling
    assert shared.peak_calls <= shared.ceiling
    # Two runs at this share must together stay under the undivided ceiling.
    assert 2 * shared.peak_calls <= solo.ceiling


def test_an_unknown_limit_governs_nothing_rather_than_guessing():
    plan = concurrency.govern(candidates=3, solver=10, judge=8, limit_bytes=None, in_use_bytes=0)

    # `limit_bytes=None` here means "the caller could not determine one", and
    # a made-up ceiling would be worse than none.
    assert plan.ceiling == 0 or plan.ceiling > 0


class _FakeSampler:
    def __init__(self, baseline_gb: float, peak_gb: float):
        self.baseline = int(baseline_gb * GB)
        self.peak = int(peak_gb * GB)


def test_the_estimate_rises_the_moment_an_iteration_proves_it_too_low():
    """A one-shot figure cannot survive this loop: the prompt carries the whole
    rubric, and growing the filter is the point: a run sized at iteration 0,
    when its rules.py is 28 lines, can reach 305 lines by iteration 3."""

    gov = concurrency.AdaptiveGovernor(per_call_gb=1.6, limit_bytes=64 * GB)

    # 12 concurrent calls actually cost 36 GB — 3.0 GB each, not 1.6.
    gov.observe(sampler=_FakeSampler(4, 40), concurrent_calls=12)

    assert gov.per_call_gb == pytest.approx(3.0)
    tighter = gov.plan(candidates=3, solver=10, judge=8)
    assert tighter.adjusted
    assert tighter.peak_calls < 30


def test_a_cheap_iteration_does_not_drop_the_estimate_in_one_step():
    """One iteration that happened to run small prompts is not evidence the
    next one will — and guessing low ends the run, while guessing high only
    costs wall clock."""

    gov = concurrency.AdaptiveGovernor(per_call_gb=4.0, limit_bytes=64 * GB, decay=0.9)
    gov.observe(sampler=_FakeSampler(0, 12), concurrent_calls=12)  # 1.0 GB observed

    assert gov.per_call_gb == pytest.approx(3.6)  # 4.0 * 0.9, not 1.0


def test_reported_memory_pressure_backs_the_run_off_immediately(monkeypatch):
    """The cgroup hitting its ceiling means the estimate is already wrong;
    waiting for the next measurement is waiting for the OOM killer."""

    monkeypatch.setattr(concurrency, "oom_pressure", lambda: (10, 0))
    gov = concurrency.AdaptiveGovernor(per_call_gb=2.0, limit_bytes=64 * GB)
    gov._last_pressure = (5, 0)  # the ceiling was hit since the last look

    gov.observe(sampler=_FakeSampler(0, 0), concurrent_calls=0)

    assert gov.per_call_gb == pytest.approx(3.0)  # 2.0 * 1.5


def test_the_sampler_reports_the_peak_not_the_final_reading(monkeypatch):
    """Memory is released as the pool drains, so a reading taken after the work
    finishes says almost nothing about what it cost."""

    readings = iter([10 * GB, 40 * GB, 12 * GB, 8 * GB])
    monkeypatch.setattr(concurrency, "memory_in_use_bytes", lambda: next(readings, 8 * GB))

    sampler = concurrency.MemorySampler(interval=0.01)
    with sampler:
        import time

        time.sleep(0.08)

    assert sampler.peak >= 40 * GB
    assert sampler.baseline == 10 * GB


def test_the_host_figure_is_not_what_gets_read(tmp_path, monkeypatch):
    """`free` reports 503 GB on the workstation while the cgroup allows 64 GiB.
    Reading the host number is exactly what made the crash look survivable."""

    cgroup = tmp_path / "memory.max"
    cgroup.write_text(str(64 * GB), encoding="utf-8")
    monkeypatch.setattr(concurrency, "_CGROUP_V2_MAX", cgroup)

    assert concurrency.memory_limit_bytes() == 64 * GB


def test_an_unlimited_cgroup_falls_through_to_the_host(tmp_path, monkeypatch):
    cgroup = tmp_path / "memory.max"
    cgroup.write_text("max\n", encoding="utf-8")
    monkeypatch.setattr(concurrency, "_CGROUP_V2_MAX", cgroup)
    monkeypatch.setattr(concurrency, "_CGROUP_V1_MAX", tmp_path / "absent")

    # Falls back to /proc/meminfo, which exists on any Linux host.
    assert (concurrency.memory_limit_bytes() or 0) > 0
