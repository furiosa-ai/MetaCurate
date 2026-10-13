"""Derive a safe concurrency ceiling from the memory this process may use.

Left alone, `candidates_parallel * solver_parallel` is pure multiplication
and nobody asks how many gigabytes the product comes to: a 3-candidate run
at 10 solver workers beside another run can put 40 headless processes on one
cgroup, hit its memory ceiling, and have the OOM killer end both runs
mid-iteration.

So the ceiling is a first-class input: measure what is available,
divide by what one headless call costs, and scale the requested knobs down to
fit. Scaling down is deliberate — a long experiment must not fail to start
over a config mistake — so every adjustment is logged loudly enough to explain
a disappointing wall clock later.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Peak RSS of one `claude -p` headless call, measured on this workstation at
# 0.41 GB for a trivial prompt. Scorer and judge prompts carry a whole
# trajectory, so the working number is several times that; 1.6 GB is what an
# observed out-of-memory failure implied (64 GiB / 40 concurrent calls at the
# breaking point). Override per-run when a pool's trajectories are unusually
# large.
DEFAULT_PER_CALL_GB = 1.6

# Fraction of the limit the loop may claim. The rest absorbs the optimizer
# itself, the page cache, and the fact that per-call footprint is an average
# rather than a bound.
DEFAULT_HEADROOM = 0.75

_CGROUP_V2_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V2_CURRENT = Path("/sys/fs/cgroup/memory.current")
_CGROUP_V1_MAX = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_MEMINFO = Path("/proc/meminfo")


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text == "max":  # cgroup v2 for "no limit"
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _meminfo_total_bytes() -> int | None:
    try:
        for line in _MEMINFO.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def memory_limit_bytes() -> int | None:
    """How much memory this process is actually allowed, not what the host has.

    `free` reports the host: on the workstation that is 503 GB while the
    cgroup caps this session at 64 GiB. Reading the host figure is what makes
    an over-parallelized run look survivable right up until it is killed.
    """

    for path in (_CGROUP_V2_MAX, _CGROUP_V1_MAX):
        limit = _read_int(path)
        # cgroup v1 writes a sentinel near 2**63 to mean "unlimited".
        if limit is not None and limit < (1 << 62):
            return limit
    return _meminfo_total_bytes()


def memory_in_use_bytes() -> int:
    """Bytes already charged to this cgroup, so a second run sees the first.

    This is a snapshot taken at startup. Two runs launched simultaneously will
    each see the other at close to zero and both claim the full budget — the
    case that has to be handled by staggering launches or by giving each run
    its own machine.
    """

    return _read_int(_CGROUP_V2_CURRENT) or 0


@dataclass(frozen=True)
class ConcurrencyPlan:
    """What the loop asked for, what it gets, and why."""

    candidates: int
    solver: int
    judge: int
    ceiling: int
    adjusted: bool
    reason: str

    @property
    def peak_calls(self) -> int:
        return self.candidates * self.solver


def max_concurrent_calls(
    *,
    per_call_gb: float = DEFAULT_PER_CALL_GB,
    headroom: float = DEFAULT_HEADROOM,
    memory_share: float = 1.0,
    limit_bytes: int | None = None,
    in_use_bytes: int | None = None,
) -> int:
    """How many headless calls fit in the memory left to this process.

    `memory_share` splits the budget between runs that will overlap. The
    startup snapshot cannot do this on its own: two runs launched together
    each see the other at nearly zero and both claim everything. Launching
    two experiments on one machine means passing 0.5 to both.
    """

    limit = limit_bytes if limit_bytes is not None else memory_limit_bytes()
    if limit is None:
        return 0  # unknown limit — callers treat 0 as "do not govern"

    in_use = in_use_bytes if in_use_bytes is not None else memory_in_use_bytes()
    usable = max(0, limit - in_use) * headroom * memory_share
    return max(1, int(usable / (per_call_gb * 1024**3)))


def govern(
    *,
    candidates: int,
    solver: int,
    judge: int,
    per_call_gb: float = DEFAULT_PER_CALL_GB,
    headroom: float = DEFAULT_HEADROOM,
    memory_share: float = 1.0,
    limit_bytes: int | None = None,
    in_use_bytes: int | None = None,
) -> ConcurrencyPlan:
    """Scale the requested concurrency down until it fits in memory.

    `solver` gives way before `candidates`. An iteration costs the slowest
    candidate rather than the sum only while every candidate is in flight, so
    trimming solver workers loses less wall clock than dropping a candidate
    out of the round entirely.
    """

    ceiling = max_concurrent_calls(
        per_call_gb=per_call_gb,
        headroom=headroom,
        memory_share=memory_share,
        limit_bytes=limit_bytes,
        in_use_bytes=in_use_bytes,
    )
    if ceiling <= 0:
        return ConcurrencyPlan(
            candidates, solver, judge, 0, False, "memory limit unknown; not governing"
        )

    if candidates * solver <= ceiling and judge <= ceiling:
        return ConcurrencyPlan(
            candidates, solver, judge, ceiling, False, "requested concurrency fits"
        )

    new_candidates = min(candidates, ceiling)
    # `candidates` is 0 when every proposal failed precheck.
    new_solver = max(1, min(solver, ceiling // max(1, new_candidates)))
    # A single candidate whose solver pool alone exceeds the ceiling is the
    # one case where candidates cannot absorb it.
    new_candidates = max(1, min(new_candidates, ceiling // new_solver))
    new_judge = max(1, min(judge, ceiling))

    return ConcurrencyPlan(
        new_candidates,
        new_solver,
        new_judge,
        ceiling,
        True,
        f"requested {candidates}x{solver}={candidates * solver} concurrent calls "
        f"but only {ceiling} fit in memory",
    )


def oom_pressure() -> tuple[int, int]:
    """Return (times the ceiling was hit, processes the OOM killer took).

    Both counters are monotonic for the life of the cgroup, so only the change
    between readings is meaningful.
    """

    events = Path("/sys/fs/cgroup/memory.events")
    hits = kills = 0
    try:
        for line in events.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            if key == "max":
                hits = int(value)
            elif key == "oom_kill":
                kills = int(value)
    except (OSError, ValueError):
        return (0, 0)
    return (hits, kills)


class MemorySampler:
    """Record the cgroup's peak usage while a block of work runs.

    A one-shot estimate cannot survive this loop: the prompt carries the whole
    rubric, and growing the filter is the entire point of the optimization: a
    run sized at iteration 0 can be killed a few iterations later, once its
    rules.py has grown tenfold.
    """

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.baseline = memory_in_use_bytes()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, memory_in_use_bytes())

    def __enter__(self) -> MemorySampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
        self.peak = max(self.peak, memory_in_use_bytes())


@dataclass
class AdaptiveGovernor:
    """Re-derive the concurrency ceiling from what the last iteration cost.

    The estimate ratchets up the moment an iteration proves it was too low and
    decays only slowly, because the two failure modes are not symmetric: too
    high wastes wall clock, too low ends the run.
    """

    per_call_gb: float = DEFAULT_PER_CALL_GB
    headroom: float = DEFAULT_HEADROOM
    memory_share: float = 1.0
    limit_bytes: int | None = None
    # Applied when the cgroup reports it hit its ceiling since the last look —
    # the run got close enough that the estimate is already wrong.
    pressure_multiplier: float = 1.5
    # Per-iteration decay toward a lower observed cost. Slow on purpose.
    decay: float = 0.9
    _last_pressure: tuple[int, int] = field(default_factory=oom_pressure)

    def plan(self, *, candidates: int, solver: int, judge: int) -> ConcurrencyPlan:
        return govern(
            candidates=candidates,
            solver=solver,
            judge=judge,
            per_call_gb=self.per_call_gb,
            headroom=self.headroom,
            memory_share=self.memory_share,
            limit_bytes=self.limit_bytes,
        )

    def observe(self, *, sampler: MemorySampler, concurrent_calls: int) -> float:
        """Fold one iteration's measured cost into the estimate.

        Returns the new per-call figure so callers can log the change.
        """

        hits, kills = oom_pressure()
        pressured = hits > self._last_pressure[0] or kills > self._last_pressure[1]
        self._last_pressure = (hits, kills)

        if concurrent_calls > 0:
            delta = max(0, sampler.peak - sampler.baseline)
            observed = delta / concurrent_calls / (1024**3)
            if observed > self.per_call_gb:
                self.per_call_gb = observed
            else:
                # Never fall to the observed value in one step: an iteration
                # that happened to run small prompts is not evidence the next
                # one will.
                self.per_call_gb = max(observed, self.per_call_gb * self.decay)

        if pressured:
            self.per_call_gb *= self.pressure_multiplier
            logger.warning(
                "concurrency governor: cgroup reported memory pressure — "
                "raising the per-call estimate to %.2f GB",
                self.per_call_gb,
            )

        return self.per_call_gb


def log_plan(plan: ConcurrencyPlan) -> None:
    """Say what was applied, always — a quiet cap reads as a slow machine."""

    if plan.ceiling == 0:
        logger.warning("concurrency governor: %s", plan.reason)
        return
    if plan.adjusted:
        logger.warning(
            "concurrency governor: %s — scaled to candidates=%d solver=%d judge=%d "
            "(%d concurrent calls)",
            plan.reason,
            plan.candidates,
            plan.solver,
            plan.judge,
            plan.peak_calls,
        )
    else:
        logger.info(
            "concurrency governor: candidates=%d solver=%d judge=%d "
            "(%d concurrent calls, ceiling %d)",
            plan.candidates,
            plan.solver,
            plan.judge,
            plan.peak_calls,
            plan.ceiling,
        )
