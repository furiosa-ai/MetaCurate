"""Paired comparison of two rankers on one fixed pair set.

Every candidate in an iteration is graded on the same judged pairs, so the
question "is the candidate better than the baseline" is a paired one: how many
pairs did the candidate get right that the baseline got wrong (`b`), and the
reverse (`c`). Pairs both rankers agree on carry no information about the
difference, however many there are. That is McNemar's test, and it is the
arithmetic the acceptance gate runs on.

Why this exists at all: taking the max of a few noisy concordances decides
most selections by a margin inside the re-score noise of a single candidate,
so a loop that does it mostly promotes measurement error — the winner's
curse. Requiring `b − c` to clear a
z threshold makes the noise the thing being tested instead of the thing being
ranked on.

Caveat the callers must remember: pairs share rows. A row that the candidate
scores differently from the baseline flips every pair it appears in, so the
discordant pairs are not independent draws and the plain `z` overstates the
evidence. `cluster_bootstrap_z` resamples rows rather than pairs and is the
check on that; it is reported, not gating, until it has been seen on real runs.

Stdlib only: `math.comb` and `math.erfc` cover the exact and normal branches,
and a bisection on `erfc` covers the one inverse we need.
"""

from __future__ import annotations

import logging
import math
import random
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

Pair = tuple[str, str]

# Below this many discordant pairs the normal approximation to the binomial
# is visibly off in the tail; the exact sum is cheap there anyway.
EXACT_BINOMIAL_BELOW = 25

# The discordant-pair counts `critical_b_table` reads off by default: dense
# where the exact branch binds, sparser where the normal one does.
CRITICAL_B_TABLE_NS: tuple[int, ...] = (5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200)


def pair_correctness(
    scores: Mapping[str, float],
    pairs: Iterable[Pair],
    *,
    ties: Iterable[Pair] | None = None,
) -> dict[Pair, float]:
    """Per-pair credit a score map earns on judged ``(hi, lo)`` pairs.

    Mirrors `ranking.concordance_detail`: 1.0 when the candidate orders the
    pair as the judge did, 0.5 when it ties the two rows itself, 0.0 when it
    reverses them or emitted no score for one of them (a ranker that declines
    to rank a row is a worse ranker, not an excused one). Pairs the judge tied
    are dropped entirely — no verdict exists to be right about, and half credit
    would be farmable by tying the same pair.

    Args:
        scores: Trajectory id to the candidate's continuous score.
        pairs: Judged ``(hi, lo)`` pairs, `hi` being the judge's preferred row.
        ties: Unordered pairs the judge tied; excluded from the result.

    Returns:
        Mapping from each retained ``(hi, lo)`` pair to its credit.
    """

    tie_keys = {frozenset(p) for p in ties} if ties else set()
    out: dict[Pair, float] = {}
    for hi, lo in pairs:
        if tie_keys and frozenset((hi, lo)) in tie_keys:
            continue
        if hi not in scores or lo not in scores:
            out[(hi, lo)] = 0.0
            continue
        delta = scores[hi] - scores[lo]
        out[(hi, lo)] = 1.0 if delta > 0 else 0.5 if delta == 0 else 0.0
    return out


@dataclass
class PairedStats:
    """McNemar-style summary of a candidate against a baseline on shared pairs.

    "Right" is credit exactly 1.0. A candidate's own tie (0.5) is not a correct
    ordering, so it counts as wrong here even though selection concordance
    still pays it half — the gate asks whether the candidate *orders* pairs the
    baseline misorders, and a tie orders nothing.
    """

    n_pairs: int
    both_right: int
    both_wrong: int
    # Candidate right, baseline wrong — the pairs the change fixed.
    b: int
    # Baseline right, candidate wrong — the pairs the change broke.
    c: int
    # (b - c) / n_pairs: the concordance gain net of own-tie effects.
    diff: float
    # (b - c) / sqrt(b + c); 0.0 when there are no discordant pairs.
    z: float
    # One-sided P(B >= b | b + c trials at 0.5): the chance a change with no
    # real effect fixes at least this many more pairs than it breaks.
    p_one_sided: float
    # "exact" below EXACT_BINOMIAL_BELOW discordant pairs, "normal" above,
    # "none" when b + c == 0 and there is nothing to test.
    method: str

    @property
    def n_discordant(self) -> int:
        return self.b + self.c


def paired_stats(cand: Mapping[Pair, float], base: Mapping[Pair, float]) -> PairedStats:
    """Discordant-pair counts and the one-sided test for candidate > baseline.

    Both maps must cover the same pairs: they come from `pair_correctness` on
    one fixed pair set, so a mismatch is a wiring bug, not a measurement.

    Args:
        cand: Candidate credit per pair, from `pair_correctness`.
        base: Baseline credit per pair, from `pair_correctness`.

    Returns:
        The paired statistics.

    Raises:
        ValueError: If the two maps do not cover exactly the same pairs.
    """

    if cand.keys() != base.keys():
        only_cand = len(cand.keys() - base.keys())
        only_base = len(base.keys() - cand.keys())
        raise ValueError(
            "paired_stats needs the same pair set on both sides; "
            f"{only_cand} pairs only in the candidate map, {only_base} only in the baseline map"
        )

    both_right = both_wrong = b = c = 0
    for key, cand_credit in cand.items():
        cand_right = cand_credit == 1.0
        base_right = base[key] == 1.0
        if cand_right and base_right:
            both_right += 1
        elif cand_right:
            b += 1
        elif base_right:
            c += 1
        else:
            both_wrong += 1

    n_pairs = len(cand)
    diff = (b - c) / n_pairs if n_pairs else 0.0
    z, p, method = stats_from_counts(b, c)
    return PairedStats(
        n_pairs=n_pairs,
        both_right=both_right,
        both_wrong=both_wrong,
        b=b,
        c=c,
        diff=diff,
        z=z,
        p_one_sided=p,
        method=method,
    )


def stats_from_counts(b: int, c: int) -> tuple[float, float, str]:
    """`z`, one-sided `p` and the method, from discordant counts alone.

    The arithmetic `paired_stats` runs once it has counted, split out so the
    same test can be run on counts pooled over two draws of the same
    comparison (the acceptance confirmation): `b` and `c`
    from the draw that won plus `b` and `c` from the promotion re-draw, one
    binomial over both. Below `EXACT_BINOMIAL_BELOW` discordant pairs the
    exact tail is used, above it the normal approximation, and with no
    discordant pair there is nothing to test (`p = 1`).

    Args:
        b: Pairs the candidate got right and the baseline wrong.
        c: Pairs the baseline got right and the candidate wrong.

    Returns:
        ``(z, p_one_sided, method)`` with method one of "exact", "normal",
        "none".

    Raises:
        ValueError: If either count is negative.
    """

    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b} c={c}")
    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0, "none"
    z = (b - c) / math.sqrt(n_discordant)
    if n_discordant < EXACT_BINOMIAL_BELOW:
        return z, _binomial_tail_at_half(b, n_discordant), "exact"
    return z, one_sided_p_from_z(z), "normal"


def critical_b(n_discordant: int, alpha: float) -> int | None:
    """Smallest `b` out of `n_discordant` discordant pairs whose one-sided `p`
    is at most `alpha` — the number of fixed pairs a candidate needs at that
    many discordant pairs, under exactly the test `stats_from_counts` runs.

    Same branch rule as the gate: the exact binomial below
    `EXACT_BINOMIAL_BELOW`, the normal approximation from there, so the
    answer is what the selector will actually demand and not a textbook
    figure. `p` falls as `b` rises at fixed `n`, so the first `b` that
    passes is the threshold and every larger `b` passes too.

    Args:
        n_discordant: ``b + c``, the number of pairs the two rankers disagree
            on.
        alpha: The one-sided tail probability the gate holds a candidate to
            (`one_sided_p_from_z` of its effective z threshold).

    Returns:
        The critical `b`, or None when even ``b = n_discordant`` (every
        discordant pair fixed, none broken) does not reach `alpha` — the
        gate cannot fire at that many discordant pairs, whatever the
        candidate does.

    Raises:
        ValueError: If `n_discordant` is negative.
    """

    if n_discordant < 0:
        raise ValueError(f"n_discordant must be non-negative, got {n_discordant}")
    for b in range(n_discordant + 1):
        _, p, _ = stats_from_counts(b, n_discordant - b)
        if p <= alpha:
            return b
    return None


def critical_b_table(
    alpha: float, ns: Iterable[int] = CRITICAL_B_TABLE_NS
) -> list[tuple[int, int | None]]:
    """`critical_b` at each `n` in `ns`, as ``(n, critical_b)`` rows.

    Args:
        alpha: The one-sided tail probability, as for `critical_b`.
        ns: Discordant-pair counts to tabulate; `CRITICAL_B_TABLE_NS` by
            default.

    Returns:
        One row per `n`, in the order given.
    """

    return [(n, critical_b(n, alpha)) for n in ns]


def render_critical_b_table(
    alpha: float, label: str, ns: Iterable[int] = CRITICAL_B_TABLE_NS
) -> str:
    """One compact text block: what `b` the gate needs at each `b + c`.

    Written once at run start for the family threshold and once for the
    strong-evidence threshold, so a reader of the log sees "b+c = 30 needs
    b >= 24" without re-deriving the binomial. A dash marks an `n` the
    threshold cannot be reached at.

    Args:
        alpha: The one-sided tail probability the row set is computed for.
        label: What the threshold is, e.g. ``"family z=2.43"``; printed in
            the header.
        ns: Discordant-pair counts to tabulate.

    Returns:
        A three-line block: header, ``b+c`` row, ``b>=`` row.
    """

    rows = critical_b_table(alpha, ns)
    width = max(3, *(len(str(n)) for n, _ in rows))
    header = f"critical b at alpha={alpha:.2e} ({label}): smallest b that passes, per b+c"
    ns_line = "  b+c: " + " ".join(f"{n:>{width}d}" for n, _ in rows)
    bs_line = "  b>=: " + " ".join(f"{('-' if b is None else str(b)):>{width}s}" for _, b in rows)
    return "\n".join((header, ns_line, bs_line))


def cluster_bootstrap_z(
    cand: Mapping[Pair, float],
    base: Mapping[Pair, float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> float:
    """Row-cluster bootstrap z for ``b − c``: mean over resamples / std.

    The plain `z` in `paired_stats` treats every discordant pair as an
    independent coin flip. It is not: pairs share rows, and one row the
    candidate re-scores flips every pair that row sits in, so the effective
    sample is closer to the number of rows involved than to `b + c`. Resampling
    trajectory ids with replacement and re-counting `b − c` on the pairs whose
    rows survived puts that dependence back. A pair is kept with weight
    ``min(count_hi, count_lo)`` — the number of times the resample contains
    both of its rows — which is the cheap stand-in for enumerating every
    (draw_i, draw_j) combination.

    Args:
        cand: Candidate credit per pair.
        base: Baseline credit per pair, same keys.
        n_resamples: Bootstrap resamples; 1,000 is stable to the first decimal.
        seed: Seed for the resampling RNG, so a rerun reproduces the number.

    Returns:
        mean(b − c) / std(b − c) across resamples; 0.0 if the std is 0 or
        there are no rows.

    Raises:
        ValueError: If the two maps do not cover exactly the same pairs.
    """

    if cand.keys() != base.keys():
        raise ValueError("cluster_bootstrap_z needs the same pair set on both sides")

    ids = sorted({row for key in cand for row in key})
    if not ids:
        return 0.0

    # Concordant pairs contribute 0 to every resample; only the discordant
    # ones need to be walked, which keeps 1,000 resamples cheap.
    discordant: list[tuple[str, str, int]] = []
    for (hi, lo), cand_credit in cand.items():
        cand_right = cand_credit == 1.0
        base_right = base[(hi, lo)] == 1.0
        if cand_right and not base_right:
            discordant.append((hi, lo, 1))
        elif base_right and not cand_right:
            discordant.append((hi, lo, -1))
    if not discordant:
        return 0.0

    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_resamples):
        counts = Counter(rng.choices(ids, k=len(ids)))
        stat = 0
        for hi, lo, sign in discordant:
            weight = min(counts.get(hi, 0), counts.get(lo, 0))
            if weight:
                stat += sign * weight
        stats.append(float(stat))

    spread = statistics.pstdev(stats)
    if spread == 0.0:
        return 0.0
    return statistics.fmean(stats) / spread


def z_threshold_for(n_candidates: int, base_z: float = 2.0) -> float:
    """One-sided z that keeps the family-wise error of `n_candidates` tests at
    the level a single test at `base_z` would have (Šidák correction).

    The selector compares every candidate to the baseline and promotes if any
    passes, so with three candidates a per-comparison 2.0 is really a looser
    family gate; this returns the per-comparison z that undoes that. With the
    default `base_z` it is about 2.43 for three candidates.

    Args:
        n_candidates: Number of candidate-vs-baseline comparisons.
        base_z: The one-sided z the family as a whole should be held to.

    Returns:
        The corrected per-comparison z threshold.

    Raises:
        ValueError: If `n_candidates` is not positive.
    """

    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    alpha = one_sided_p_from_z(base_z)
    alpha_per_test = 1.0 - (1.0 - alpha) ** (1.0 / n_candidates)
    return z_from_one_sided_p(alpha_per_test)


def one_sided_p_from_z(z: float) -> float:
    """Upper-tail standard normal probability P(Z >= z)."""

    return 0.5 * math.erfc(z / math.sqrt(2.0))


def z_from_one_sided_p(p: float) -> float:
    """Inverse of `one_sided_p_from_z`, by bisection on `erfc`.

    The stdlib has no inverse normal CDF; the tail we need is monotone and a
    bisection to 1e-12 is more than the thresholds warrant.
    """

    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if one_sided_p_from_z(mid) > p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def _binomial_tail_at_half(successes: int, trials: int) -> float:
    """P(X >= successes) for X ~ Binomial(trials, 0.5)."""

    if successes <= 0:
        return 1.0
    total = sum(math.comb(trials, k) for k in range(successes, trials + 1))
    return total / 2**trials
