"""Pure math for the ranking optimization loop.

Candidates assign every trajectory a continuous score
(higher = better SFT training data). Candidates are compared without gold
labels: pairs of trajectories that all candidates order the same way become
free internal pseudo-truth ("consensus pairs"), while pairs the candidates
disagree on are sent to an LLM judge elsewhere. This module contains only
the deterministic math: pair classification, stratified disagreement
sampling, pairwise concordance, Kendall-style distance, and score
distribution reporting.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_HIST_BINS = 10
_BAR_WIDTH = 20
_LEVEL_DECIMALS = 6


@dataclass
class PairTruth:
    """Pairwise supervision extracted from a set of candidate score maps.

    Attributes:
        consensus_pairs: Ordered pairs ``(hi, lo)`` where every candidate
            ranks ``hi`` strictly above ``lo``.
        disagreement_pairs: Unordered pairs (deterministic order) on which at
            least two candidates disagree in strict order, sampled down to
            the disagreement budget, stratified across score deciles.
        sampled_universe: Number of trajectory pairs that were examined.
    """

    consensus_pairs: list[tuple[str, str]] = field(default_factory=list)
    disagreement_pairs: list[tuple[str, str]] = field(default_factory=list)
    audit_pairs: list[tuple[str, str]] = field(default_factory=list)
    sampled_universe: int = 0


def _sample_pair_indices(n: int, max_pairs: int, rng: random.Random) -> list[tuple[int, int]]:
    """Return index pairs ``(i, j)`` with ``i < j`` over ``range(n)``.

    All ``n * (n - 1) / 2`` pairs when that fits within ``max_pairs``,
    otherwise a deterministic uniform sample of exactly ``max_pairs``
    distinct pairs drawn from ``rng``.
    """
    total = n * (n - 1) // 2
    if total <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    chosen: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(chosen) < max_pairs:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        pair = (i, j) if i < j else (j, i)
        if pair in seen:
            continue
        seen.add(pair)
        chosen.append(pair)
    return chosen


def _sign(delta: float) -> int:
    """Return -1, 0, or 1 for the sign of ``delta``."""
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def _pair_mean(
    scores_by_candidate: dict[str, dict[str, float]],
    candidates: list[str],
    a: str,
    b: str,
) -> float:
    """Mean of the two trajectories' scores across candidates.

    The stratification key, so an audit sample spans the score distribution
    the same way the disagreement sample does.
    """
    return sum(
        (scores_by_candidate[c][a] + scores_by_candidate[c][b]) / 2 for c in candidates
    ) / len(candidates)


def _stratified_sample(
    pairs_with_mean: list[tuple[tuple[str, str], float]],
    budget: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Sample up to ``budget`` pairs stratified across deciles of the mean score.

    Pairs are ranked by their mean score and split into ten rank-based
    buckets (deciles). Selection proceeds round-robin across non-empty
    buckets, taking one pair per bucket per round, so the sample spans the
    whole score distribution instead of clustering at one cut boundary.
    """
    if budget <= 0 or not pairs_with_mean:
        return []

    ranked = sorted(pairs_with_mean, key=lambda item: (item[1], item[0]))
    n = len(ranked)
    buckets: list[list[tuple[str, str]]] = [[] for _ in range(_HIST_BINS)]
    for rank, (pair, _mean) in enumerate(ranked):
        buckets[rank * _HIST_BINS // n].append(pair)
    for bucket in buckets:
        rng.shuffle(bucket)

    sampled: list[tuple[str, str]] = []
    while len(sampled) < budget and any(buckets):
        for bucket in buckets:
            if len(sampled) >= budget:
                break
            if bucket:
                sampled.append(bucket.pop())
    return sampled


def truth_pairs(
    scores_by_candidate: dict[str, dict[str, float]],
    *,
    max_pairs: int = 4000,
    disagreement_budget: int = 48,
    seed: int = 17,
    consensus_share: float = 0.0,
) -> PairTruth:
    """Classify trajectory pairs into consensus and disagreement supervision.

    Args:
        scores_by_candidate: ``{candidate_id: {trajectory_id: score}}``. Only
            trajectory ids present in every candidate's map are used.
        max_pairs: Cap on the number of unordered pairs examined; beyond it a
            deterministic random sample of this size is drawn.
        disagreement_budget: Maximum number of pairs handed to the judge.
        seed: Seed for all sampling decisions.
        consensus_share: Share of the judge budget spent auditing pairs the
            candidates agree on; the rest goes to pairs they order
            differently, stratified across deciles of the mean score.

    Returns:
        A :class:`PairTruth` with all consensus pairs found, the sampled
        disagreement pairs, and the number of pairs examined.
    """
    candidates = sorted(scores_by_candidate)
    if not candidates:
        return PairTruth()

    common_ids = set(scores_by_candidate[candidates[0]])
    for candidate in candidates[1:]:
        common_ids &= set(scores_by_candidate[candidate])
    ids = sorted(common_ids)
    if len(ids) < 2:
        return PairTruth()

    rng = random.Random(seed)
    index_pairs = _sample_pair_indices(len(ids), max_pairs, rng)

    consensus: list[tuple[str, str]] = []
    audit_pairs: list[tuple[str, str]] = []
    disagreements: list[tuple[tuple[str, str], float]] = []
    for i, j in index_pairs:
        a, b = ids[i], ids[j]
        # A tie is an ABSTENTION, not a veto. Under "a tie in any candidate
        # skips the pair", a single degenerate candidate — one constant score
        # for every row — would tie every pair and erase the entire truth
        # channel: selection goes blind (everyone's concordance 0), and the
        # degenerate filter itself wins on the secondary terms. Strict votes
        # decide; ties sit out. Consensus needs at least two strict votes so a single
        # opinionated candidate cannot dictate truth alone.
        votes = [
            s
            for c in candidates
            if (s := _sign(scores_by_candidate[c][a] - scores_by_candidate[c][b])) != 0
        ]
        signs = set(votes)
        strict_votes = len(votes)
        if not signs:
            continue  # everyone abstained — no order information at all
        if len(signs) == 1:
            if strict_votes < 2:
                continue  # one lone strict vote is an opinion, not consensus
            hi, lo = (a, b) if signs == {1} else (b, a)
            consensus.append((hi, lo))
        else:
            pair_mean = sum(
                (scores_by_candidate[c][a] + scores_by_candidate[c][b]) / 2 for c in candidates
            ) / len(candidates)
            disagreements.append(((a, b), pair_mean))

    # Two strata, one budget, split by design rather than by circumstance.
    #
    # The disagreement stratum is where selection signal lives: every pair
    # in it separates at least two candidates. The consensus stratum
    # separates nobody — all candidates score 1.0 on it — but it is the
    # only place the truth channel itself can be checked, and it is 98.8%
    # of the pool. Sampling it is not a fallback for an empty disagreement
    # set; it is the audit that makes "every candidate agrees" falsifiable.
    #
    # Without it the objective has a blind spot with a perfect score: if
    # every candidate ranked the pool exactly backwards and agreed, there
    # would be no disagreement to judge, consensus would be the whole truth
    # set, and concordance would read 1.0. A fixed audit share means the
    # judge sees consensus pairs every iteration, so that state is
    # detectable whether or not disagreement happens to be scarce.
    audit_budget = int(round(disagreement_budget * consensus_share))
    disagree_budget = disagreement_budget - audit_budget
    sampled_disagreements = _stratified_sample(disagreements, disagree_budget, rng)
    # A stratum with nothing left to sample hands its remainder to the
    # other one — the ordinary behaviour of a stratified sampler, not a
    # special case: late in a run the non-consensus pairs can dwindle to a
    # few dozen out of 4,000, or to none at all.
    if audit_budget:
        # Only an enabled audit absorbs the leftover. With the audit off the
        # disagreement sample is left exactly as `_stratified_sample` drew it.
        audit_budget += disagree_budget - len(sampled_disagreements)
    audit = _stratified_sample(
        [(pair, _pair_mean(scores_by_candidate, candidates, *pair)) for pair in consensus],
        audit_budget,
        rng,
    )
    judged = {frozenset(pair) for pair in audit}
    # The judge outranks a candidate vote, so an audited pair leaves
    # consensus rather than sitting in the truth set twice — once in the
    # candidates' order and once in the judge's.
    consensus = [pair for pair in consensus if frozenset(pair) not in judged]
    audit_pairs = audit
    logger.debug(
        "truth_pairs: %d ids, %d pairs examined, %d consensus, %d/%d judged",
        len(ids),
        len(index_pairs),
        len(consensus),
        len(sampled_disagreements),
        len(disagreements),
    )
    return PairTruth(
        consensus_pairs=consensus,
        disagreement_pairs=sampled_disagreements,
        audit_pairs=audit_pairs,
        sampled_universe=len(index_pairs),
    )


@dataclass(frozen=True)
class ConcordanceResult:
    """A concordance measurement and what it was measured over.

    `n_offered` vs `n_scored` is the part worth reading: the difference is
    pairs this candidate could not be asked about, because it emitted no
    score for one of the two rows. The ranking contract is a continuous
    score for EVERY row (`policy.py`), so that difference is a contract
    violation and `missing_ids` names how many rows it happened on.
    """

    concordance: float
    n_offered: int
    n_scored: int
    missing_ids: int
    # Pairs the judge adjudicated as TIE and that therefore left the exam
    # entirely. Reported rather than folded into `n_offered` so a run can see
    # how much of its paid-for judging produced no supervision.
    n_tied: int = 0

    @property
    def n_unscored(self) -> int:
        return self.n_offered - self.n_scored


def concordance_detail(
    scores: dict[str, float],
    pairs: list[tuple[str, str]],
    *,
    unscored: str = "skip",
    ties: Iterable[tuple[str, str]] | None = None,
) -> ConcordanceResult:
    """Concordance plus the coverage behind it.

    `unscored` decides what an unscoreable pair costs:

    * ``"skip"`` — drop it and divide by what remained. Used by the
      diagnostics that are not selection.
    * ``"loss"`` — count it against this candidate. Correct for selection: a
      ranker that declines to rank some rows is a worse ranker, and the
      alternative is worse than forgiving it. `truth_pairs` intersects the
      candidates' score maps, so under ``"skip"`` one candidate's missing row
      removes that row's pairs from *every* candidate's exam — the incumbent
      is charged for a rival's bug (a `universe=19701` against 19900 means
      C(199,2): precisely one row went unscored for some candidate).

    `ties` are unordered pairs the JUDGE tied — including the pairs whose two
    presentation orders named different winners (`pair_judge`). They leave
    both the numerator and the denominator. Half credit would be the wrong
    reading twice over: it scores a candidate on a pair no verdict exists
    for, and it does so in a way a candidate can farm by tying the same pair
    itself. A candidate's OWN tie is a different thing and still earns 0.5 —
    there the truth is known and the candidate declined to express it.
    """

    if unscored not in ("skip", "loss"):
        raise ValueError(f"unscored must be 'skip' or 'loss', got {unscored!r}")

    tie_keys = {frozenset(p) for p in ties} if ties else set()
    eligible = [p for p in pairs if frozenset(p) not in tie_keys] if tie_keys else list(pairs)

    credit = 0.0
    scored = 0
    missing: set[str] = set()
    for hi, lo in eligible:
        if hi not in scores or lo not in scores:
            missing.update(i for i in (hi, lo) if i not in scores)
            continue
        scored += 1
        sign = _sign(scores[hi] - scores[lo])
        if sign > 0:
            credit += 1.0
        elif sign == 0:
            credit += 0.5

    denominator = len(eligible) if unscored == "loss" else scored
    return ConcordanceResult(
        concordance=credit / denominator if denominator else 0.0,
        n_offered=len(eligible),
        n_scored=scored,
        missing_ids=len(missing),
        n_tied=len(pairs) - len(eligible),
    )


def concordance(
    scores: dict[str, float],
    pairs: list[tuple[str, str]],
    *,
    unscored: str = "skip",
    ties: Iterable[tuple[str, str]] | None = None,
) -> float:
    """Fraction of ordered ``(hi, lo)`` pairs that ``scores`` orders correctly.

    A tie in ``scores`` counts 0.5. See `concordance_detail` for what
    `unscored` and `ties` do, and why selection should not use the default.
    """

    return concordance_detail(scores, pairs, unscored=unscored, ties=ties).concordance


def kendall_distance(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    *,
    max_pairs: int = 4000,
    seed: int = 17,
) -> float:
    """Pairwise disagreement rate between two score maps over common ids.

    Each sampled common pair contributes 1.0 when the maps order it in
    strictly opposite ways, 0.5 when exactly one map ties it, and 0.0 when
    both agree strictly or both tie. Returns 0.0 when fewer than two common
    ids exist.
    """
    ids = sorted(set(scores_a) & set(scores_b))
    if len(ids) < 2:
        return 0.0

    rng = random.Random(seed)
    index_pairs = _sample_pair_indices(len(ids), max_pairs, rng)

    disagreement = 0.0
    for i, j in index_pairs:
        a, b = ids[i], ids[j]
        sign_a = _sign(scores_a[a] - scores_a[b])
        sign_b = _sign(scores_b[a] - scores_b[b])
        if sign_a == 0 and sign_b == 0:
            continue
        if sign_a == 0 or sign_b == 0:
            disagreement += 0.5
        elif sign_a != sign_b:
            disagreement += 1.0
    return disagreement / len(index_pairs)


def distinct_levels(scores: dict[str, float]) -> int:
    """Number of distinct score values after rounding to 6 decimals."""
    return len({round(v, _LEVEL_DECIMALS) for v in scores.values()})


# The guard is a floor, not a target: below three levels a scorer cannot
# express more than "better, same, worse" and is not ranking at all.
_MIN_LEVELS_FLOOR = 3
# Scaled so that a 200-row pool lands on 5, the size the constant was chosen
# for.
_MIN_LEVELS_SHARE = 0.025

# What a degenerate candidate is charged ON TOP of having its concordance
# forced to 0. Both halves are needed and neither alone is enough: zeroing
# the agreement term leaves a candidate that cannot rank competing on the
# secondary terms, where a filter with one distinct score could win. It lives
# here rather than as a literal at the point of
# subtraction because three places quote it — the optimizer applies it, the
# ranking telemetry describes it, and the proposer's system prompt states it
# as part of the objective — and a prompt that misquotes the objective is
# the specific bug `build_proposer_system` exists to prevent.
DEGENERACY_PENALTY = 1.0


def default_min_distinct_levels(n_train: int) -> int:
    """The degeneracy floor for a pool of ``n_train`` rows.

    This is a DEGENERACY guard — it catches a scorer that emits one constant,
    or three, and therefore orders nothing (a filter that cannot rank must
    not win selection on the secondary terms). It is not a
    resolution target, and setting it like one is how it goes wrong in the
    other direction: a floor near the pool size fails candidates that rank
    the pool perfectly well in coarse steps, and on a 50-row pool there is
    not much room between "degenerate" and "fine".

    So: 5 at 200 rows, and 3 at 50, which is the smallest number that can
    express an ordering at all. Runs at different pool sizes stay comparable because the guard keeps
    catching the same failure rather than a different fraction of candidates.
    """

    return max(_MIN_LEVELS_FLOOR, round(_MIN_LEVELS_SHARE * max(0, n_train)))


def _histogram(values: list[float]) -> list[int]:
    """Histogram over 10 equal-width bins between min and max of ``values``.

    All-equal inputs put everything in the first bin.
    """
    counts = [0] * _HIST_BINS
    if not values:
        return counts
    lo, hi = min(values), max(values)
    if hi == lo:
        counts[0] = len(values)
        return counts
    width = (hi - lo) / _HIST_BINS
    for v in values:
        counts[min(int((v - lo) / width), _HIST_BINS - 1)] += 1
    return counts


def _entropy_bits(counts: list[int]) -> float:
    """Shannon entropy of a histogram, in bits."""
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def distribution_report(
    scores: dict[str, float],
    resolved: dict[str, bool] | None = None,
) -> dict:
    """Summarize a score distribution as a JSON-safe dict.

    Args:
        scores: ``{trajectory_id: score}``.
        resolved: Optional ``{trajectory_id: resolved?}`` map; when given the
            report also carries the mean score of resolved vs unresolved
            trajectories and their separation.

    Returns:
        Dict with ``n``, ``mean``, ``std``, ``min``, ``max``,
        ``distinct_levels``, ``decile_counts``, ``entropy``, and (when
        ``resolved`` is given) ``resolved_mean``, ``unresolved_mean``, and
        ``separation`` (``resolved_mean - unresolved_mean``); the latter
        three are ``None`` when a group is empty.
    """
    values = list(scores.values())
    n = len(values)
    counts = _histogram(values)
    if n:
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
        lo, hi = min(values), max(values)
    else:
        mean = std = lo = hi = 0.0
    report: dict = {
        "n": n,
        "mean": mean,
        "std": std,
        "min": lo,
        "max": hi,
        "distinct_levels": distinct_levels(scores),
        "decile_counts": counts,
        "entropy": _entropy_bits(counts),
    }
    if resolved is not None:
        resolved_scores = [v for k, v in scores.items() if resolved.get(k) is True]
        unresolved_scores = [v for k, v in scores.items() if resolved.get(k) is False]
        resolved_mean = sum(resolved_scores) / len(resolved_scores) if resolved_scores else None
        unresolved_mean = (
            sum(unresolved_scores) / len(unresolved_scores) if unresolved_scores else None
        )
        report["resolved_mean"] = resolved_mean
        report["unresolved_mean"] = unresolved_mean
        report["separation"] = (
            resolved_mean - unresolved_mean
            if resolved_mean is not None and unresolved_mean is not None
            else None
        )
    return report


def render_distribution(report: dict) -> str:
    """Render a :func:`distribution_report` dict as a compact prompt block.

    One summary line, one ASCII bar per decile (scaled to a 20-char maximum),
    and a resolved/unresolved separation line when the report carries one.
    Deterministic for a given report.
    """
    lines = [
        (
            f"n={report['n']} mean={report['mean']:.4f} std={report['std']:.4f} "
            f"min={report['min']:.4f} max={report['max']:.4f} "
            f"levels={report['distinct_levels']} entropy={report['entropy']:.3f}"
        )
    ]
    counts = report["decile_counts"]
    peak = max(counts) if counts else 0
    for idx, count in enumerate(counts):
        bar = "#" * round(count / peak * _BAR_WIDTH) if peak else ""
        lines.append(f"d{idx} |{bar:<{_BAR_WIDTH}}| {count}")
    if report.get("separation") is not None:
        lines.append(
            f"resolved_mean={report['resolved_mean']:.4f} "
            f"unresolved_mean={report['unresolved_mean']:.4f} "
            f"separation={report['separation']:+.4f}"
        )
    return "\n".join(lines)


def missed_pairs(
    scores: dict[str, float], judged_pairs: list[tuple[str, str]]
) -> list[tuple[str, str, float]]:
    """Judged pairs this score map orders WRONGLY, most confidently wrong first.

    Input pairs are judge-ordered (winner, loser). A miss is a pair the map
    inverts (loser strictly above winner); ties are not misses (they are
    half-credit in concordance, and carry no design signal about direction).
    Returns (winner, loser, inversion_gap) where inversion_gap =
    scores[loser] - scores[winner] > 0 — the bigger the gap, the more
    confidently the design got the pair backwards, which is what makes it
    instructive evidence for the proposer.
    """

    out: list[tuple[str, str, float]] = []
    for winner, loser in judged_pairs:
        if winner not in scores or loser not in scores:
            continue
        gap = scores[loser] - scores[winner]
        if gap > 0:
            out.append((winner, loser, gap))
    out.sort(key=lambda t: -t[2])
    return out
