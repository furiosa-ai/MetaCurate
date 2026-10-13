"""Selector — chooses the next filter from a set of scored candidates.

`Selector.choose_paired` is a McNemar gate on a fixed, hidden selection pair
set:

    * The current filter is always in the candidate pool as the baseline, and
      it stays unless a candidate beats it pair-for-pair.
    * A candidate that regresses on stored counterexamples is disqualified
      unless the baseline regresses too.
    * There is no `loss` tie-break.

Picking among several candidates that all clear the gate with a bare
`argmax` on selection concordance would read a margin far below the noise
of a concordance difference as a win — the winner's curse the gate exists to
remove. So the leader's lead is tested too, and when it does not survive the
test a tie-break that is not the noisy metric decides — or, if none can, the
baseline stands.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from meta_filter.core.schemas import CandidateEvaluation
from meta_filter.gate.paired_test import (
    PairedStats,
    cluster_bootstrap_z,
    critical_b,
    one_sided_p_from_z,
    paired_stats,
    z_threshold_for,
)

logger = logging.getLogger(__name__)

# How much higher, in z, a candidate over a soft cap has to reach. Additive on
# purpose: doubling the threshold (2.43 → 4.85, α 6e-7) is out of reach of the
# exact binomial with fewer than 25 discordant pairs; one z-unit takes 2.43 to
# 3.43, α 0.0076 → 0.0003.
STRONG_EXTRA_Z = 1.0

# The z the leader must clear against each *other* eligible
# candidate before its lead is allowed to decide the iteration.
#
# Deliberately above the baseline gate's 2.0, and the asymmetry of the two
# errors is the whole argument. A missed separation costs a tie-break — the
# smaller filter, else the baseline — which is a cheap, defensible outcome. A
# false separation promotes the winner's curse this test exists to remove, and
# then spends the rest of the run proposing from it.
#
# 2.5 is a judgement, not a measurement: nobody has measured this null. See
# `_separation` for why it is not the baseline gate's null and for what would
# actually calibrate it.
AMONG_CANDIDATES_Z = 2.5

Pair = tuple[str, str]


@dataclass
class SelectionResult:
    winner: str
    reason: str
    ranked: list[str]
    # Candidates that sit on the (loss, complexity) or (loss, cost) Pareto
    # frontiers — the "PARETO_SAFE" set from Meta-Harness §4. Independent of
    # the scalar-loss winner, so the researcher can inspect both.
    pareto_ids_complexity: list[str] = field(default_factory=list)
    pareto_ids_cost: list[str] = field(default_factory=list)
    winner_pareto_safe: bool = False
    # `score` is the winner's train-side `metrics.score`, `baseline_score`
    # the baseline's; reported, not selected on.
    score: float = 0.0
    baseline_score: float = 0.0
    # Which path produced this result; `choose_paired` sets "mcnemar".
    gate: str = "sort"
    # Paired-gate telemetry. Mean per-pair credit on the selection set for the
    # winner and for the baseline; b/c/z/p/z_cluster are the winner's paired
    # statistics against the baseline. When the baseline is retained they are
    # the strongest challenger's (highest z), so `summary.csv` shows what the
    # gate turned away rather than a row of zeros — the baseline against
    # itself is trivially 0/0 and says nothing.
    sel_concordance: float = 0.0
    baseline_sel_concordance: float = 0.0
    b: int = 0
    c: int = 0
    z: float = 0.0
    p: float = 1.0
    z_cluster: float = 0.0
    # The z the winner (or strongest challenger) had to clear, after any
    # strong-evidence increment (`STRONG_EXTRA_Z`).
    threshold: float = 0.0
    # Every non-baseline candidate's statistics, eligibility and effective
    # threshold, keyed by candidate id. Written to `selection.json`.
    per_candidate: dict[str, dict] = field(default_factory=dict)
    # The winner was accepted on the simplification rule (`b >= c`,
    # no significance test) rather than on the paired test. The optimizer
    # records it as `gate_decision = "accepted_simplification"` and holds the
    # re-draw confirmation to `b2 >= c2` instead of the pooled test.
    simplification_winner: bool = False
    # How the choice *among* the eligible candidates was settled.
    # "" when no such choice arose (fewer than two eligible), "separated" when
    # the leader beat every rival on the paired test, "complexity"/"change"/
    # "order" when it did not and that tie-break decided, "hold" when nothing
    # could separate them and the baseline was kept instead, "below_baseline"
    # when the among-candidates stage did pick someone and the baseline guard
    # then refused them.
    among_tie_break: str = ""


@dataclass
class Selector:
    regression_max_violations: int = 0

    def choose_paired(
        self,
        evaluations: list[CandidateEvaluation],
        correctness_by_cand: Mapping[str, Mapping[Pair, float]],
        *,
        z_threshold: float | None = None,
        min_discordant: int = 5,
        strong_evidence: set[str] | frozenset[str] = frozenset(),
        strong_extra_z: float = STRONG_EXTRA_Z,
        bootstrap_seed: int = 0,
        simplification: set[str] | frozenset[str] = frozenset(),
        complexity_by_cand: Mapping[str, float] | None = None,
        among_z_threshold: float = AMONG_CANDIDATES_Z,
        change_size_by_cand: Mapping[str, float] | None = None,
    ) -> SelectionResult:
        """Promote a candidate only if it beats the baseline pair-for-pair.

        Taking the highest score of three or four noisy measurements mostly
        picks noise — the winner's curse. Here each candidate is compared to
        the baseline on
        the same judged pairs: `b` pairs it fixed, `c` pairs it broke, and it
        is eligible only if `b + c >= min_discordant` and its one-sided
        `p` is at most the tail probability of the z threshold (raised by
        `strong_extra_z` for ids in `strong_evidence`, the candidates that
        grew past a soft cap — additive, not multiplied: doubling 2.43 gives
        4.85, an α of 6e-7 the exact binomial cannot reach below 25
        discordant pairs).
        `z = (b - c) / sqrt(b + c)` is still reported, but the gate is on `p`
        so that the exact-binomial branch below 25 discordant pairs actually
        binds — b=6, c=0 has z=2.45 and clears 2.43, yet its exact p is
        0.0156 against the 0.0076 that z stands for.

        Among the eligible, the leader's margin is itself a measurement and is
        tested rather than trusted: the same paired machinery runs between the
        leader and every other eligible candidate at `among_z_threshold`. A
        leader that separates from all of them is promoted on concordance. A
        leader that does not is
        *tied* with the rivals it could not separate from, and the tie is
        settled off the noisy metric — lowest complexity, then smallest
        change, then proposal order — or, when none of those can separate the
        group, by keeping the baseline. Holding is the safe action here;
        promoting a coin flip is the failure mode being fixed. There is still
        deliberately no `loss` tie-break, and complexity enters only to break
        a tie the test already declared: size breaking *score* ties would
        leave nothing resisting growth.

        The row-cluster bootstrap `z_cluster` is computed and reported for
        every candidate but does not gate — pairs share rows so the plain `z`
        is optimistic, and the bootstrap has not yet been seen on real runs.

        `simplification` candidates (smaller than the baseline in every
        complexity component and strictly smaller in total) are held to a
        different rule: eligible iff `b >= c`, with no significance test and
        no discordant-pair floor, and an exact tie with the baseline does not
        disqualify them — `b = c = 0` is exactly what dropping a rule that
        never fired or an axis that orders nothing looks like, and refusing
        it would mean nothing ever shrinks. Nothing else in the loop rewards
        a smaller filter, since complexity is out of the acceptance
        decision. A regular candidate that passes the paired test always
        outranks a simplification; a simplification wins only when no regular
        candidate does.

        Args:
            evaluations: Candidates plus exactly one baseline.
            correctness_by_cand: Candidate id to its per-pair credit on the
                selection set (`paired_test.pair_correctness`), including the
                baseline's. Every valid candidate and the baseline must be
                present.
            z_threshold: Per-comparison z gate. ``None`` derives it from the
                number of valid non-baseline candidates with
                `paired_test.z_threshold_for`.
            min_discordant: Fewer discordant pairs than this and there is
                nothing to test; the candidate is ineligible.
            strong_evidence: Candidate ids whose threshold is raised by
                `strong_extra_z`.
            strong_extra_z: How much higher, in z, the `strong_evidence`
                threshold sits; 1.0 takes the three-candidate 2.43 (α
                0.0076) to 3.43 (α 0.0003).
            bootstrap_seed: Seed for `cluster_bootstrap_z`.
            simplification: Candidate ids accepted on `b >= c` alone (see
                above).
            complexity_by_cand: Candidate id to its `complexity_score`, the
                tie-break between two simplifications that keep equally many
                pairs, and the first tie-break between eligible
                candidates the paired test could not separate. Missing ids
                sort last, which keeps an unmeasured candidate from winning a
                tie on a zero.
            among_z_threshold: The z the leader must clear against each other
                eligible candidate. NOT the baseline gate's threshold and not
                derived from it — see `AMONG_CANDIDATES_Z` and `_separation`
                for why this null is a different, uncalibrated one.
            change_size_by_cand: Candidate id to how much it changed the
                baseline — edited criteria, diff lines, whatever the caller
                measures, as long as smaller means less changed. Second
                tie-break in the chain, after complexity. Missing ids sort
                last, same rule. The selector cannot derive this itself: it
                sees scored filters, not diffs.

        Returns:
            A `SelectionResult` with ``gate="mcnemar"`` and the paired
            telemetry populated.

        Raises:
            ValueError: No candidates, not exactly one baseline, or a
                correctness map missing for the baseline or a valid candidate.
        """

        if not evaluations:
            raise ValueError("no candidates to select from")
        baselines = [e for e in evaluations if e.is_baseline]
        if len(baselines) != 1:
            raise ValueError(f"expected exactly one baseline candidate, got {len(baselines)}")
        baseline = baselines[0]
        if baseline.candidate_id not in correctness_by_cand:
            raise ValueError(f"no correctness map for baseline {baseline.candidate_id!r}")
        base_corr = correctness_by_cand[baseline.candidate_id]

        valid = [e for e in evaluations if e.invalid_reason is None]
        challengers = [e for e in valid if not e.is_baseline]
        missing = [e.candidate_id for e in challengers if e.candidate_id not in correctness_by_cand]
        if missing:
            raise ValueError(f"no correctness map for candidates {missing}")

        # Regression filter: candidates that break stored
        # counterexamples are out unless the baseline breaks them too, in
        # which case nobody is clean and the whole pool competes.
        clean_ids = {e.candidate_id for e in valid if self._regression_clean(e)}
        regression_fallback = baseline.candidate_id not in clean_ids and not (
            clean_ids & {e.candidate_id for e in challengers}
        )

        threshold = (
            z_threshold if z_threshold is not None else z_threshold_for(max(len(challengers), 1))
        )
        baseline_sel = _mean_credit(base_corr)

        per_candidate: dict[str, dict] = {}
        eligible: list[tuple[str, float]] = []
        # (id, sel_concordance) of every simplification candidate that lost no pair.
        simplification_eligible: list[tuple[str, float]] = []
        for e in challengers:
            cid = e.candidate_id
            corr = correctness_by_cand[cid]
            stats = paired_stats(corr, base_corr)
            z_cluster = cluster_bootstrap_z(corr, base_corr, seed=bootstrap_seed)
            strong = cid in strong_evidence
            effective = threshold + strong_extra_z if strong else threshold
            regression_ok = cid in clean_ids or regression_fallback
            sel = _mean_credit(corr)
            is_simplification = cid in simplification

            # The gate is the one-sided p-value against the threshold's tail
            # probability, not `z` against the threshold. Below
            # `EXACT_BINOMIAL_BELOW` discordant pairs `p` is the exact
            # binomial, and there the normal `z` overstates the evidence:
            # b=6, c=0 has z=2.45, which clears 2.43, but the exact chance of
            # 6-0 under no effect is 1/64 = 0.0156, twice the 0.0076 that
            # z=2.43 stands for. Gating on `p` makes the exact branch bind;
            # in the normal branch it is the same test written the other way.
            p_threshold = one_sided_p_from_z(effective)
            blockers: list[str] = []
            if not regression_ok:
                blockers.append("regressed on stored counterexamples")
            if is_simplification:
                # The simplification rule: no pair lost. Nothing else gates it.
                if stats.b < stats.c:
                    blockers.append(f"simplification lost pairs: b={stats.b} < c={stats.c}")
            else:
                if stats.n_discordant < min_discordant:
                    blockers.append(f"b+c={stats.n_discordant} < min_discordant={min_discordant}")
                if stats.p_one_sided > p_threshold:
                    blockers.append(
                        f"p={stats.p_one_sided:.4f} > p_threshold={p_threshold:.4f} "
                        f"(z={stats.z:.2f} vs threshold={effective:.2f}, {stats.method})"
                    )
            is_eligible = not blockers
            if is_eligible and is_simplification:
                simplification_eligible.append((cid, sel))
            elif is_eligible:
                eligible.append((cid, sel))

            per_candidate[cid] = {
                **_stats_dict(stats),
                "z_cluster": z_cluster,
                "sel_concordance": sel,
                "eligible": is_eligible,
                "threshold": effective,
                "p_threshold": p_threshold,
                # The b this candidate needed at its own b + c to clear
                # `p_threshold`; None when no b could have.
                "min_b_required": critical_b(stats.n_discordant, p_threshold),
                "strong_evidence": strong,
                "regression_ok": regression_ok,
                "simplification": is_simplification,
                "blockers": blockers,
            }
            logger.debug(
                "paired gate %s: b=%d c=%d z=%.2f z_cluster=%.2f p=%.4f (%s) "
                "sel=%.4f vs baseline %.4f threshold=%.2f eligible=%s",
                cid,
                stats.b,
                stats.c,
                stats.z,
                z_cluster,
                stats.p_one_sided,
                stats.method,
                sel,
                baseline_sel,
                effective,
                is_eligible,
            )

        # The leader is the highest selection concordance among the
        # eligible — but its lead over the others is tested on the same
        # pairs, and only a tested lead decides. An exact tie is a
        # measure-zero special case of what `_separation` catches.
        winner = baseline
        simplification_won = False
        among_tie_break = ""
        among_note = ""
        order = {e.candidate_id: i for i, e in enumerate(challengers)}
        if eligible:
            sel_by_id = dict(eligible)
            top = max(sel for _, sel in eligible)
            # An exact tie for the top spot goes to proposal order, so which
            # candidate is called the leader is a function of the data rather
            # than of dict iteration. The one that loses the coin flip is in
            # the comparison set below either way, so nothing rests on it.
            leader_id = min((cid for cid, sel in eligible if sel == top), key=order.__getitem__)
            rivals = [cid for cid, _ in eligible if cid != leader_id]

            comparisons: dict[str, dict] = {}
            for rid in rivals:
                record = _separation(
                    correctness_by_cand[leader_id],
                    correctness_by_cand[rid],
                    threshold=among_z_threshold,
                    min_discordant=min_discordant,
                    seed=bootstrap_seed,
                )
                comparisons[rid] = record
                per_candidate[rid]["vs_leader"] = {"leader": leader_id, **record}
                logger.debug(
                    "among-candidates %s vs %s: b=%d c=%d z=%.2f z_cluster=%.2f p=%.4f "
                    "threshold=%.2f separated=%s",
                    leader_id,
                    rid,
                    record["b"],
                    record["c"],
                    record["z"],
                    record["z_cluster"],
                    record["p"],
                    among_z_threshold,
                    record["separated"],
                )
            tied = [rid for rid in rivals if not comparisons[rid]["separated"]]

            if not rivals:
                chosen_id = leader_id
            elif not tied:
                chosen_id, among_tie_break = leader_id, "separated"
            else:
                chosen_id, among_tie_break = _tie_break_among(
                    [leader_id, *tied],
                    complexity=complexity_by_cand or {},
                    change=change_size_by_cand or {},
                    order=order,
                )

            if rivals:
                among_note = _among_note(leader_id, comparisons, tied, among_tie_break, chosen_id)

            # Whoever came out of that still has to be ahead of the baseline
            # on concordance. This is the pre-2.6 `top > baseline_sel` guard,
            # kept and moved onto the chosen candidate: clearing the paired
            # gate on b − c does not imply a higher mean credit once the
            # candidate's own ties are paid at half.
            if chosen_id is not None and sel_by_id[chosen_id] > baseline_sel:
                winner = next(e for e in challengers if e.candidate_id == chosen_id)
            else:
                if chosen_id is not None and rivals:
                    among_note += (
                        f" {chosen_id} is not ahead of the baseline on selection concordance "
                        f"({sel_by_id[chosen_id]:.4f} <= {baseline_sel:.4f}), so the baseline "
                        "stands anyway."
                    )
                    among_tie_break = "below_baseline"
                chosen_id = None

            if rivals:
                per_candidate[leader_id]["among_candidates"] = {
                    "leader": True,
                    "threshold": among_z_threshold,
                    "tested": rivals,
                    "tied_with": tied,
                    "tie_break": among_tie_break,
                    "chosen": chosen_id,
                }
        if winner.is_baseline and simplification_eligible:
            # No regular candidate passed: the simplification that keeps the
            # most hidden pairs wins, and an exact tie there goes to the
            # smaller filter (then proposal order). A tie with the baseline's
            # own concordance is expected here and is not a disqualification.
            complexity = complexity_by_cand or {}
            best_id = min(
                simplification_eligible,
                key=lambda item: (
                    -item[1],
                    complexity.get(item[0], float("inf")),
                    order[item[0]],
                ),
            )[0]
            winner = next(e for e in challengers if e.candidate_id == best_id)
            simplification_won = True

        # The stats the result reports: the winner's, or when the baseline is
        # kept, the strongest challenger's so the CSV shows what was refused.
        reported_id = winner.candidate_id
        if winner.is_baseline and per_candidate:
            reported_id = max(per_candidate, key=lambda cid: per_candidate[cid]["z"])
        reported = per_candidate.get(reported_id)

        if winner.is_baseline:
            if reported is None:
                reason = (
                    f"kept baseline (sel_concordance={baseline_sel:.4f}); "
                    "no valid candidate to compare."
                )
            elif eligible:
                reason = (
                    f"kept baseline (sel_concordance={baseline_sel:.4f}); "
                    f"{len(eligible)} eligible but none strictly ahead. Closest: "
                    f"{reported_id} b={reported['b']} "
                    f"c={reported['c']}, {_needed_b_text(reported)}; z={reported['z']:.2f} "
                    f"p={reported['p']:.4f} threshold={reported['threshold']:.2f}."
                )
            else:
                reason = (
                    f"kept baseline (sel_concordance={baseline_sel:.4f}); no candidate "
                    f"passed the paired gate. Closest: {reported_id} b={reported['b']} "
                    f"c={reported['c']}, {_needed_b_text(reported)}; z={reported['z']:.2f} "
                    f"p={reported['p']:.4f} threshold={reported['threshold']:.2f} "
                    f"({'; '.join(reported['blockers'])})."
                )
        elif simplification_won:
            assert reported is not None
            reason = (
                f"selected {reported_id} as a SIMPLIFICATION — sel_concordance "
                f"{reported['sel_concordance']:.4f} vs baseline {baseline_sel:.4f}; "
                f"b={reported['b']} >= c={reported['c']}; no regular candidate passed the "
                "paired gate, and a candidate smaller in every complexity component is "
                "accepted when it loses no hidden pair."
            )
        else:
            assert reported is not None
            reason = (
                f"selected {reported_id} — sel_concordance {reported['sel_concordance']:.4f} "
                f"> baseline {baseline_sel:.4f}; b={reported['b']} c={reported['c']} "
                f"p={reported['p']:.4f} <= p_threshold {reported['p_threshold']:.4f} "
                f"(z={reported['z']:.2f} vs threshold {reported['threshold']:.2f}, "
                f"{reported['method']}; z_cluster={reported['z_cluster']:.2f})."
            )
        if among_note:
            reason += " " + among_note
        if regression_fallback:
            reason += (
                " NOTE: every candidate regressed on historical counterexamples; "
                "the regression filter was lifted for this iteration."
            )

        ranked_ids = [winner.candidate_id] + sorted(
            (e.candidate_id for e in valid if e.candidate_id != winner.candidate_id),
            key=lambda cid: (
                -(per_candidate[cid]["eligible"] if cid in per_candidate else False),
                -(per_candidate[cid]["sel_concordance"] if cid in per_candidate else baseline_sel),
                -(per_candidate[cid]["z"] if cid in per_candidate else 0.0),
            ),
        )
        pareto_complexity = _pareto_frontier(evaluations, x_attr="complexity_score", y_attr="loss")
        pareto_cost = _pareto_frontier(evaluations, x_attr="estimated_filter_cost", y_attr="loss")
        winner_safe = winner.candidate_id in pareto_complexity or winner.candidate_id in pareto_cost
        return SelectionResult(
            winner=winner.candidate_id,
            reason=reason,
            ranked=ranked_ids,
            pareto_ids_complexity=pareto_complexity,
            pareto_ids_cost=pareto_cost,
            winner_pareto_safe=winner_safe,
            score=winner.metrics.score,
            baseline_score=baseline.metrics.score,
            gate="mcnemar",
            sel_concordance=baseline_sel if winner.is_baseline else reported["sel_concordance"],
            baseline_sel_concordance=baseline_sel,
            b=reported["b"] if reported else 0,
            c=reported["c"] if reported else 0,
            z=reported["z"] if reported else 0.0,
            p=reported["p"] if reported else 1.0,
            z_cluster=reported["z_cluster"] if reported else 0.0,
            threshold=reported["threshold"] if reported else threshold,
            per_candidate=per_candidate,
            simplification_winner=simplification_won,
            among_tie_break=among_tie_break,
        )

    def _regression_clean(self, e: CandidateEvaluation) -> bool:
        return (
            e.metrics.regression_must_accept_violations <= self.regression_max_violations
            and e.metrics.regression_must_reject_violations <= self.regression_max_violations
        )


def _mean_credit(correctness: Mapping[Pair, float]) -> float:
    """Selection concordance: mean per-pair credit, own ties at 0.5."""

    return sum(correctness.values()) / len(correctness) if correctness else 0.0


def _separation(
    leader: Mapping[Pair, float],
    rival: Mapping[Pair, float],
    *,
    threshold: float,
    min_discordant: int,
    seed: int,
) -> dict:
    """Is the leader *measurably* ahead of one rival on the same pairs?

    The baseline gate's machinery, pointed at two candidates instead of at a
    candidate and the baseline: `paired_stats` for b, c and the exact/normal
    one-sided p, `cluster_bootstrap_z` for the row sharing that makes the
    plain z optimistic. Both have to clear `threshold`, deliberately — gating
    on p is what makes the exact binomial bind below 25 discordant pairs (b=6,
    c=0 is z 2.45 and exact p 0.0156), and gating on `z_cluster` is what keeps
    one re-scored row that sits in 67 pairs from counting as 67 independent
    wins.

    UNCALIBRATED, and a pass here must not be read as the evidence a pass
    against the baseline is. The two nulls are not the same null. A candidate
    and the baseline are scored in separate passes; two candidates share the
    cached cells of every criterion neither of them edited, so their scores
    are correlated and their difference is quieter — the concordance-
    difference noise at the cell sharing the loop actually gets is about
    two thirds of what it is with independent cells. McNemar
    conditions on the discordant set and so absorbs part of that, but how much
    is unmeasured, and the part it does not absorb runs the wrong way for us:
    the discordant pairs that survive sharing are concentrated in the rows the
    edited criteria move, which is exactly the row dependence
    `cluster_bootstrap_z`'s resampling assumes it can see.

    What would calibrate it, and nothing else will: a placebo *pair* — two
    candidates whose edits are known to be equivalent, or one candidate scored
    twice through the candidate path — drawn at several sharing levels, with
    the empirical quantile of `z_cluster` under that null read off directly.
    Until those draws exist the
    threshold is a judgement about which error we would rather make.

    Args:
        leader: The leading candidate's per-pair credit.
        rival: The other candidate's per-pair credit, same pairs.
        threshold: z the leader must clear on both statistics.
        min_discordant: Below this many discordant pairs the two candidates
            barely differ on the exam and nothing is separable; same floor the
            baseline gate uses.
        seed: Seed for `cluster_bootstrap_z`.

    Returns:
        The comparison as written to `selection.json`: b, c, z, p, method,
        z_cluster, threshold, and `separated`. Oriented leader-first — `b`
        counts pairs the leader orders as the judge did and the rival does
        not, `c` the reverse — even though the caller files the record under
        the rival's id.
    """

    stats = paired_stats(leader, rival)
    z_cluster = cluster_bootstrap_z(leader, rival, seed=seed)
    p_threshold = one_sided_p_from_z(threshold)
    separated = (
        stats.n_discordant >= min_discordant
        and stats.p_one_sided <= p_threshold
        and z_cluster >= threshold
    )
    return {
        **_stats_dict(stats),
        "z_cluster": z_cluster,
        "threshold": threshold,
        "p_threshold": p_threshold,
        "separated": separated,
    }


def _tie_break_among(
    group: list[str],
    *,
    complexity: Mapping[str, float],
    change: Mapping[str, float],
    order: Mapping[str, int],
) -> tuple[str | None, str]:
    """Settle a group the paired test could not separate, off the noisy metric.

    Lowest complexity first — the rule the simplification lane already runs on,
    and the only signal in the loop that pushes back on growth. Then the
    smallest change, on the same "a smaller step is the safer step" argument.
    Only then proposal order, and only when something substantive was actually
    measured: an unmeasured signal leaves the group *unknown*, not tied, and
    ordering unknowns by the proposer's output order is the coin flip this
    exists to stop. A group nothing could speak to returns `None`, which the
    caller reads as "keep the baseline".

    The case for letting proposal order decide anyway is real: every
    candidate in the group has already cleared the paired gate, so holding
    discards a measured improvement; and proposal order at least does not
    look at the noisy metric, so whatever it picks reports a concordance that
    means what it says. Three things answer it, the third decisively.

    The baseline gate those candidates cleared is itself uncalibrated, so "a
    measured improvement" is a weaker claim here than it sounds. A filter
    chosen by nothing is not inert either: it is what the proposer reads and
    edits for the rest of the run, so holding costs one iteration while
    promoting on output order steers every iteration after it.

    And an unconditional proposal-order fallback breaks two invariants:
    `test_exact_tie_between_eligible_candidates_keeps_baseline` (an exact tie
    among the eligible "is not evidence of an improvement and keeps the
    baseline") and `test_no_loss_or_complexity_tie_break` (two candidates with
    identical pair vectors must NOT be separated by the selector — size
    breaking ties would let filters grow without resistance). Both go red the
    moment this returns a winner for a group nothing measured.

    Missing ids sort last (`inf`) rather than at zero, so an unmeasured
    candidate cannot win a tie on a default — the same rule, for the same
    reason, as the simplification lane's `complexity.get(cid, inf)`.

    Args:
        group: The tied candidate ids; the leader first.
        complexity: Candidate id to complexity score; may be empty.
        change: Candidate id to change size; may be empty.
        order: Candidate id to proposal index.

    Returns:
        ``(winner_id, tie_break_name)``, or ``(None, "hold")`` when no signal
        could separate the group.
    """

    measured = False
    for name, signal in (("complexity", complexity), ("change", change)):
        values = [signal.get(cid, math.inf) for cid in group]
        if not all(math.isfinite(v) for v in values):
            # Somebody in the group was never measured on this axis, so it
            # cannot say they are equal and it cannot say they differ.
            continue
        measured = True
        best = min(values)
        leaders = [cid for cid, value in zip(group, values, strict=True) if value == best]
        if len(leaders) == 1:
            return leaders[0], name
    if measured:
        return min(group, key=order.__getitem__), "order"
    return None, "hold"


def _among_note(
    leader: str,
    comparisons: Mapping[str, dict],
    tied: list[str],
    tie_break: str,
    chosen: str | None,
) -> str:
    """The among-candidates clause of the reason string: what that test did.

    A run log that says only "selected c3" cannot be audited afterwards. This
    is the sentence that lets someone reading `selection.json` six weeks later
    see that c2 led, failed to separate, and lost the tie on size.
    """

    if not comparisons:
        return ""
    threshold = next(iter(comparisons.values()))["threshold"]
    if not tied:
        closest = min(comparisons, key=lambda cid: comparisons[cid]["z_cluster"])
        rec = comparisons[closest]
        return (
            f"Leader {leader} separated from every other eligible candidate "
            f"({len(comparisons)} tested); closest was {closest} b={rec['b']} c={rec['c']} "
            f"z={rec['z']:.2f} z_cluster={rec['z_cluster']:.2f} >= {threshold:.2f}."
        )
    detail = "; ".join(
        f"vs {cid}: b={comparisons[cid]['b']} c={comparisons[cid]['c']} "
        f"z={comparisons[cid]['z']:.2f} z_cluster={comparisons[cid]['z_cluster']:.2f}"
        for cid in tied
    )
    head = (
        f"Leader {leader} was not separable from {', '.join(tied)} "
        f"({detail}; threshold {threshold:.2f})"
    )
    if tie_break == "hold":
        return (
            f"{head}, and no tie-break separated them either — the baseline stands rather "
            "than a coin flip."
        )
    return f"{head}; fell through to {tie_break}, which chose {chosen}."


def _needed_b_text(record: Mapping) -> str:
    """The "needed b ≥ 19" clause for a per-candidate record, or why no b would do.

    Reads `min_b_required` (from `paired_test.critical_b` at the candidate's
    own b + c and p threshold), so the kept-baseline line says how far the
    closest challenger was in pairs, not only in z.
    """

    needed = record.get("min_b_required")
    if needed is None:
        return f"no b reaches the threshold at b+c={record['b'] + record['c']}"
    return f"needed b ≥ {needed}"


def _stats_dict(stats: PairedStats) -> dict:
    """The `PairedStats` fields under the names `selection.json` uses."""

    return {
        "n_pairs": stats.n_pairs,
        "both_right": stats.both_right,
        "both_wrong": stats.both_wrong,
        "b": stats.b,
        "c": stats.c,
        "diff": stats.diff,
        "z": stats.z,
        "p": stats.p_one_sided,
        "method": stats.method,
    }


def _pareto_frontier(
    evaluations: list[CandidateEvaluation],
    *,
    x_attr: str,
    y_attr: str,
) -> list[str]:
    """Return candidate_ids on the (x, y) Pareto frontier — both lower-better.

    A candidate is on the frontier iff no other candidate has (x' <= x, y' <= y)
    with at least one strict inequality.
    """

    points: list[tuple[str, float, float]] = [
        (
            e.candidate_id,
            float(getattr(e.metrics, x_attr, 0.0)),
            float(getattr(e.metrics, y_attr, 0.0)),
        )
        for e in evaluations
        if e.invalid_reason is None
    ]
    if not points:
        return []
    # Sort by x ascending; for ties in x, prefer lower y so we keep the
    # dominating one first.
    points.sort(key=lambda p: (p[1], p[2]))
    frontier: list[str] = []
    best_y = float("inf")
    for cid, _x, y in points:
        if y < best_y:
            frontier.append(cid)
            best_y = y
    return frontier
