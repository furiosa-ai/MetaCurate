"""Tests for the pure ranking math in meta_filter.gate.ranking."""

from __future__ import annotations

import json
import math
import random

from meta_filter.gate.ranking import (
    PairTruth,
    concordance,
    default_min_distinct_levels,
    distinct_levels,
    distribution_report,
    kendall_distance,
    render_distribution,
    truth_pairs,
)


def test_truth_pairs_consensus_classification() -> None:
    """All candidates agreeing strictly yields (hi, lo) consensus pairs."""
    scores = {
        "cand_a": {"t1": 3.0, "t2": 2.0, "t3": 1.0},
        "cand_b": {"t1": 30.0, "t2": 20.0, "t3": 10.0},
    }
    truth = truth_pairs(scores)
    assert truth.sampled_universe == 3
    assert sorted(truth.consensus_pairs) == [("t1", "t2"), ("t1", "t3"), ("t2", "t3")]
    assert truth.disagreement_pairs == []


def test_truth_pairs_disagreement_classification() -> None:
    """A pair ordered opposite ways by two candidates becomes a disagreement."""
    scores = {
        "cand_a": {"t1": 1.0, "t2": 2.0},
        "cand_b": {"t1": 2.0, "t2": 1.0},
    }
    truth = truth_pairs(scores)
    assert truth.consensus_pairs == []
    assert truth.disagreement_pairs == [("t1", "t2")]


def test_truth_pairs_tie_is_abstention_not_veto() -> None:
    """A tie sits out; strict votes decide. One lone strict vote is skipped."""
    scores = {
        "cand_a": {"t1": 1.0, "t2": 1.0},  # tie -> abstains
        "cand_b": {"t1": 2.0, "t2": 1.0},  # the only strict vote
    }
    truth = truth_pairs(scores)
    # A single strict vote is an opinion, not consensus.
    assert truth.consensus_pairs == []
    assert truth.disagreement_pairs == []
    assert truth.sampled_universe == 1


def test_truth_pairs_degenerate_candidate_cannot_poison_channel() -> None:
    """An all-tie candidate must not erase truth."""
    scores = {
        "cand_a": {"t1": 3.0, "t2": 1.0},
        "cand_b": {"t1": 4.0, "t2": 2.0},
        "degen": {"t1": 5.0, "t2": 5.0},  # constant scorer, ties everything
    }
    truth = truth_pairs(scores)
    assert truth.consensus_pairs == [("t1", "t2")]
    assert truth.disagreement_pairs == []


def test_truth_pairs_disagreement_survives_a_tie_voter() -> None:
    scores = {
        "cand_a": {"t1": 3.0, "t2": 1.0},
        "cand_b": {"t1": 1.0, "t2": 3.0},
        "degen": {"t1": 5.0, "t2": 5.0},
    }
    truth = truth_pairs(scores)
    assert truth.consensus_pairs == []
    assert truth.disagreement_pairs == [("t1", "t2")]


def test_truth_pairs_uses_id_intersection() -> None:
    """Only trajectory ids present in every candidate's map participate."""
    scores = {
        "cand_a": {"t1": 2.0, "t2": 1.0, "only_a": 9.0},
        "cand_b": {"t1": 4.0, "t2": 3.0, "only_b": 9.0},
    }
    truth = truth_pairs(scores)
    assert truth.sampled_universe == 1
    assert truth.consensus_pairs == [("t1", "t2")]
    assert all("only_a" not in p and "only_b" not in p for p in truth.consensus_pairs)


def _disagreeing_candidates(n: int) -> dict[str, dict[str, float]]:
    """Two candidates that disagree on every pair of n trajectories."""
    ids = [f"t{i:03d}" for i in range(n)]
    ascending = {tid: float(i) for i, tid in enumerate(ids)}
    descending = {tid: float(n - i) for i, tid in enumerate(ids)}
    return {"cand_a": ascending, "cand_b": descending}


def test_truth_pairs_deterministic_and_budget_respected() -> None:
    """Same seed reproduces the exact sample; budget caps the output size."""
    scores = _disagreeing_candidates(30)
    first = truth_pairs(scores, disagreement_budget=12, seed=17)
    second = truth_pairs(scores, disagreement_budget=12, seed=17)
    assert first == second
    assert isinstance(first, PairTruth)
    assert len(first.disagreement_pairs) == 12
    assert len(set(first.disagreement_pairs)) == 12


def test_truth_pairs_stratified_across_deciles() -> None:
    """A small budget still draws pairs from across the mean-score range."""
    scores = _disagreeing_candidates(40)
    truth = truth_pairs(scores, disagreement_budget=10, seed=17)
    assert len(truth.disagreement_pairs) == 10

    means = []
    for a, b in truth.disagreement_pairs:
        per_candidate = [(scores[c][a] + scores[c][b]) / 2 for c in ("cand_a", "cand_b")]
        means.append(sum(per_candidate) / 2)
    # With both candidates mirror images, the pair mean is constant per pair
    # only in cand-mean; use cand_a means to check the spread instead.
    cand_a_means = [
        (scores["cand_a"][a] + scores["cand_a"][b]) / 2 for a, b in truth.disagreement_pairs
    ]
    spread = max(cand_a_means) - min(cand_a_means)
    assert spread > 10.0, "stratified sample should span the score range"


def test_truth_pairs_universe_sampling_cap() -> None:
    """When the pair universe exceeds max_pairs, exactly max_pairs are examined."""
    ids = {f"t{i:03d}": float(i) for i in range(50)}  # 1225 pairs total
    scores = {"cand_a": ids, "cand_b": dict(ids)}
    truth = truth_pairs(scores, max_pairs=100, seed=17)
    assert truth.sampled_universe == 100
    assert len(truth.consensus_pairs) == 100  # identical candidates agree everywhere
    again = truth_pairs(scores, max_pairs=100, seed=17)
    assert truth == again


def test_concordance_correct_ties_and_missing_ids() -> None:
    """Concordance credits 1.0 for correct order, 0.5 for ties, skips missing."""
    scores = {"a": 3.0, "b": 2.0, "c": 2.0}
    pairs = [
        ("a", "b"),  # correct -> 1.0
        ("b", "a"),  # wrong -> 0.0
        ("b", "c"),  # tie -> 0.5
        ("a", "zz"),  # missing id -> skipped
    ]
    assert concordance(scores, pairs) == (1.0 + 0.0 + 0.5) / 3


def test_concordance_empty_input() -> None:
    """Empty pair lists and all-missing pairs both yield 0.0."""
    assert concordance({"a": 1.0}, []) == 0.0
    assert concordance({}, [("a", "b")]) == 0.0


def test_kendall_distance_symmetry_and_extremes() -> None:
    """Distance is symmetric, 0 on identical maps, 1 on fully reversed maps."""
    rng = random.Random(3)
    scores_a = {f"t{i}": rng.random() for i in range(20)}
    scores_b = {tid: -v for tid, v in scores_a.items()}
    assert kendall_distance(scores_a, scores_a) == 0.0
    assert kendall_distance(scores_a, scores_b) == 1.0
    scores_c = {f"t{i}": rng.random() for i in range(20)}
    assert kendall_distance(scores_a, scores_c) == kendall_distance(scores_c, scores_a)


def test_kendall_distance_tie_cases() -> None:
    """One-sided ties count 0.5, two-sided ties 0.0; tiny maps return 0.0."""
    tied = {"a": 1.0, "b": 1.0}
    strict = {"a": 2.0, "b": 1.0}
    assert kendall_distance(tied, strict) == 0.5
    assert kendall_distance(tied, dict(tied)) == 0.0
    assert kendall_distance({"a": 1.0}, {"a": 2.0}) == 0.0
    assert kendall_distance({}, {}) == 0.0


def test_distinct_levels_rounding() -> None:
    """Values equal at 6 decimals collapse to one level."""
    assert distinct_levels({"a": 1.0000001, "b": 1.0000002}) == 1
    assert distinct_levels({"a": 1.00001, "b": 1.00002}) == 2
    assert distinct_levels({}) == 0


def test_distribution_report_bimodal_with_resolved() -> None:
    """Bimodal scores land in the extreme bins and separation is computed."""
    scores = {f"lo{i}": 0.0 + i * 0.001 for i in range(5)}
    scores |= {f"hi{i}": 1.0 - i * 0.001 for i in range(5)}
    resolved = {tid: tid.startswith("hi") for tid in scores}
    report = distribution_report(scores, resolved)

    assert report["n"] == 10
    assert sum(report["decile_counts"]) == 10
    assert report["decile_counts"][0] == 5
    assert report["decile_counts"][9] == 5
    assert math.isclose(report["entropy"], 1.0, abs_tol=1e-9)
    assert report["separation"] == report["resolved_mean"] - report["unresolved_mean"]
    assert report["separation"] > 0.9
    json.dumps(report)  # JSON-safe


def test_distribution_report_all_equal_and_empty() -> None:
    """All-equal scores fill one bin; empty input stays well-formed."""
    report = distribution_report({"a": 2.0, "b": 2.0, "c": 2.0})
    assert report["decile_counts"] == [3] + [0] * 9
    assert report["entropy"] == 0.0
    assert report["distinct_levels"] == 1

    empty = distribution_report({})
    assert empty["n"] == 0
    assert empty["decile_counts"] == [0] * 10
    assert "separation" not in empty


def test_render_distribution_stable_and_bounded() -> None:
    """Rendering is deterministic, bars cap at 20 chars, separation is shown."""
    scores = {f"t{i}": i / 9 for i in range(10)}
    resolved = {f"t{i}": i >= 5 for i in range(10)}
    report = distribution_report(scores, resolved)

    rendered = render_distribution(report)
    assert rendered == render_distribution(report)
    lines = rendered.splitlines()
    assert len(lines) == 12  # summary + 10 deciles + separation
    assert lines[0].startswith("n=10 mean=")
    assert all(line.count("#") <= 20 for line in lines[1:11])
    assert max(line.count("#") for line in lines[1:11]) == 20
    assert "separation=" in lines[-1]

    without_resolved = render_distribution(distribution_report(scores))
    assert len(without_resolved.splitlines()) == 11


def test_missed_pairs_orders_by_inversion_gap() -> None:
    from meta_filter.gate.ranking import missed_pairs

    scores = {"w1": 10.0, "l1": 40.0, "w2": 50.0, "l2": 55.0, "w3": 90.0, "l3": 10.0}
    judged = [("w1", "l1"), ("w2", "l2"), ("w3", "l3"), ("w4", "l4")]
    missed = missed_pairs(scores, judged)
    # w3 > l3 is ordered correctly; w4/l4 unknown ids skipped; the two
    # misses come back most-confidently-wrong first.
    assert [(a, b) for a, b, _ in missed] == [("w1", "l1"), ("w2", "l2")]
    assert missed[0][2] == 30.0


def test_missed_pairs_ties_are_not_misses() -> None:
    from meta_filter.gate.ranking import missed_pairs

    assert missed_pairs({"a": 5.0, "b": 5.0}, [("a", "b")]) == []


# ------------------------------------------- full pair universe / consensus


def test_max_pairs_controls_how_much_of_the_universe_is_classified() -> None:
    """The cap is a sampling decision, not a correctness one.

    200 rows make 19,900 pairs; the default looks at 4,000. Classification
    costs nothing beyond sign arithmetic over scores already paid for, so the
    cap's only real effect is shrinking the disagreement pool the judge budget
    draws from.
    """
    scores = {
        "a": {f"t{i}": float(i) for i in range(200)},
        "b": {f"t{i}": float((i * 7) % 200) for i in range(200)},
    }
    capped = truth_pairs(scores, max_pairs=4000)
    full = truth_pairs(scores, max_pairs=19_900)
    assert capped.sampled_universe == 4000
    assert full.sampled_universe == 19_900
    # Five times the universe, and the non-consensus pool scales with it.
    capped_disputed = capped.sampled_universe - len(capped.consensus_pairs)
    full_disputed = full.sampled_universe - len(full.consensus_pairs)
    assert full_disputed > capped_disputed * 3


def test_audit_share_spends_budget_on_consensus_every_iteration() -> None:
    """A fixed share of the judge budget always audits agreed pairs.

    Not a fallback for an empty disagreement set: the audit is what makes
    "every candidate agrees" falsifiable. Without it a committee that ranked
    the pool exactly backwards in unison would produce no disagreement,
    consensus would be the entire truth set, and concordance would read 1.0.
    """
    scores = {
        "a": {f"t{i}": float(i) for i in range(40)},
        "b": {f"t{i}": float(i) * 2 for i in range(40)},  # agrees everywhere
    }
    plain = truth_pairs(scores, disagreement_budget=20)
    assert plain.audit_pairs == [], "off by default"

    audited = truth_pairs(scores, disagreement_budget=20, consensus_share=0.25)
    # Every pair is consensus here, so the disagreement stratum is empty and
    # its share passes to the audit — a sampler exhausting one stratum.
    assert len(audited.audit_pairs) == 20
    judged = {frozenset(p) for p in audited.audit_pairs}
    assert judged & {frozenset(p) for p in audited.consensus_pairs} == set()


def test_audit_share_leaves_disagreement_the_larger_share() -> None:
    """Selection signal keeps its budget; the audit takes only its share."""
    scores = {
        "a": {f"t{i}": float(i) for i in range(40)},
        "b": {f"t{i}": float(-i) for i in range(40)},  # disputes everywhere
    }
    audited = truth_pairs(scores, disagreement_budget=20, consensus_share=0.25)
    assert len(audited.audit_pairs) == 0, "no consensus exists to audit"
    assert len(audited.disagreement_pairs) == 15, "20 minus the 25% audit share"


def test_audit_pairs_keep_the_candidates_order() -> None:
    """The audit compares two orderings, so the candidates' one must survive."""
    scores = {
        "a": {"hi": 10.0, "lo": 1.0},
        "b": {"hi": 20.0, "lo": 2.0},
    }
    audited = truth_pairs(scores, disagreement_budget=4, consensus_share=1.0)
    assert audited.audit_pairs == [("hi", "lo")]


def test_audit_is_off_by_default() -> None:
    """The default path must stay identical to every run before this."""
    scores = {
        "a": {f"t{i}": float(i) for i in range(30)},
        "b": {f"t{i}": float(i) * 2 for i in range(30)},
    }
    assert truth_pairs(scores, disagreement_budget=20) == truth_pairs(
        scores, disagreement_budget=20, consensus_share=0.0
    )


# --- the degeneracy floor scales with the pool -------------------------------


def test_the_floor_is_five_at_two_hundred_rows() -> None:
    """5 at 200 rows, the size the constant was chosen for.

    The guard has to keep catching the same failure across pool sizes, or
    numbers from a 50-row run and a 200-row run are not comparable — the
    smaller pool would simply be failing more candidates.
    """

    assert default_min_distinct_levels(200) == 5


def test_a_fifty_row_pool_gets_a_lower_floor_not_a_higher_one() -> None:
    """A 50-row pool has fewer levels available, so the floor must come DOWN.

    This is a degeneracy guard, not a resolution target. A floor set near the
    pool size fails candidates that rank the pool perfectly well in coarse
    steps, which is the opposite of what it is for.
    """

    assert default_min_distinct_levels(50) == 3
    assert default_min_distinct_levels(50) < default_min_distinct_levels(200)


def test_the_floor_never_drops_below_three() -> None:
    """Two levels is "better or worse" — a gate, not a ranking."""

    assert default_min_distinct_levels(12) == 3
    assert default_min_distinct_levels(1) == 3
    assert default_min_distinct_levels(0) == 3
