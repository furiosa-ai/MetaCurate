"""The paired arithmetic the acceptance gate runs on.

Hand-built correctness maps, so every b, c, z and p below can be checked on
paper. The cluster bootstrap test builds the case the plain z gets wrong on
purpose: every discordant pair hangs off one row, so the "evidence" is one
row's re-score counted many times.
"""

from __future__ import annotations

import math

import pytest

from meta_filter.gate.paired_test import (
    CRITICAL_B_TABLE_NS,
    EXACT_BINOMIAL_BELOW,
    PairedStats,
    cluster_bootstrap_z,
    critical_b,
    critical_b_table,
    one_sided_p_from_z,
    pair_correctness,
    paired_stats,
    render_critical_b_table,
    stats_from_counts,
    z_from_one_sided_p,
    z_threshold_for,
)

# --- pair_correctness -------------------------------------------------------


def test_pair_correctness_mirrors_concordance_detail():
    scores = {"a": 3.0, "b": 2.0, "c": 2.0, "d": 1.0}
    pairs = [("a", "b"), ("b", "c"), ("d", "a"), ("a", "zz")]
    credit = pair_correctness(scores, pairs)
    assert credit == {
        ("a", "b"): 1.0,  # ordered as judged
        ("b", "c"): 0.5,  # candidate's own tie earns half
        ("d", "a"): 0.0,  # reversed
        ("a", "zz"): 0.0,  # missing id is a wrong answer, not a skipped pair
    }


def test_judge_ties_leave_the_map_in_either_orientation():
    scores = {"a": 3.0, "b": 2.0, "c": 1.0}
    pairs = [("a", "b"), ("b", "c")]
    credit = pair_correctness(scores, pairs, ties=[("c", "b")])
    assert set(credit) == {("a", "b")}


# --- paired_stats -----------------------------------------------------------


def _maps(pattern: str) -> tuple[dict, dict]:
    """Build (cand, base) maps from a string of cell codes.

    ``R`` both right, ``W`` both wrong, ``B`` candidate right / baseline wrong,
    ``C`` the reverse, ``T`` candidate ties (0.5) where the baseline is right.
    """

    cand: dict = {}
    base: dict = {}
    for i, code in enumerate(pattern):
        key = (f"h{i}", f"l{i}")
        cand[key], base[key] = {
            "R": (1.0, 1.0),
            "W": (0.0, 0.0),
            "B": (1.0, 0.0),
            "C": (0.0, 1.0),
            "T": (0.5, 1.0),
        }[code]
    return cand, base


def test_b_c_z_on_a_hand_built_map():
    cand, base = _maps("RRRWW" + "B" * 9 + "C" * 4)
    stats = paired_stats(cand, base)
    assert isinstance(stats, PairedStats)
    assert (stats.n_pairs, stats.both_right, stats.both_wrong) == (18, 3, 2)
    assert (stats.b, stats.c, stats.n_discordant) == (9, 4, 13)
    assert stats.diff == pytest.approx(5 / 18)
    assert stats.z == pytest.approx(5 / math.sqrt(13))


def test_own_tie_counts_as_wrong_for_the_paired_test():
    cand, base = _maps("TTT")
    stats = paired_stats(cand, base)
    assert (stats.b, stats.c) == (0, 3)
    assert stats.z < 0


def test_exact_binomial_below_25_discordant():
    cand, base = _maps("B" * 8 + "C" * 2)
    stats = paired_stats(cand, base)
    assert stats.method == "exact"
    # P(X >= 8 | n=10, 0.5) = (45 + 10 + 1) / 1024
    assert stats.p_one_sided == pytest.approx(56 / 1024)


def test_normal_approximation_from_25_discordant():
    cand, base = _maps("B" * 18 + "C" * 7)
    stats = paired_stats(cand, base)
    assert stats.method == "normal"
    z = 11 / math.sqrt(25)
    assert stats.z == pytest.approx(z)
    assert stats.p_one_sided == pytest.approx(0.5 * math.erfc(z / math.sqrt(2)))


def test_no_discordant_pairs_means_nothing_to_test():
    cand, base = _maps("RRWW")
    stats = paired_stats(cand, base)
    assert (stats.b, stats.c, stats.z, stats.p_one_sided, stats.method) == (0, 0, 0.0, 1.0, "none")


def test_mismatched_pair_sets_are_a_wiring_bug():
    cand, base = _maps("RB")
    del base[("h1", "l1")]
    with pytest.raises(ValueError, match="same pair set"):
        paired_stats(cand, base)


# --- thresholds -------------------------------------------------------------


def test_z_threshold_for_three_candidates_is_about_2_4():
    z3 = z_threshold_for(3)
    assert 2.38 <= z3 <= 2.45
    assert z_threshold_for(1) == pytest.approx(2.0, abs=1e-6)
    assert z_threshold_for(2) < z3 < z_threshold_for(4)


def test_z_threshold_for_rejects_zero_comparisons():
    with pytest.raises(ValueError):
        z_threshold_for(0)


def test_inverse_normal_round_trips():
    for z in (-1.0, 0.0, 1.0, 2.0, 2.4, 3.5):
        assert z_from_one_sided_p(one_sided_p_from_z(z)) == pytest.approx(z, abs=1e-9)


# --- cluster bootstrap ------------------------------------------------------


def test_cluster_bootstrap_is_finite_and_below_plain_z_when_pairs_share_a_row():
    # One row "star" the candidate re-scored above 16 partners: 16 discordant
    # pairs, all of them the same single piece of evidence.
    cand: dict = {}
    base: dict = {}
    for i in range(16):
        key = ("star", f"p{i}")
        cand[key], base[key] = 1.0, 0.0
    for i in range(20):
        key = (f"x{i}", f"y{i}")
        cand[key], base[key] = 1.0, 1.0

    plain = paired_stats(cand, base).z
    assert plain == pytest.approx(4.0)
    boot = cluster_bootstrap_z(cand, base, n_resamples=500, seed=3)
    assert math.isfinite(boot)
    assert 0.0 < boot < plain


def test_cluster_bootstrap_is_deterministic_in_its_seed():
    cand, base = _maps("B" * 10 + "C" * 3 + "R" * 5)
    first = cluster_bootstrap_z(cand, base, n_resamples=200, seed=7)
    again = cluster_bootstrap_z(cand, base, n_resamples=200, seed=7)
    assert first == again
    assert math.isfinite(first) and first > 0.0


def test_cluster_bootstrap_with_no_discordant_pairs_is_zero():
    cand, base = _maps("RRWW")
    assert cluster_bootstrap_z(cand, base, n_resamples=50) == 0.0


# --- critical_b: the b the gate needs at each b + c ------------------------------


@pytest.mark.parametrize("z", [2.0, 2.43, 3.43, 4.86])
def test_critical_b_is_where_stats_from_counts_starts_passing(z):
    """For every n up to 120 and every b, p <= alpha iff b >= critical_b(n) —
    the table and the gate are the same test, exact branch and normal branch
    both, with None meaning no b at all passes."""

    alpha = one_sided_p_from_z(z)
    for n in range(1, 121):
        needed = critical_b(n, alpha)
        for b in range(n + 1):
            _, p, method = stats_from_counts(b, n - b)
            assert method == ("exact" if n < EXACT_BINOMIAL_BELOW else "normal")
            passes = p <= alpha
            assert passes == (needed is not None and b >= needed), (z, n, b, needed, p)


def test_critical_b_is_none_below_the_reachable_n():
    """At alpha 0.0076 (z 2.43) a clean sweep first passes at n=8 (1/256);
    below that nothing can. At 0.0003 (z 3.43) the first reachable n is 12
    (1/4096). Zero discordant pairs is p = 1 and never passes."""

    family = one_sided_p_from_z(2.43)
    assert [critical_b(n, family) for n in range(0, 9)] == [None] * 8 + [8]
    strong = one_sided_p_from_z(3.43)
    assert all(critical_b(n, strong) is None for n in range(0, 12))
    assert critical_b(12, strong) == 12
    with pytest.raises(ValueError):
        critical_b(-1, family)


def test_critical_b_hand_checked_rows():
    family = one_sided_p_from_z(2.43)
    # b+c=10: 9-1 is 11/1024 = 0.0107 > 0.0076, 10-0 is 1/1024.
    assert critical_b(10, family) == 10
    # b+c=27, normal branch: 19-8 is z 2.12, 20-7 is z 2.50.
    assert critical_b(27, family) == 20
    # b+c=29: 24-5 (z 3.53) is what the one-z-higher gate needs.
    assert critical_b(29, one_sided_p_from_z(3.43)) == 24


def test_critical_b_table_shape_and_render():
    alpha = one_sided_p_from_z(2.43)
    table = critical_b_table(alpha)
    assert [n for n, _ in table] == list(CRITICAL_B_TABLE_NS)
    assert len(table) == 14
    assert table[0] == (5, None)
    assert table[1] == (8, 8)
    assert all(b is None or 0 <= b <= n for n, b in table)
    assert critical_b_table(alpha, ns=(8, 27)) == [(8, 8), (27, 20)]

    text = render_critical_b_table(alpha, "family z=2.43", ns=(5, 8, 27, 100))
    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("critical b at alpha=7.55e-03 (family z=2.43)")
    assert lines[1].split() == ["b+c:", "5", "8", "27", "100"]
    assert lines[2].split() == ["b>=:", "-", "8", "20", str(critical_b(100, alpha))]
