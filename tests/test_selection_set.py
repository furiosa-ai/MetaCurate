"""Hidden fixed selection set: pair planning, judging, persistence, scoring."""

from __future__ import annotations

import json

import pytest

from meta_filter.pair_judge import PairResolution
from meta_filter.schemas import Decision, FilterDecision, Trajectory, Turn
from meta_filter.selection_set import (
    SELECTION_SET_FILE,
    SelectionSet,
    assert_disjoint,
    build_selection_pairs,
    build_selection_set,
    judge_selection_pairs,
    plan_selection_pairs,
    score_on_selection,
    selection_row_ids_in_text,
)


def _ids(n: int, prefix: str = "sel") -> list[str]:
    return [f"{prefix}-{i:03d}::att0.host" for i in range(n)]


def _traj(tid: str) -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task=f"task for {tid}",
        turns=[Turn(role="assistant", action="read", content="opened a file")],
    )


def _universe_size(n: int) -> int:
    return n * (n - 1) // 2


# --- planning -----------------------------------------------------------------


def test_pairs_are_deterministic_from_seed_and_ids():
    ids = _ids(40)
    first = build_selection_pairs(ids, n_pairs=100, seed=7)
    again = build_selection_pairs(list(reversed(ids)), n_pairs=100, seed=7)
    other_seed = build_selection_pairs(ids, n_pairs=100, seed=8)

    assert first == again, "the sorted id list, not its input order, decides the pairs"
    assert first != other_seed
    assert len(first) == 100


def test_pairs_are_sorted_tuples_with_no_duplicates():
    ids = _ids(30)
    pairs = build_selection_pairs(ids, n_pairs=200, seed=3)

    assert len(pairs) == 200
    assert len({frozenset(p) for p in pairs}) == 200
    assert all(a < b for a, b in pairs)
    assert pairs == sorted(pairs)


def test_universe_no_larger_than_n_is_returned_whole():
    ids = _ids(6)  # C(6, 2) == 15
    plan = plan_selection_pairs(ids, n_pairs=15, seed=1)
    assert len(plan.pairs) == 15 == _universe_size(6)
    assert not plan.stratified
    assert (plan.n_hard, plan.n_uniform) == (0, 15)

    # A budget larger than the universe cannot invent pairs either.
    assert len(build_selection_pairs(ids, n_pairs=500, seed=1)) == 15
    # And with scores present the whole universe still comes back — there is
    # nothing to stratify when everything is taken.
    scores = {tid: float(i) for i, tid in enumerate(ids)}
    assert len(build_selection_pairs(ids, n_pairs=20, seed=1, initial_scores=scores)) == 15


def test_without_scores_the_draw_is_uniform_and_matches_the_probe_convention():
    from meta_filter.probe import fixed_validation_pairs

    ids = _ids(25)
    ours = build_selection_pairs(ids, n_pairs=60, seed=4242)
    theirs = fixed_validation_pairs(ids, n_pairs=60, seed=4242)
    assert ours == theirs


def test_stratification_puts_half_the_pairs_in_the_small_gap_quartile():
    """Scores are integers 0..n-1, so |gap| is |i - j| and the quartile is knowable."""

    n = 40
    ids = _ids(n)
    scores = {tid: float(i) for i, tid in enumerate(ids)}
    n_pairs = 120
    plan = plan_selection_pairs(ids, n_pairs=n_pairs, seed=11, initial_scores=scores)

    assert plan.stratified
    assert len(plan.pairs) == n_pairs
    assert len({frozenset(p) for p in plan.pairs}) == n_pairs
    assert plan.n_hard == n_pairs // 2
    assert plan.n_uniform == n_pairs - plan.n_hard

    # Recover the hard quartile the same way: rank all pairs by gap.
    universe = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]
    by_gap = sorted(universe, key=lambda p: (abs(scores[p[0]] - scores[p[1]]), p))
    hard_quartile = set(by_gap[: len(by_gap) // 4])

    in_hard = sum(1 for p in plan.pairs if p in hard_quartile)
    # Exactly half were drawn from the quartile; the uniform half may land
    # there too only by chance, and it cannot — the uniform draw is over the
    # complement. So the count is exactly n_hard.
    assert in_hard == plan.n_hard

    # Sanity: a uniform draw would put ~25% there, not 50%.
    uniform = build_selection_pairs(ids, n_pairs=n_pairs, seed=11)
    assert sum(1 for p in uniform if p in hard_quartile) < plan.n_hard


def test_stratification_is_deterministic_and_skipped_on_partial_scores():
    ids = _ids(30)
    scores = {tid: float(i % 7) for i, tid in enumerate(ids)}
    a = plan_selection_pairs(ids, n_pairs=80, seed=5, initial_scores=scores)
    b = plan_selection_pairs(ids, n_pairs=80, seed=5, initial_scores=dict(scores))
    assert a == b

    partial = dict(scores)
    partial.pop(ids[3])
    plan = plan_selection_pairs(ids, n_pairs=80, seed=5, initial_scores=partial)
    assert not plan.stratified
    assert plan.pairs == build_selection_pairs(ids, n_pairs=80, seed=5)


def test_stratification_tops_up_from_the_hard_pool_when_the_rest_is_short():
    # 8 ids -> 28 pairs; quartile cut = 7 hard, 21 rest. Ask for 26: 13 hard
    # wanted, 13 rest wanted -> hard pool has only 7, so rest supplies 19.
    ids = _ids(8)
    scores = {tid: float(i) for i, tid in enumerate(ids)}
    plan = plan_selection_pairs(ids, n_pairs=26, seed=2, initial_scores=scores)
    assert len(plan.pairs) == 26
    assert plan.n_hard + plan.n_uniform == 26
    assert plan.n_hard == 7
    assert len({frozenset(p) for p in plan.pairs}) == 26


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError):
        build_selection_pairs(_ids(5), n_pairs=-1, seed=0)


# --- judging ------------------------------------------------------------------


class _FakeJudge:
    """Stands in for PairJudge: resolves by id order, ties and drops on request."""

    def __init__(
        self, *, tie: set[frozenset[str]] = frozenset(), fail: set[frozenset[str]] = frozenset()
    ):
        self.tie = set(tie)
        self.fail = set(fail)
        self.calls: list[tuple[list[tuple[str, str]], int]] = []
        self.client = type("Client", (), {"model": "fake-judge-1"})()
        self.cache_dir = "fake/cache"

    def resolve_pairs_detail(self, pairs, traj_by_id, *, max_workers=8):
        self.calls.append((list(pairs), max_workers))
        out = PairResolution()
        for a, b in pairs:
            if a not in traj_by_id or b not in traj_by_id:
                continue
            key = frozenset((a, b))
            if key in self.fail:
                out.unresolved.append((a, b))
            elif key in self.tie:
                out.tied.append((a, b))
            else:
                # The judge prefers the lexically LATER id, so the resolved
                # order is the reverse of the planned (sorted) order — the
                # test can tell "kept the judge's order" from "kept ours".
                out.resolved.append((max(a, b), min(a, b)))
        return out


def test_judge_selection_pairs_keeps_judge_order_and_drops_ties():
    ids = _ids(5)
    traj_by_id = {tid: _traj(tid) for tid in ids}
    pairs = build_selection_pairs(ids, n_pairs=10, seed=1)  # whole universe
    tie = {frozenset((ids[0], ids[1]))}
    fail = {frozenset((ids[2], ids[3]))}
    judge = _FakeJudge(tie=tie, fail=fail)

    sel = judge_selection_pairs(pairs, traj_by_id, judge, max_workers=3, provenance={"seed": 1})

    assert judge.calls == [(pairs, 3)]
    assert len(sel.pairs) == 8
    assert all(hi > lo for hi, lo in sel.pairs), "resolved pairs keep the judge's (hi, lo)"
    assert sel.tied == [(ids[0], ids[1])]
    assert sel.unresolved == [(ids[2], ids[3])]
    assert sel.row_ids == sorted(ids)
    assert sel.provenance["seed"] == 1
    assert sel.provenance["judge_model"] == "fake-judge-1"
    assert sel.provenance["n_resolved"] == 8
    assert sel.provenance["n_tied"] == 1
    assert sel.provenance["tie_share"] == pytest.approx(1 / 9, abs=1e-4)
    assert "built_at" in sel.provenance
    # Tied pairs are not in the exam in either direction.
    tied_keys = {frozenset(p) for p in sel.tied}
    assert not any(frozenset(p) in tied_keys for p in sel.pairs)


def test_build_selection_set_records_the_plan_provenance():
    ids = _ids(20)
    rows = [_traj(tid) for tid in ids]
    scores = {tid: float(i) for i, tid in enumerate(ids)}
    sel = build_selection_set(rows, _FakeJudge(), n_pairs=50, seed=9, initial_scores=scores)

    assert len(sel.pairs) == 50
    prov = sel.provenance
    assert prov["n_requested"] == 50
    assert prov["seed"] == 9
    assert prov["stratified"] is True
    assert prov["n_hard"] == 25
    assert prov["n_uniform"] == 25
    assert prov["tie_share"] == 0.0
    assert sel.row_ids == sorted(ids)


def test_judge_label_falls_back_to_the_cache_dir():
    class _NoModel:
        cache_dir = "data/pair_judge_cache"

        def resolve_pairs_detail(self, pairs, traj_by_id, *, max_workers=8):
            return PairResolution(resolved=list(pairs))

    ids = _ids(3)
    sel = judge_selection_pairs(
        build_selection_pairs(ids, n_pairs=3, seed=0),
        {tid: _traj(tid) for tid in ids},
        _NoModel(),
    )
    assert sel.provenance["judge_model"] == "data/pair_judge_cache"


# --- persistence ------------------------------------------------------------------


def test_save_and_load_round_trip_preserves_order_and_provenance(tmp_path):
    sel = SelectionSet(
        row_ids=["r3", "r1", "r2"],
        pairs=[("r3", "r1"), ("r2", "r3")],
        tied=[("r1", "r2")],
        unresolved=[],
        provenance={"seed": 4, "stratified": True, "judge_model": "m", "n_hard": 1},
    )
    written = sel.save(tmp_path)
    assert written == tmp_path / SELECTION_SET_FILE

    loaded = SelectionSet.load(tmp_path)
    assert loaded == sel
    assert loaded.pairs == [("r3", "r1"), ("r2", "r3")]
    assert all(isinstance(p, tuple) for p in loaded.pairs)

    # Ordinary JSON, ids as-is: the exposure guard greps for them.
    raw = json.loads(written.read_text(encoding="utf-8"))
    assert raw["row_ids"] == ["r3", "r1", "r2"]
    assert raw["pairs"] == [["r3", "r1"], ["r2", "r3"]]
    assert raw["provenance"]["seed"] == 4

    # An explicit file path works too.
    explicit = sel.save(tmp_path / "custom.json")
    assert SelectionSet.load(explicit) == sel


# --- concordance ------------------------------------------------------------------


def _three_pair_set() -> SelectionSet:
    return SelectionSet(row_ids=["a", "b", "c", "d"], pairs=[("a", "b"), ("b", "c"), ("c", "d")])


def test_concordance_of_perfect_inverted_and_partial_scores():
    sel = _three_pair_set()

    perfect = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    inverted = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    assert sel.concordance(perfect) == 1.0
    assert sel.concordance(inverted) == 0.0

    # `d` unscored: the (c, d) pair is charged, not skipped.
    partial = {"a": 4.0, "b": 3.0, "c": 2.0}
    assert sel.concordance(partial) == pytest.approx(2 / 3)
    detail = sel.concordance_detail(partial)
    assert (detail.n_offered, detail.n_scored, detail.missing_ids) == (3, 2, 1)

    # The candidate's own tie is half credit in concordance ...
    tied = {"a": 4.0, "b": 3.0, "c": 3.0, "d": 1.0}
    assert sel.concordance(tied) == pytest.approx(2.5 / 3)
    # ... and wrong in the correctness vector the paired test consumes.
    assert sel.correctness(tied) == [True, False, True]
    assert sel.correctness(partial) == [True, True, False]
    assert sel.correctness(perfect) == [True, True, True]
    assert sel.correctness({}) == [False, False, False]
    assert len(sel) == 3


# --- scoring ------------------------------------------------------------------


class _FakeSolver:
    def __init__(self, scores: dict[str, float | None], failed: set[str] = frozenset()):
        self.scores = scores
        self.failed = set(failed)
        self.seen: list[list[str]] = []

    def run(self, loaded_filter, trajectories):
        self.seen.append([t.trajectory_id for t in trajectories])
        return [
            FilterDecision(
                trajectory_id=t.trajectory_id,
                decision=Decision.REVIEW,
                rank_score=self.scores.get(t.trajectory_id),
                scorer_failed=t.trajectory_id in self.failed,
            )
            for t in trajectories
        ]


def test_score_on_selection_drops_failed_and_unscored_rows():
    ids = _ids(4)
    rows = [_traj(tid) for tid in ids]
    solver = _FakeSolver(
        {ids[0]: 3, ids[1]: 2.5, ids[2]: 9.0, ids[3]: None},
        failed={ids[2]},
    )
    sentinel_filter = object()

    scores = score_on_selection(sentinel_filter, solver, rows)

    assert solver.seen == [ids]
    assert scores == {ids[0]: 3.0, ids[1]: 2.5}
    assert all(isinstance(v, float) for v in scores.values())


# --- guards ------------------------------------------------------------------


def test_assert_disjoint_names_the_overlapping_ids():
    sel = ["s1", "s2", "s3", "s4"]
    assert_disjoint(sel, train_ids=["t1", "t2"], probe_ids=["p1"])

    with pytest.raises(ValueError) as exc:
        assert_disjoint(sel, train_ids=["t1", "s2"], probe_ids=["s4", "s3", "p1"])
    message = str(exc.value)
    assert "s2" in message and "train" in message
    assert "s3" in message and "s4" in message and "probe" in message
    assert "s1" not in message


def test_selection_row_ids_in_text_finds_leaks_and_only_leaks():
    ids = _ids(5)
    prompt = f"Here is some evidence about {ids[2]} and also {ids[4]}. Nothing else."
    assert selection_row_ids_in_text(prompt, ids) == sorted([ids[2], ids[4]])
    assert selection_row_ids_in_text("clean prompt", ids) == []
    assert selection_row_ids_in_text("", ids) == []
    assert selection_row_ids_in_text(prompt, []) == []
