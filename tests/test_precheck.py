"""The candidate precheck: what a candidate scores before the loop pays.

The measurement only earns its place if two things hold. It has to be nearly
free — which means a criterion the candidate did not touch must come out of
the scorer cache rather than off a fresh call — and it has to be the SAME
number selection will apply, or the proposer is being steered by a proxy.
Both are asserted here against a real `LLMRubricScorer` with a counting
client, so "reused" means an LLM call that did not happen.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from meta_filter.core.filter_loader import load_filter
from meta_filter.core.llm_scorer import LLMRubricScorer
from meta_filter.core.schemas import Trajectory, Turn
from meta_filter.gate import precheck as pc
from meta_filter.llm.base import LLMResponse

ROOT = Path(__file__).resolve().parents[1]
INITIAL_RANKING = ROOT / "tests" / "fixtures" / "filters" / "six_criteria"


# --- changed-criterion detection, on rubric text alone ----------------------

HEADER = "# Rubric\n\nScale is +2 / -2. Doubt is 0.\n\n"
SECTION_A = "## Criterion: Alpha\n\nDoes it ground the task?\n\n+2: yes\n-2: no\n"
SECTION_B = "## Criterion: Beta\n\nDoes it verify?\n\n+2: yes\n-2: no\n"


def _rubric(header: str = HEADER, *sections: str) -> str:
    return header + "\n".join(sections or (SECTION_A, SECTION_B))


def test_editing_one_criterion_leaves_the_others_unchanged():
    delta = pc.changed_criteria(
        _rubric(),
        ["Alpha", "Beta"],
        _rubric(HEADER, SECTION_A, SECTION_B.replace("Does it verify?", "Does it TEST?")),
        ["Alpha", "Beta"],
    )

    assert delta.changed == ("Beta",)
    assert delta.unchanged == ("Alpha",)
    assert delta.added == ()
    assert delta.removed == ()
    assert delta.header_changed is False


def test_editing_the_preamble_changes_every_criterion():
    """The preamble is in every criterion's prompt, so it is real re-scoring.

    Calling these criteria "unchanged" would make the precheck understate its
    own cost precisely where the proposer most needs the warning: a one-line
    edit to the scale definition re-scores the whole pool.
    """

    delta = pc.changed_criteria(
        _rubric(),
        ["Alpha", "Beta"],
        _rubric("# Rubric\n\nScale is +3 / -3.\n\n"),
        ["Alpha", "Beta"],
    )

    assert set(delta.changed) == {"Alpha", "Beta"}
    assert delta.unchanged == ()
    assert delta.header_changed is True


def test_added_and_removed_criteria_are_named_separately():
    delta = pc.changed_criteria(
        _rubric(),
        ["Alpha", "Beta"],
        _rubric(HEADER, SECTION_A, "## Criterion: Gamma\n\nNew axis.\n\n+2: yes\n"),
        ["Alpha", "Gamma"],
    )

    assert delta.added == ("Gamma",)
    assert delta.removed == ("Beta",)
    assert delta.unchanged == ("Alpha",)
    assert "Gamma" in delta.changed


def test_widening_a_criterions_declared_fields_changes_only_that_criterion():
    """A `reads:` declaration is part of the cache key, not decoration."""

    widened = SECTION_B.replace(
        "## Criterion: Beta\n", "## Criterion: Beta\n<!-- reads: gold_patch -->\n"
    )
    delta = pc.changed_criteria(
        _rubric(), ["Alpha", "Beta"], _rubric(HEADER, SECTION_A, widened), ["Alpha", "Beta"]
    )

    assert delta.changed == ("Beta",)
    assert delta.unchanged == ("Alpha",)


# --- the cache census, against a real scorer --------------------------------


class _CountingScorerClient:
    """A scorer that answers whatever criterion the prompt asks for.

    The score is a pure function of (trajectory id, criterion), so a cached
    cell and a fresh one are indistinguishable by value — the only way the
    tests below can tell them apart is `calls`, which is the point.
    """

    model = "stub-scorer"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, *, system=None, temperature=0.0, cache_system=False):
        self.calls += 1
        criterion = prompt.split("Criteria to score (in order): ", 1)[1].split("\n", 1)[0].strip()
        tid = prompt.split("Task:\n", 1)[1].split("\n", 1)[0].strip().removeprefix("task for ")
        value = (sum(map(ord, tid + criterion)) % 5) - 2
        return LLMResponse(
            text=json.dumps(
                {"scores": {criterion: {"score": value, "evidence": "e"}}, "summary": "s"}
            ),
            prompt_tokens=10,
            completion_tokens=5,
            model=self.model,
        )


def _traj(tid: str) -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task=f"task for {tid}",
        turns=[
            Turn(role="agent", action="grep", content=f"look at {tid}"),
            Turn(role="agent", action="apply_patch", content="--- a/x\n+++ b/x\n+fix\n" * 20),
            Turn(role="agent", action="pytest", content="2 passed"),
        ],
        metadata={"instance_id": tid},
    )


def _copy_filter(dest: Path) -> Path:
    shutil.copytree(INITIAL_RANKING, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def _edit_one_criterion(filter_dir: Path, criterion: str, marker: str) -> None:
    """Rewrite one criterion's body, leaving the preamble and siblings alone."""

    rubric = (filter_dir / "rubric.md").read_text(encoding="utf-8")
    head, _, tail = rubric.partition(f"## Criterion: {criterion}\n")
    assert tail, f"{criterion} is not a section of this rubric"
    body, sep, rest = tail.partition("\n## ")
    (filter_dir / "rubric.md").write_text(
        f"{head}## Criterion: {criterion}\n{body}\n{marker}\n{sep}{rest}", encoding="utf-8"
    )


@pytest.fixture()
def scorer(tmp_path):
    return LLMRubricScorer(client=_CountingScorerClient(), cache_dir=tmp_path / "scorer_cache")


@pytest.fixture()
def parent(tmp_path):
    return load_filter(_copy_filter(tmp_path / "parent"))


@pytest.fixture()
def rows():
    return {f"t{i}": _traj(f"t{i}") for i in range(6)}


def _warm(scorer, parent, rows):
    """Pay for the parent's cells once, the way an earlier iteration would."""

    criteria = list(parent.policy_module.RUBRIC_CRITERIA)
    for traj in rows.values():
        scorer.score(traj, criteria, parent.rubric_md)
    return criteria


def test_an_untouched_criterion_is_reused_and_the_edited_one_is_not(tmp_path, scorer, parent, rows):
    criteria = _warm(scorer, parent, rows)
    candidate = _copy_filter(tmp_path / "cand")
    _edit_one_criterion(candidate, "Error Recovery", "-2: also gives up silently.")
    loaded = load_filter(candidate)

    reused, missing = pc.cache_census(scorer, list(rows.values()), criteria, loaded.rubric_md)

    assert missing == len(rows), "exactly the edited criterion needs a call, on every row"
    assert reused == len(rows) * (len(criteria) - 1)


def test_the_census_counts_cells_the_scorer_would_actually_call_for(tmp_path, scorer, parent, rows):
    """The census is a prediction; here it is checked against the bill."""

    criteria = _warm(scorer, parent, rows)
    candidate = _copy_filter(tmp_path / "cand")
    _edit_one_criterion(candidate, "Task Grounding", "-2: never reads the issue.")
    loaded = load_filter(candidate)

    _, predicted = pc.cache_census(scorer, list(rows.values()), criteria, loaded.rubric_md)
    before = scorer.client.calls
    for traj in rows.values():
        scorer.score(traj, criteria, loaded.rubric_md)

    assert scorer.client.calls - before == predicted


def test_a_rewritten_preamble_costs_every_cell(tmp_path, scorer, parent, rows):
    criteria = _warm(scorer, parent, rows)
    candidate = _copy_filter(tmp_path / "cand")
    rubric = (candidate / "rubric.md").read_text(encoding="utf-8")
    (candidate / "rubric.md").write_text(
        rubric.replace("symmetric +2 / +1 / 0 / -1 / -2 scale", "0-10 scale"), encoding="utf-8"
    )
    loaded = load_filter(candidate)

    reused, missing = pc.cache_census(scorer, list(rows.values()), criteria, loaded.rubric_md)

    assert reused == 0
    assert missing == len(rows) * len(criteria)


def test_a_scorer_without_a_cache_reports_no_census(parent, rows):
    from meta_filter.core.solver import HeuristicRubricScorer

    assert pc.cache_census(
        HeuristicRubricScorer(),
        list(rows.values()),
        list(parent.policy_module.RUBRIC_CRITERIA),
        parent.rubric_md,
    ) == (0, 0)


# --- the concordance arithmetic ---------------------------------------------


class _FixedScorer:
    """Scores every criterion at a value the trajectory id dictates.

    Lets a test state the ordering it wants rather than discovering one: the
    policy is monotone in the rubric mean, so a higher fixture value is a
    higher `rank_score`.
    """

    def __init__(self, by_id: dict[str, float]) -> None:
        self.by_id = by_id

    def score(self, trajectory, criteria, rubric_md=""):
        from meta_filter.core.solver import ScorerCost

        value = self.by_id[trajectory.trajectory_id]
        return {c: value for c in criteria}, [], ScorerCost(llm_calls=0, llm_tokens=0)


def _precheck(tmp_path, parent, rows, values, pairs, **kwargs):
    candidate = _copy_filter(tmp_path / "cand")
    return pc.precheck_candidate(
        candidate,
        parent_filter=parent,
        trajectories_by_id=rows,
        truth_pairs=pairs,
        scorer=_FixedScorer(values),
        lazy_scoring=False,
        **kwargs,
    )


def test_concordance_is_the_share_of_pairs_ordered_the_judges_way(tmp_path, parent):
    rows = {tid: _traj(tid) for tid in ("a", "b", "c")}
    # a > b > c by construction; the judge says a > b, b > c and c > a.
    result = _precheck(
        tmp_path,
        parent,
        rows,
        {"a": 2.0, "b": 1.0, "c": 0.0},
        [("a", "b"), ("b", "c"), ("c", "a")],
    )

    assert result.n_pairs == 3
    assert result.concordance == pytest.approx(2 / 3)
    assert [m["hi"] for m in result.misordered_pairs] == ["c"]
    assert result.misordered_pairs[0]["direction"] == "inverted"


def test_a_judge_tie_leaves_both_the_numerator_and_the_denominator(tmp_path, parent):
    """Half credit would score a candidate on a pair no verdict exists for."""

    rows = {tid: _traj(tid) for tid in ("a", "b", "c")}
    result = _precheck(
        tmp_path,
        parent,
        rows,
        {"a": 2.0, "b": 1.0, "c": 0.0},
        [("a", "b"), ("c", "a")],
        tied_pairs=[("c", "a")],
    )

    assert result.n_pairs == 1, "the tied pair left the exam"
    assert result.n_tied == 1
    assert result.concordance == pytest.approx(1.0)
    assert result.misordered_pairs == [], "a tied pair is not a mistake to report"


def test_the_candidates_own_tie_is_half_credit_and_is_still_reported(tmp_path, parent):
    rows = {tid: _traj(tid) for tid in ("a", "b")}
    result = _precheck(tmp_path, parent, rows, {"a": 1.0, "b": 1.0}, [("a", "b")])

    assert result.concordance == pytest.approx(0.5)
    assert result.misordered_pairs[0]["direction"] == "tied"


def test_the_score_is_concordance_minus_complexity(tmp_path, parent):
    """The precheck must not invent an objective — this is `compute_score`."""

    rows = {tid: _traj(tid) for tid in ("a", "b")}
    result = _precheck(tmp_path, parent, rows, {"a": 2.0, "b": 0.0}, [("a", "b")])

    expected = result.concordance - 0.15 * min(1.0, result.complexity / 120.0)
    assert result.score == pytest.approx(expected)
    assert result.redundancy_penalty == 0.0, "every criterion holds the same value here"


def test_a_live_weight_the_precheck_cannot_measure_is_named(tmp_path, parent):
    rows = {tid: _traj(tid) for tid in ("a", "b")}
    result = _precheck(
        tmp_path,
        parent,
        rows,
        {"a": 2.0, "b": 0.0},
        [("a", "b")],
        score_weights={"lambda_review": 0.3},
    )

    assert result.omitted_terms == ["review_rate"]


# --- the artifact -----------------------------------------------------------


def test_precheck_json_is_written_for_every_candidate(tmp_path, parent):
    rows = {tid: _traj(tid) for tid in ("a", "b")}
    cands = [_copy_filter(tmp_path / "c0"), _copy_filter(tmp_path / "c1")]

    results = pc.run_precheck(
        cands,
        parent_filter=parent,
        trajectories_by_id=rows,
        truth_pairs=[("a", "b")],
        scorer=_FixedScorer({"a": 2.0, "b": 0.0}),
        lazy_scoring=False,
        initial_dir=parent.directory,
    )

    assert set(results) == {"c0", "c1"}
    for cand in cands:
        payload = json.loads((cand / "precheck.json").read_text(encoding="utf-8"))
        assert set(payload) >= {
            "concordance",
            "n_pairs",
            "n_tied",
            "redundancy_penalty",
            "complexity",
            "score",
            "cells_rescored",
            "cells_reused",
            "cost_usd",
            "misordered_pairs",
        }
        assert payload["candidate_id"] == cand.name
        assert payload["error"] == ""


def test_a_candidate_that_does_not_import_is_reported_not_raised(tmp_path, parent):
    broken = _copy_filter(tmp_path / "broken")
    (broken / "policy.py").write_text("this is not python(\n", encoding="utf-8")

    result = pc.precheck_candidate(
        broken,
        parent_filter=parent,
        trajectories_by_id={"a": _traj("a")},
        truth_pairs=[],
        scorer=_FixedScorer({}),
    )

    assert result.error
    assert result.score == 0.0
    assert json.loads(json.dumps(result.to_dict()))["error"] == result.error


def test_a_candidate_with_no_rubric_criteria_is_caught_before_any_scoring(tmp_path, parent):
    """The same contract the optimizer enforces, said one leg earlier."""

    empty = _copy_filter(tmp_path / "empty")
    source = (empty / "policy.py").read_text(encoding="utf-8")
    start = source.index("RUBRIC_CRITERIA = [")
    end = source.index("]", start) + 1
    (empty / "policy.py").write_text(
        source[:start] + "RUBRIC_CRITERIA = []" + source[end:], encoding="utf-8"
    )

    result = pc.precheck_candidate(
        empty,
        parent_filter=parent,
        trajectories_by_id={"a": _traj("a")},
        truth_pairs=[("a", "a")],
        scorer=_FixedScorer({}),
    )

    assert "RUBRIC_CRITERIA" in result.error


def test_no_judged_pair_in_the_pool_is_stated_rather_than_scored_as_zero(tmp_path, parent):
    result = pc.precheck_candidate(
        _copy_filter(tmp_path / "cand"),
        parent_filter=parent,
        trajectories_by_id={"a": _traj("a")},
        truth_pairs=[("x", "y")],
        scorer=_FixedScorer({}),
    )

    assert "no already-judged pair" in result.error
    assert result.n_pairs == 0


def test_the_revision_prompt_says_what_the_precheck_did_and_did_not_do(tmp_path, parent):
    rows = {tid: _traj(tid) for tid in ("a", "b")}
    results = pc.run_precheck(
        [_copy_filter(tmp_path / "c0")],
        parent_filter=parent,
        trajectories_by_id=rows,
        truth_pairs=[("b", "a")],
        scorer=_FixedScorer({"a": 2.0, "b": 0.0}),
        lazy_scoring=False,
        write=False,
        initial_dir=parent.directory,
    )
    block = pc.render_precheck_block(results)

    assert "no judge was called" in block
    assert "not the whole training pool" in block
    assert "was not" in block and "re-scored" in block
    assert "c0" in block
    assert "inverted: b" in block


def test_the_precheck_bills_its_own_scoring_to_the_run_ledger(tmp_path, parent, rows):
    """`cost_usd` is a ledger delta, not an estimate over token counts.

    `FilterDecision` carries `llm_tokens` as one number with no prompt /
    completion split, so pricing it here would have to guess a ratio. The
    run already meters every client call by role, and the precheck runs
    between the proposer's two legs with nothing else in flight, so the
    scorer role's movement across it IS the precheck's bill.
    """

    from meta_filter.llm.cost_meter import CostLedger, metered

    ledger = CostLedger()
    scorer = LLMRubricScorer(
        client=metered(_CountingScorerClient(), "solver", ledger),
        cache_dir=tmp_path / "cache",
    )
    result = pc.precheck_candidate(
        _copy_filter(tmp_path / "cand"),
        parent_filter=parent,
        trajectories_by_id=rows,
        truth_pairs=[("t0", "t1"), ("t2", "t3")],
        scorer=scorer,
        lazy_scoring=False,
        cost_ledger=ledger,
    )

    # The stub reports no price, so the honest reading of its 0.0 is a
    # hole, not a free call. Four rows lie in those two pairs and the rubric
    # has six criteria, and lazy scoring is off, so every cell is a call.
    assert result.cost_source == "unpriced"
    assert result.llm_calls == 4 * len(parent.policy_module.RUBRIC_CRITERIA)
    assert ledger.snapshot()["solver"].calls == result.llm_calls


class _PrefetchingScorer(_FixedScorer):
    """A scorer with the LLM scorer's parallel warm-up entry point."""

    def __init__(self, by_id):
        super().__init__(by_id)
        self.prefetch_calls: list[tuple[int, tuple[str, ...]]] = []

    def prefetch_parallel(self, trajectories, criteria, rubric_md):
        self.prefetch_calls.append((len(trajectories), tuple(criteria)))
        return len(trajectories)


def test_the_precheck_warms_the_cache_in_parallel_before_scoring(tmp_path, parent):
    """Solver.run scores one row at a time; a rewritten rubric on the 50-row
    pool took ~2 s per cell serially, so the precheck must go through the
    scorer's parallel prefetch first, over exactly the rows the truth pairs
    name and the candidate's own criteria."""

    rows = {tid: _traj(tid) for tid in ("a", "b", "c", "d")}
    scorer = _PrefetchingScorer({"a": 2.0, "b": 1.0, "c": 0.0, "d": -1.0})
    candidate = _copy_filter(tmp_path / "cand")
    result = pc.precheck_candidate(
        candidate,
        parent_filter=parent,
        trajectories_by_id=rows,
        truth_pairs=[("a", "b"), ("b", "c")],  # d is not in any pair
        scorer=scorer,
        lazy_scoring=False,
    )

    assert not result.error
    assert len(scorer.prefetch_calls) == 1
    n_rows, criteria = scorer.prefetch_calls[0]
    assert n_rows == 3
    assert criteria and all(isinstance(c, str) for c in criteria)


def test_a_vanished_candidate_dir_does_not_crash_the_precheck(tmp_path, parent):
    """A Codex agent can `rm -rf` the baseline copy between writing its
    candidates and the precheck; writing precheck.json into the missing
    directory would raise FileNotFoundError and kill the run."""

    rows = {tid: _traj(tid) for tid in ("a", "b")}
    gone = tmp_path / "iter001_gone"
    assert not gone.exists()
    results = pc.run_precheck(
        [gone],
        parent_filter=parent,
        trajectories_by_id=rows,
        truth_pairs=[("a", "b")],
        scorer=_FixedScorer({"a": 1.0, "b": 0.0}),
        lazy_scoring=False,
        initial_dir=parent.directory,
    )
    assert results[gone.name].error
    assert not gone.exists()  # not recreated just to hold a readout


def test_the_optimizer_rebuilds_a_deleted_baseline_before_the_precheck(tmp_path, parent):
    from meta_filter.loop.optimizer import Optimizer

    baseline_dir = tmp_path / "iter001_baseline"
    assert not baseline_dir.exists()
    Optimizer._ensure_baseline_dir(baseline_dir, parent, iteration=1)
    assert (baseline_dir / "rubric.md").exists()
    before = sorted(p.name for p in baseline_dir.iterdir())
    Optimizer._ensure_baseline_dir(baseline_dir, parent, iteration=1)  # idempotent
    assert sorted(p.name for p in baseline_dir.iterdir()) == before


# --- the growth caps, threaded through the precheck ---------------------------


def _grow_rubric(filter_dir: Path, n_new: int) -> None:
    """Append `n_new` criteria to the rubric, past the +2 soft cap when n_new > 2."""

    rubric = (filter_dir / "rubric.md").read_text(encoding="utf-8")
    for i in range(n_new):
        rubric += f"\n## Criterion: Extra {i}\n\nA new axis.\n\n+2: yes\n-2: no\n"
    (filter_dir / "rubric.md").write_text(rubric, encoding="utf-8")


def test_an_initial_dir_makes_the_precheck_report_the_constraints(tmp_path, parent):
    """The parent is the per-iteration baseline; `initial_dir` the run cap."""

    rows = {tid: _traj(tid) for tid in ("a", "b")}
    initial = _copy_filter(tmp_path / "initial")
    grown = _copy_filter(tmp_path / "grown")
    _grow_rubric(grown, 3)
    same = _copy_filter(tmp_path / "same")

    results = pc.run_precheck(
        [grown, same],
        parent_filter=parent,
        trajectories_by_id=rows,
        truth_pairs=[("a", "b")],
        scorer=_FixedScorer({"a": 2.0, "b": 0.0}),
        lazy_scoring=False,
        initial_dir=initial,
    )

    grown_report = results["grown"].constraint
    assert grown_report is not None
    assert grown_report.status == "needs_strong_evidence"
    n = len(parent.policy_module.RUBRIC_CRITERIA)
    assert f"rubric_criteria {n + 3} > baseline {n} + 2 (soft)" in grown_report.reasons
    assert grown_report.added == ["Extra 0", "Extra 1", "Extra 2"]
    assert results["same"].constraint is not None
    assert results["same"].constraint.status == "ok"

    payload = json.loads((grown / "precheck.json").read_text(encoding="utf-8"))
    assert payload["constraint"]["status"] == "needs_strong_evidence"
    assert payload["constraint"]["baseline"]["rubric_criteria"] == n
    assert payload["constraint"]["initial"]["rubric_criteria"] == n
    assert (
        json.loads((same / "precheck.json").read_text(encoding="utf-8"))["constraint"]["status"]
        == "ok"
    )


def test_the_revision_prompt_names_a_fired_constraint_and_stays_quiet_otherwise(tmp_path, parent):
    rows = {tid: _traj(tid) for tid in ("a", "b")}
    initial = _copy_filter(tmp_path / "initial")
    grown = _copy_filter(tmp_path / "grown")
    _grow_rubric(grown, 3)
    same = _copy_filter(tmp_path / "same")

    results = pc.run_precheck(
        [grown, same],
        parent_filter=parent,
        trajectories_by_id=rows,
        truth_pairs=[("a", "b")],
        scorer=_FixedScorer({"a": 2.0, "b": 0.0}),
        lazy_scoring=False,
        write=False,
        initial_dir=initial,
    )
    block = pc.render_precheck_block(results)

    assert "constraint: needs_strong_evidence — rubric_criteria" in block
    # Exactly one constraint line: the unchanged candidate gets none.
    assert block.count("constraint:") == 1
    grown_section = block.split("### grown")[1].split("### same")[0]
    assert "constraint:" in grown_section


def test_an_invalid_constraint_is_shouted_in_the_revision_prompt(tmp_path, parent):
    rows = {tid: _traj(tid) for tid in ("a", "b")}
    initial = _copy_filter(tmp_path / "initial")
    pruned = _copy_filter(tmp_path / "pruned")
    rubric = (pruned / "rubric.md").read_text(encoding="utf-8")
    # Drop the last two criteria: below the initial - 1 floor.
    head, _, _ = rubric.partition("## Criterion: Verification Completeness")
    (pruned / "rubric.md").write_text(head, encoding="utf-8")

    results = pc.run_precheck(
        [pruned],
        parent_filter=parent,
        trajectories_by_id=rows,
        truth_pairs=[("a", "b")],
        scorer=_FixedScorer({"a": 2.0, "b": 0.0}),
        lazy_scoring=False,
        write=False,
        initial_dir=initial,
    )
    block = pc.render_precheck_block(results)

    report = results["pruned"].constraint
    assert report.status == "invalid"
    assert report.deleted == ["Verification Completeness", "Error Recovery"]
    assert "constraint: INVALID — rubric_criteria 4 < initial 6 - 1 (deletion floor)" in block
    assert "deleted: Verification Completeness, Error Recovery" in block


def test_a_broken_candidate_still_gets_its_constraint_line(tmp_path, parent):
    """The caps read files, so they are known even when the code does not load."""

    initial = _copy_filter(tmp_path / "initial")
    broken = _copy_filter(tmp_path / "broken")
    _grow_rubric(broken, 3)
    (broken / "policy.py").write_text("this is not python(\n", encoding="utf-8")

    results = pc.run_precheck(
        [broken],
        parent_filter=parent,
        trajectories_by_id={"a": _traj("a")},
        truth_pairs=[],
        scorer=_FixedScorer({}),
        write=False,
        initial_dir=initial,
    )
    block = pc.render_precheck_block(results)

    assert results["broken"].error
    assert results["broken"].constraint.status == "needs_strong_evidence"
    assert "NOT SCOREABLE" in block and "constraint: needs_strong_evidence" in block


def test_custom_limits_reach_the_check(tmp_path, parent):
    from meta_filter.gate.constraints import ConstraintLimits

    rows = {tid: _traj(tid) for tid in ("a", "b")}
    initial = _copy_filter(tmp_path / "initial")
    grown = _copy_filter(tmp_path / "grown")
    _grow_rubric(grown, 1)

    results = pc.run_precheck(
        [grown],
        parent_filter=parent,
        trajectories_by_id=rows,
        truth_pairs=[("a", "b")],
        scorer=_FixedScorer({"a": 2.0, "b": 0.0}),
        lazy_scoring=False,
        write=False,
        initial_dir=initial,
        limits=ConstraintLimits(soft_add={"rubric_criteria": 0}),
    )

    assert results["grown"].constraint.status == "needs_strong_evidence"
