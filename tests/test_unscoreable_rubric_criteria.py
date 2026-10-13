"""A candidate the solver cannot score is refused at the precheck, not lost later.

`Solver.run` scores exactly the names in `policy.py`'s `RUBRIC_CRITERIA`.
With none, `decide()` is handed an empty mapping and returns one constant for
every row, so the candidate orders nothing. Refused only at scoring time, as
a WARNING and a dropped candidate, `precheck.json` would still call it
`"status": "ok"` inside an empty `gated_out` and `summary.csv` would show
nothing but a smaller `n_candidates` — and a revision leg that rewrites
policy.py without the declaration can lose most of an iteration's candidates
that way, with the proposer never told, so it writes the same broken shape
again.

So the check sits in `constraints.check_candidate`, where it
fires `invalid` like any other breach: `gate_candidates` drops the candidate,
`precheck.json` names it under `gated_out` with the reason, and
`render_precheck_block` puts the reason in front of the proposer between its
two legs. The guard in `_evaluate_candidate_inner` stays as a backstop,
loud and recorded in the iteration record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_filter.core.filter_loader import load_filter, materialize_candidate
from meta_filter.gate import constraints as cx
from meta_filter.gate.precheck import render_precheck_block, run_precheck
from tests.selection_gate_fixture import (
    BASELINE_POLICY,
    RUBRIC,
    RULES,
    TRUTH_POLICY,
    MetaScorer,
    build_optimizer,
    prepare,
)

DECIDE = """

def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': scores.get('Truth', 0.0), 'confidence': 0.5}
"""

# What the nine real candidates look like: policy.py rewritten with its
# weights and its threshold intact and the declaration simply gone.
NO_CRITERIA_POLICY = "RUBRIC_WEIGHTS = {'Truth': 1.0}\nACCEPT_SCORE = 65.0" + DECIDE
EMPTY_CRITERIA_POLICY = "RUBRIC_CRITERIA = []" + DECIDE
DICT_CRITERIA_POLICY = "RUBRIC_CRITERIA = {'Truth': 1.0}" + DECIDE
NUMBER_CRITERIA_POLICY = "RUBRIC_CRITERIA = ['Truth', 2]" + DECIDE
BLANK_CRITERIA_POLICY = "RUBRIC_CRITERIA = ['Truth', '   ']" + DECIDE
# A real shape: the names come from the weights dict at import time, and no
# source-level reader can resolve them.
DERIVED_CRITERIA_POLICY = "WEIGHTS = {'Truth': 1.0}\nRUBRIC_CRITERIA = list(WEIGHTS)" + DECIDE
DERIVED_EMPTY_POLICY = "WEIGHTS = {}\nRUBRIC_CRITERIA = list(WEIGHTS)" + DECIDE


def _filter(root: Path, name: str, policy: str) -> Path:
    path = root / name
    materialize_candidate(path, rubric_md=RUBRIC, rules_py=RULES, policy_py=policy)
    return path


def _check(candidate: Path, initial: Path) -> cx.ConstraintReport:
    return cx.check_candidate(candidate, baseline_dir=initial, initial_dir=initial)


# --- the source-level check -------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        (NO_CRITERIA_POLICY, cx.NO_CRITERIA_REASON),
        (EMPTY_CRITERIA_POLICY, cx.EMPTY_CRITERIA_REASON),
        (NUMBER_CRITERIA_POLICY, cx.BAD_CRITERIA_REASON),
    ],
    ids=["missing", "empty", "not-strings"],
)
def test_an_unscoreable_declaration_is_invalid_and_the_reason_names_the_file(
    tmp_path, policy, reason
):
    initial = _filter(tmp_path, "initial", TRUTH_POLICY)
    candidate = _filter(tmp_path, "cand", policy)

    assert cx.rubric_criteria_defect(candidate) == reason
    report = _check(candidate, initial)
    assert report.status == "invalid"
    assert reason in report.reasons
    assert reason.startswith("policy.py ")
    assert "the solver has nothing to score" in reason


def test_a_healthy_candidate_fires_nothing(tmp_path):
    initial = _filter(tmp_path, "initial", TRUTH_POLICY)
    candidate = _filter(tmp_path, "cand", BASELINE_POLICY)

    assert cx.rubric_criteria_defect(candidate) is None
    assert _check(candidate, initial).status == "ok"


def test_a_declaration_the_source_cannot_evaluate_is_left_to_the_value(tmp_path):
    """The false positive this check must not have.

    `RUBRIC_CRITERIA = list(WEIGHTS)` is a working declaration that no AST
    reader can resolve. Firing on it would gate out two of the filters this
    repository ships, so the source-level check says nothing and the value
    read off the imported module decides — which catches the empty case the
    source cannot see.
    """

    initial = _filter(tmp_path, "initial", TRUTH_POLICY)
    good = _filter(tmp_path, "derived", DERIVED_CRITERIA_POLICY)
    empty = _filter(tmp_path, "derived_empty", DERIVED_EMPTY_POLICY)

    assert cx.rubric_criteria_defect(good) is None
    assert cx.rubric_criteria_defect(empty) is None, "the source cannot know"
    assert _check(good, initial).status == "ok"
    assert _check(empty, initial).status == "ok"

    assert cx.rubric_criteria_defect_of(load_filter(good).policy_module.RUBRIC_CRITERIA) is None
    assert (
        cx.rubric_criteria_defect_of(load_filter(empty).policy_module.RUBRIC_CRITERIA)
        == cx.EMPTY_CRITERIA_REASON
    )


def test_a_name_bound_outside_the_module_body_is_not_an_absence(tmp_path):
    candidate = _filter(
        tmp_path,
        "guarded",
        "import os\n\nif os.environ.get('X'):\n    RUBRIC_CRITERIA = ['Truth']\n"
        "else:\n    RUBRIC_CRITERIA = ['Noisy']" + DECIDE,
    )
    assert cx.rubric_criteria_defect(candidate) is None


def test_every_shipped_filter_stays_clean():
    """The check has no false positive on the shipped seed filters."""

    filters = Path(__file__).resolve().parent.parent / "filters"
    fired = [
        str(p.parent.relative_to(filters))
        for p in sorted(filters.rglob("policy.py"))
        if cx.rubric_criteria_defect(p.parent) is not None
    ]
    assert fired == []


# --- the value-level check --------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, cx.NO_CRITERIA_REASON),
        ([], cx.EMPTY_CRITERIA_REASON),
        ((), cx.EMPTY_CRITERIA_REASON),
        ({"Truth": 1.0}, cx.BAD_CRITERIA_REASON),
        ("Truth", cx.BAD_CRITERIA_REASON),
        (["Truth", 2], cx.BAD_CRITERIA_REASON),
        (["Truth", "  "], cx.BAD_CRITERIA_REASON),
        (["Truth"], None),
        (("Truth", "Noisy"), None),
    ],
)
def test_the_value_reader_agrees_with_the_source_reader(value, reason):
    assert cx.rubric_criteria_defect_of(value) == reason


def test_recording_a_defect_on_a_built_report_is_idempotent(tmp_path):
    initial = _filter(tmp_path, "initial", TRUTH_POLICY)
    report = _check(_filter(tmp_path, "cand", BASELINE_POLICY), initial)
    assert report.status == "ok"

    cx.record_criteria_defect(report, cx.EMPTY_CRITERIA_REASON)
    cx.record_criteria_defect(report, cx.EMPTY_CRITERIA_REASON)
    assert report.status == "invalid"
    assert report.reasons.count(cx.EMPTY_CRITERIA_REASON) == 1


# --- through the precheck and into the prompt -------------------------------------


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        (NO_CRITERIA_POLICY, cx.NO_CRITERIA_REASON),
        (EMPTY_CRITERIA_POLICY, cx.EMPTY_CRITERIA_REASON),
        (DICT_CRITERIA_POLICY, cx.BAD_CRITERIA_REASON),
        (BLANK_CRITERIA_POLICY, cx.BAD_CRITERIA_REASON),
        (DERIVED_EMPTY_POLICY, cx.EMPTY_CRITERIA_REASON),
    ],
    ids=["missing", "empty", "a-dict", "a-blank-name", "derived-empty"],
)
def test_the_precheck_fires_invalid_and_the_revision_block_shows_the_reason(
    tmp_path, policy, reason
):
    initial = _filter(tmp_path, "initial", TRUTH_POLICY)
    candidate = _filter(tmp_path / "cands", "iter001_00_broken", policy)

    results = run_precheck(
        [candidate],
        parent_filter=load_filter(initial),
        trajectories_by_id={},
        truth_pairs=[],
        scorer=MetaScorer(),
        initial_dir=initial,
    )
    result = results["iter001_00_broken"]
    assert result.criteria_defect == reason
    assert result.error == reason
    assert result.constraint is not None and result.constraint.status == "invalid"
    assert reason in result.constraint.reasons
    assert json.loads((candidate / "precheck.json").read_text("utf-8"))["criteria_defect"] == reason

    block = render_precheck_block(results)
    assert "### iter001_00_broken — NOT SCOREABLE" in block
    assert f"constraint: INVALID — {reason}" in block


def test_a_healthy_candidate_is_untouched_by_the_precheck(tmp_path):
    initial = _filter(tmp_path, "initial", TRUTH_POLICY)
    candidate = _filter(tmp_path / "cands", "iter001_00_fine", BASELINE_POLICY)

    result = run_precheck(
        [candidate],
        parent_filter=load_filter(initial),
        trajectories_by_id={},
        truth_pairs=[],
        scorer=MetaScorer(),
        initial_dir=initial,
    )["iter001_00_fine"]
    assert result.criteria_defect == ""
    assert result.constraint is not None and result.constraint.status == "ok"
    assert "INVALID" not in render_precheck_block({"iter001_00_fine": result})


# --- end to end ------------------------------------------------------------------

# Iteration 0 disputes every pair so that iteration 1 has judged pairs to
# precheck against, and loses, so iteration 1 still edits the initial filter.
LOSING_FIRST = [("inverted", BASELINE_POLICY.replace("scores['Noisy']", "-scores['Noisy']"))]
BROKEN_SIBLINGS = [
    ("truth", TRUTH_POLICY),
    ("no_criteria", NO_CRITERIA_POLICY),
    ("empty_criteria", EMPTY_CRITERIA_POLICY),
]


def _iteration(run_dir: Path, k: int, name: str) -> dict:
    return json.loads((run_dir / "iterations" / f"{k:03d}" / name).read_text("utf-8"))


@pytest.fixture(scope="module")
def gated(tmp_path_factory):
    """Three candidates proposed in iteration 1, two of them unscoreable.

    Returns the run directory and the proposer, because half of what this
    fix is for is what the proposer was told.
    """

    run = prepare(tmp_path_factory.mktemp("unscoreable"))
    optimizer = build_optimizer(
        run,
        script={0: LOSING_FIRST, 1: BROKEN_SIBLINGS},
        iterations=2,
        run_id="unscoreable",
    )
    optimizer.config.candidates_per_iteration = len(BROKEN_SIBLINGS)
    return optimizer.run(), optimizer.proposer


@pytest.fixture(scope="module")
def gated_run(gated) -> Path:
    return gated[0]


def test_the_revision_leg_is_told_which_file_to_repair(gated):
    """What the nine real candidates never got: the reason, before leg 2.

    The block is handed to `Proposer.revise` between the two legs, so a
    proposer that dropped the declaration can put it back in the same
    iteration instead of writing the same shape again next time.
    """

    block = next(
        json.loads(p)["precheck_block"] for p in gated[1].prompts if '"leg": "revise"' in p
    )
    assert "### iter001_01_no_criteria — NOT SCOREABLE" in block
    assert f"constraint: INVALID — {cx.NO_CRITERIA_REASON}" in block
    assert f"constraint: INVALID — {cx.EMPTY_CRITERIA_REASON}" in block
    assert "### iter001_00_truth — score" in block


def test_the_broken_siblings_are_gated_and_the_healthy_one_is_scored(gated_run):
    precheck = _iteration(gated_run, 1, "precheck.json")

    assert precheck["scored"] == ["iter001_00_truth"]
    assert set(precheck["gated_out"]) == {"iter001_01_no_criteria", "iter001_02_empty_criteria"}
    assert (
        precheck["gated_out"]["iter001_01_no_criteria"]["reason"]
        == f"constraint invalid: {cx.NO_CRITERIA_REASON}"
    )
    assert (
        precheck["gated_out"]["iter001_02_empty_criteria"]["reason"]
        == f"constraint invalid: {cx.EMPTY_CRITERIA_REASON}"
    )
    assert precheck["constraints"]["iter001_00_truth"]["status"] == "ok"


def test_the_gated_siblings_never_reach_the_solver_but_stay_on_disk(gated_run):
    solved = {p.stem for p in (gated_run / "iterations" / "001" / "solver_outputs").iterdir()}
    assert "iter001_00_truth" in solved
    assert not solved & {"iter001_01_no_criteria", "iter001_02_empty_criteria"}

    validation = {e["candidate_id"]: e for e in _iteration(gated_run, 1, "validation.json")}
    for cid in ("iter001_01_no_criteria", "iter001_02_empty_criteria"):
        assert cid in validation and validation[cid]["invalid_reason"] is not None
        assert (gated_run / "iterations" / "001" / "candidates" / cid / "policy.py").exists()


def test_nothing_vanishes_between_the_gate_and_the_scoring_pass(gated_run):
    for k in (0, 1):
        assert _iteration(gated_run, k, "record.json")["discarded_at_scoring"] == {}
