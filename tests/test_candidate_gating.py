"""Candidate gating: propose wide, precheck-rank, score few.

The proposer writes `candidates_per_iteration` directories; the precheck
ranks them on the pairs a judge has already adjudicated; constraint-invalid
ones are dropped first, at most one deletion candidate survives, and only the
top `candidates_scored` go through the full solver pass. Everything held back
stays on disk and is named in the iteration's `precheck.json`.

Built on `tests.selection_gate_fixture`: a deterministic scorer that reads
scores off the row's task text, and a judge that orders by `truth`. Iteration 0
proposes an inverted filter so that every pair is disputed and judged — that
is what iteration 1's precheck ranks on — and loses, so iteration 1 still
edits the initial filter.

The second half pins the prompt side of the edit discipline: the
`Hypothesis:` rule, the growth caps printed with the parent's real counts,
and the gating sentence only where gating is on.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from unittest import mock

import pytest

from meta_filter.agentic_proposer import _agentic_instructions, _read_diagnosis
from meta_filter.constraints import ConstraintLimits
from meta_filter.filter_loader import materialize_candidate
from meta_filter.llm.base import LLMResponse
from meta_filter.paired_test import z_threshold_for
from meta_filter.proposer import ProposerResult, _write_rationale_md, render_edit_discipline
from tests.proposer_prompt_fixture import REPORTS, a_filter, stub_metrics
from tests.selection_gate_fixture import (
    BASELINE_POLICY,
    RUBRIC,
    RULES,
    TRUTH_POLICY,
    ScriptedProposer,
    build_optimizer,
    prepare,
)

# --- candidate filters, by precheck rank ---------------------------------------

INVERTED_NOISY_POLICY = """RUBRIC_CRITERIA = ['Noisy']


def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': -scores['Noisy'], 'confidence': 0.5}
"""

INVERTED_TRUTH_POLICY = """RUBRIC_CRITERIA = ['Truth']


def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': -scores['Truth'], 'confidence': 0.5}
"""

# The fixture's TRAIN_ONLY filter in 3 LOC: its 7-LOC original breaches the
# 1.5x hard cap against the initial filter's 3-LOC policy and is (correctly)
# gated as invalid — the caps are tight on a filter this small.
TRAIN_ONLY_COMPACT_POLICY = """RUBRIC_CRITERIA = ['Noisy', 'TrainBoost']
def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': scores['Noisy'] + scores['TrainBoost']}
"""

# Every row the same: every pair a tie, concordance exactly one half.
CONSTANT_POLICY = """RUBRIC_CRITERIA = ['Truth']


def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': 1.0, 'confidence': 0.5}
"""

# The fixture's initial filter has 3 criteria: the hard cap is 1.5 x 3 = 4.5,
# so six criteria breach it (and the +2 soft cap) — `invalid`.
OVERGROWN_RUBRIC = RUBRIC + "".join(
    f"\n## Criterion: Extra{i}\n+2 when something else entirely.\n" for i in range(3)
)

# rules.py grows from 2 LOC to 3: over the +15 % soft cap (2.3), under the
# 1.5 x hard cap (3.0) — `needs_strong_evidence`, nothing more.
PADDED_RULES = "def rule_noop(trajectory):\n    marker = 1\n    return None\n"


class RichProposer(ScriptedProposer):
    """The fixture's scripted proposer, with rubric and rules per candidate.

    `script[k]` entries are ``(name, policy)``,
    ``(name, policy, rubric_md, rules_py)`` or
    ``(name, policy, rubric_md, rules_py, rationale_md)``; the two-tuple
    form keeps the fixture's rubric and rules. `retry_script[k]`, when given,
    is what leg 2 writes after a leg 1 that wrote nothing (the retry path).
    """

    retry_script: dict[int, list] = {}

    def _write(self, candidates_root, iteration: int, step) -> list[Path]:
        dirs = []
        for index, entry in enumerate(step):
            name, policy = entry[0], entry[1]
            rubric = entry[2] if len(entry) > 2 else RUBRIC
            rules = entry[3] if len(entry) > 3 else RULES
            cand = Path(candidates_root) / f"iter{iteration:03d}_{index:02d}_{name}"
            materialize_candidate(cand, rubric_md=rubric, rules_py=rules, policy_py=policy)
            if len(entry) > 4:
                (cand / "rationale.md").write_text(entry[4], encoding="utf-8")
            dirs.append(cand)
        return dirs

    def propose(self, *, candidates_root, iteration, **kwargs):
        step = self.script.get(iteration, [])
        if isinstance(step, Exception):
            raise step
        prompt = json.dumps({"iteration": iteration, **kwargs}, default=str, sort_keys=True)
        self.prompts.append(prompt)
        return ProposerResult(
            prompt=prompt,
            response=LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0),
            candidate_dirs=self._write(candidates_root, iteration, step),
        )

    def revise(self, *, result, precheck_block, candidates_root, iteration):
        if result.candidate_dirs or iteration not in self.retry_script:
            return super().revise(
                result=result,
                precheck_block=precheck_block,
                candidates_root=candidates_root,
                iteration=iteration,
            )
        self.revise_calls += 1
        self.prompts.append(json.dumps({"iteration": iteration, "leg": "retry"}))
        return ProposerResult(
            prompt=f"{result.prompt}\n\nretry",
            response=result.response,
            candidate_dirs=self._write(candidates_root, iteration, self.retry_script[iteration]),
            legs=result.legs + 1,
        )


def _build(run, *, script, iterations, run_id, candidates_scored=None, **kwargs):
    optimizer = build_optimizer(run, script=script, iterations=iterations, run_id=run_id, **kwargs)
    optimizer.proposer = RichProposer(script, run.judge)
    optimizer.config.candidates_per_iteration = max(len(step) for step in script.values())
    optimizer.config.candidates_scored = candidates_scored
    return optimizer


def _validation(run_dir: Path, k: int) -> dict[str, dict]:
    payload = json.loads(
        (run_dir / "iterations" / f"{k:03d}" / "validation.json").read_text("utf-8")
    )
    return {entry["candidate_id"]: entry for entry in payload}


def _precheck(run_dir: Path, k: int) -> dict:
    return json.loads((run_dir / "iterations" / f"{k:03d}" / "precheck.json").read_text("utf-8"))


def _selection(run_dir: Path, k: int) -> dict:
    return json.loads((run_dir / "iterations" / f"{k:03d}" / "selection.json").read_text("utf-8"))


def _ranking(run_dir: Path, k: int) -> dict:
    return json.loads((run_dir / "iterations" / f"{k:03d}" / "ranking.json").read_text("utf-8"))


def _summary(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# Iteration 0 disputes every pair and loses; iteration 1 is the one under test.
LOSING_FIRST = [("inverted_noisy", INVERTED_NOISY_POLICY)]


@pytest.fixture(scope="module")
def gated_run(tmp_path_factory) -> Path:
    """Five proposed, two scored.

    `truth` and `train_only` are both perfect on the judged train pairs —
    identical correctness vectors — so under the duplicate collapse
    they are one behaviour and only the first is scored; `constant` takes
    the second slot, and the two inverted candidates are rank-gated.
    """

    run = prepare(tmp_path_factory.mktemp("gated"))
    optimizer = _build(
        run,
        script={
            0: LOSING_FIRST,
            1: [
                ("truth", TRUTH_POLICY),
                ("train_only", TRAIN_ONLY_COMPACT_POLICY),
                ("constant", CONSTANT_POLICY),
                ("inverted_truth", INVERTED_TRUTH_POLICY),
                ("inverted_noisy", INVERTED_NOISY_POLICY),
            ],
        },
        iterations=2,
        run_id="gated",
        candidates_scored=2,
    )
    return optimizer.run()


SCORED = {"iter001_00_truth", "iter001_02_constant"}
HELD = {"iter001_01_train_only", "iter001_03_inverted_truth", "iter001_04_inverted_noisy"}
RANK_HELD = {"iter001_03_inverted_truth", "iter001_04_inverted_noisy"}


def test_exactly_candidates_scored_plus_the_baseline_carry_metrics(gated_run):
    validation = _validation(gated_run, 1)

    scored = {cid for cid, entry in validation.items() if entry["invalid_reason"] is None}
    assert scored == SCORED | {"iter001_baseline"}
    assert validation["iter001_00_truth"]["metrics"]["critic_agreement"] == pytest.approx(1.0)


def test_the_held_back_candidates_are_named_with_their_precheck_rank(gated_run):
    payload = _precheck(gated_run, 1)

    assert payload["candidates_scored"] == 2
    assert set(payload["scored"]) == SCORED
    assert set(payload["gated_out"]) == HELD
    for name, held in payload["gated_out"].items():
        assert held["score"] is not None
        assert held["constraint_status"] == "ok"
    # The duplicate is collapsed before the rank cut, so the rank is over
    # the four distinct survivors, not the five proposed.
    assert (
        payload["gated_out"]["iter001_01_train_only"]["reason"]
        == "duplicate of iter001_00_truth on the judged train pairs"
    )
    for name in RANK_HELD:
        assert re.fullmatch(r"precheck rank [34] of 4", payload["gated_out"][name]["reason"]), name
    # The ranks follow the precheck scores: the inverted-noisy scorer keeps
    # the pairs the baseline inverts, the inverted-truth scorer loses every one.
    assert payload["gated_out"]["iter001_04_inverted_noisy"]["reason"] == "precheck rank 3 of 4"
    assert payload["gated_out"]["iter001_03_inverted_truth"]["reason"] == "precheck rank 4 of 4"


def test_held_back_candidates_show_why_in_validation_json_and_were_never_solved(gated_run):
    validation = _validation(gated_run, 1)
    solved = {p.stem for p in (gated_run / "iterations" / "001" / "solver_outputs").iterdir()}

    for cid in RANK_HELD:
        assert validation[cid]["invalid_reason"].startswith("precheck rank")
    assert validation["iter001_01_train_only"]["invalid_reason"].startswith("duplicate of")
    for cid in HELD:
        assert validation[cid]["metrics"]["critic_agreement"] == 0.0
        assert cid not in solved
    assert SCORED <= solved


def test_held_back_directories_stay_on_disk_for_the_next_proposer(gated_run):
    candidates = gated_run / "iterations" / "001" / "candidates"
    for cid in HELD | SCORED:
        assert (candidates / cid / "policy.py").exists(), cid
        # The per-candidate readout is written for every proposed candidate,
        # gated or not: it is what the proposer reads next iteration.
        assert (candidates / cid / "precheck.json").exists(), cid


def test_the_summary_row_counts_the_gate_and_names_the_winners_constraint_status(gated_run):
    rows = _summary(gated_run)

    assert rows[0]["gated_out"] == "0"
    assert rows[0]["constraint_status"] == "baseline", "iteration 0 kept the baseline"
    assert rows[1]["gated_out"] == "3"
    assert rows[1]["selected"] == "iter001_00_truth"
    assert rows[1]["constraint_status"] == "ok"
    header = list(rows[0])
    # Followed only by the confirmation columns.
    assert header[-6:-4] == ["gated_out", "constraint_status"]
    assert header[-4:] == ["sel_b2", "sel_c2", "sel_p_pooled", "gate_confirmed"]


def test_ranking_json_records_each_scored_candidates_constraint_status(gated_run):
    candidates = _ranking(gated_run, 1)["candidates"]

    assert candidates["iter001_baseline"]["constraint_status"] == "baseline"
    for cid in SCORED:
        assert candidates[cid]["constraint_status"] == "ok"
    assert not (HELD & set(candidates)), "held-back candidates have no ranking diag"


def test_the_run_config_records_the_gate(gated_run):
    import yaml

    cfg = yaml.safe_load((gated_run / "config.yaml").read_text("utf-8"))
    assert cfg["candidates_scored"] == 2


def test_without_candidates_scored_every_proposed_candidate_is_scored(tmp_path):
    run = prepare(tmp_path)
    optimizer = _build(
        run,
        script={
            0: LOSING_FIRST,
            1: [
                ("truth", TRUTH_POLICY),
                ("constant", CONSTANT_POLICY),
                ("inverted_truth", INVERTED_TRUTH_POLICY),
            ],
        },
        iterations=2,
        run_id="ungated",
    )
    run_dir = optimizer.run()

    validation = _validation(run_dir, 1)
    assert all(entry["invalid_reason"] is None for entry in validation.values())
    assert _precheck(run_dir, 1)["gated_out"] == {}
    assert _precheck(run_dir, 1)["candidates_scored"] is None
    assert _summary(run_dir)[1]["gated_out"] == "0"


# --- duplicate candidates collapse on the judged train pairs -----------------

# TRUTH_POLICY three ways: identical ordering on every pair, different bytes.
# The first carries one extra policy line, so it is the more complex copy and
# the collapse must keep the second, not the first.
TRUTH_PADDED_POLICY = TRUTH_POLICY + "\nPADDING = 1\n"
TRUTH_CONF6_POLICY = TRUTH_POLICY.replace("'confidence': 0.5", "'confidence': 0.6")
TRUTH_CONF7_POLICY = TRUTH_POLICY.replace("'confidence': 0.5", "'confidence': 0.7")
# The baseline restated: same score on every row as the filter it edits.
BASELINE_RESTATED_POLICY = BASELINE_POLICY.replace("'confidence': 0.5", "'confidence': 0.9")


@pytest.fixture(scope="module")
def duplicates_run(tmp_path_factory) -> Path:
    run = prepare(tmp_path_factory.mktemp("duplicates"))
    optimizer = _build(
        run,
        script={
            0: LOSING_FIRST,
            1: [
                ("truth_padded", TRUTH_PADDED_POLICY),
                ("truth_b", TRUTH_CONF6_POLICY),
                ("truth_c", TRUTH_CONF7_POLICY),
                ("restated_baseline", BASELINE_RESTATED_POLICY),
                ("constant", CONSTANT_POLICY),
            ],
        },
        iterations=2,
        run_id="duplicates",
    )
    return optimizer.run()


def test_three_identical_candidates_score_once_and_the_lowest_complexity_copy_is_kept(
    duplicates_run,
):
    payload = _precheck(duplicates_run, 1)
    held = payload["gated_out"]

    assert set(payload["scored"]) == {"iter001_01_truth_b", "iter001_04_constant"}
    for name in ("iter001_00_truth_padded", "iter001_02_truth_c"):
        assert held[name]["reason"] == "duplicate of iter001_01_truth_b on the judged train pairs"
        assert held[name]["constraint_status"] == "ok"
        assert held[name]["score"] is not None
    # The three copies carry one correctness hash; the survivors carry others.
    after = payload["after"]
    hashes = {name: r["correctness_hash"] for name, r in after.items()}
    assert hashes["iter001_00_truth_padded"] == hashes["iter001_01_truth_b"]
    assert hashes["iter001_01_truth_b"] == hashes["iter001_02_truth_c"]
    assert hashes["iter001_04_constant"] != hashes["iter001_01_truth_b"]
    assert (
        after["iter001_00_truth_padded"]["complexity"] > after["iter001_01_truth_b"]["complexity"]
    )


def test_a_candidate_identical_to_the_baseline_on_the_judged_pairs_is_a_no_op(duplicates_run):
    from meta_filter.optimizer import NOOP_REASON

    held = _precheck(duplicates_run, 1)["gated_out"]
    assert held["iter001_03_restated_baseline"]["reason"] == NOOP_REASON
    validation = _validation(duplicates_run, 1)
    assert validation["iter001_03_restated_baseline"]["invalid_reason"] == NOOP_REASON
    solved = {p.stem for p in (duplicates_run / "iterations" / "001" / "solver_outputs").iterdir()}
    assert "iter001_03_restated_baseline" not in solved
    assert "iter001_00_truth_padded" not in solved and "iter001_02_truth_c" not in solved
    assert {"iter001_01_truth_b", "iter001_04_constant"} <= solved


def test_distinct_candidates_are_untouched_by_the_collapse_and_the_winner_is_the_kept_copy(
    duplicates_run,
):
    validation = _validation(duplicates_run, 1)
    assert validation["iter001_04_constant"]["invalid_reason"] is None
    assert validation["iter001_01_truth_b"]["invalid_reason"] is None
    rows = _summary(duplicates_run)
    assert rows[1]["selected"] == "iter001_01_truth_b"
    assert rows[1]["gated_out"] == "3"
    # No selection-set pair went into the decision: the vectors are the
    # precheck's, on train pairs only.
    for name, held in _precheck(duplicates_run, 1)["gated_out"].items():
        assert "selection" not in held["reason"], name


def test_iteration_zero_has_no_judged_pairs_and_collapses_nothing(tmp_path):
    run = prepare(tmp_path)
    optimizer = _build(
        run,
        script={0: [("truth_a", TRUTH_POLICY), ("truth_b", TRUTH_CONF6_POLICY)]},
        iterations=1,
        run_id="no-collapse-at-zero",
    )
    run_dir = optimizer.run()
    assert _precheck(run_dir, 0)["gated_out"] == {}
    validation = _validation(run_dir, 0)
    assert validation["iter000_00_truth_a"]["invalid_reason"] is None
    assert validation["iter000_01_truth_b"]["invalid_reason"] is None


class _Readout:
    """A `PrecheckResult` stand-in with only what `gate_candidates` reads."""

    def __init__(self, score, credit=None, complexity=1.0, error=""):
        self.score = score
        self.constraint = None
        self.error = error
        self.complexity = complexity
        self.pair_credit = credit or {}

    def correctness_key(self):
        if self.error or not self.pair_credit:
            return None
        return tuple(sorted(self.pair_credit.items()))


def test_gate_candidates_collapses_equal_vectors_and_holds_baseline_copies():
    from meta_filter.optimizer import NOOP_REASON, gate_candidates

    right = {("h1", "l1"): 1.0, ("h2", "l2"): 1.0}
    wrong = {("h1", "l1"): 1.0, ("h2", "l2"): 0.0}
    dirs = [Path(f"/c/{n}") for n in ("a", "b", "c", "d", "e")]
    results = {
        "a": _Readout(0.9, right, complexity=5.0),
        "b": _Readout(0.9, right, complexity=4.0),  # lowest complexity: kept
        "c": _Readout(0.9, right, complexity=4.0),  # ties with b, later: held
        "d": _Readout(0.5, wrong),  # the baseline's vector: no-op
        "e": _Readout(0.1, error="ImportError"),  # no vector: untouched
    }
    kept, held = gate_candidates(dirs, results, candidates_scored=None, parent=_Readout(0.5, wrong))
    assert [d.name for d in kept] == ["b", "e"]
    assert held["a"]["reason"] == "duplicate of b on the judged train pairs"
    assert held["c"]["reason"] == "duplicate of b on the judged train pairs"
    assert held["d"]["reason"] == NOOP_REASON
    assert held["a"]["score"] == 0.9 and held["a"]["constraint_status"] is None


def test_gate_candidates_collapse_precedes_the_rank_cut(tmp_path):
    from meta_filter.optimizer import gate_candidates

    right = {("h1", "l1"): 1.0}
    half = {("h1", "l1"): 0.5}
    dirs = [Path(f"/c/{n}") for n in ("a", "b", "c")]
    results = {"a": _Readout(0.9, right), "b": _Readout(0.9, right), "c": _Readout(0.5, half)}
    kept, held = gate_candidates(dirs, results, candidates_scored=2)
    # Without the collapse the rank cut would keep a and b; with it b is a
    # duplicate and c takes the second slot.
    assert [d.name for d in kept] == ["a", "c"]
    assert held["b"]["reason"] == "duplicate of a on the judged train pairs"


def test_gate_candidates_without_a_parent_or_vectors_collapses_nothing():
    from meta_filter.optimizer import gate_candidates

    dirs = [Path("/c/a"), Path("/c/b")]
    kept, held = gate_candidates(
        dirs, {"a": _Readout(0.3), "b": _Readout(0.3)}, candidates_scored=None
    )
    assert kept == dirs and held == {}


# --- constraints feed the gate ------------------------------------------------


@pytest.fixture(scope="module")
def constrained_run(tmp_path_factory) -> Path:
    """The soft-cap -> +1 z path, on a filter small enough to state in one
    number.

    `soft_floor` is switched off for the duration: production gives rules.py
    35 lines of headroom whatever the baseline, which these 2-LOC fixtures sit
    far inside, and a filter big enough to breach the floor without also
    breaching the 1.5x hard cap would need a 71-line rules.py. What is under
    test here is the gate's response to a breach, not which cap produced it;
    `test_constraints.py` owns the floor itself.
    """

    run = prepare(tmp_path_factory.mktemp("constrained"))
    share_only = ConstraintLimits(soft_floor={})
    with mock.patch.object(ConstraintLimits, "DEFAULT", share_only):
        return _constrained_run(run)


def _constrained_run(run: Path) -> Path:
    optimizer = _build(
        run,
        script={
            0: LOSING_FIRST,
            1: [
                ("overgrown", TRUTH_POLICY, OVERGROWN_RUBRIC, RULES),
                ("padded", TRUTH_POLICY, RUBRIC, PADDED_RULES),
                ("constant", CONSTANT_POLICY),
            ],
        },
        iterations=2,
        run_id="constrained",
    )
    return optimizer.run()


def test_an_invalid_candidate_is_never_scored_and_says_which_cap_fired(constrained_run):
    validation = _validation(constrained_run, 1)
    held = _precheck(constrained_run, 1)["gated_out"]

    reason = validation["iter001_00_overgrown"]["invalid_reason"]
    assert reason.startswith("constraint invalid:")
    assert "rubric_criteria 6 > 1.5 x initial 3 (hard)" in reason
    assert held["iter001_00_overgrown"]["constraint_status"] == "invalid"
    assert set(held) == {"iter001_00_overgrown"}, "invalid is gated with no candidates_scored"
    assert validation["iter001_01_padded"]["invalid_reason"] is None
    assert validation["iter001_02_constant"]["invalid_reason"] is None
    assert (constrained_run / "iterations" / "001" / "candidates" / "iter001_00_overgrown").is_dir()


def test_a_soft_cap_breach_raises_the_candidates_gate_threshold_by_one_z(constrained_run):
    selection = _selection(constrained_run, 1)
    per_candidate = selection["per_candidate"]

    padded = per_candidate["iter001_01_padded"]
    constant = per_candidate["iter001_02_constant"]
    assert padded["strong_evidence"] is True
    assert constant["strong_evidence"] is False
    assert padded["threshold"] == pytest.approx(constant["threshold"] + 1.0)
    assert "min_b_required" in padded and "min_b_required" in constant
    assert "iter001_00_overgrown" not in per_candidate

    constraints = _precheck(constrained_run, 1)["constraints"]
    assert constraints["iter001_01_padded"]["status"] == "needs_strong_evidence"
    assert constraints["iter001_00_overgrown"]["status"] == "invalid"
    assert constraints["iter001_02_constant"]["status"] == "ok"
    assert _ranking(constrained_run, 1)["candidates"]["iter001_01_padded"]["constraint_status"] == (
        "needs_strong_evidence"
    )


def test_the_gate_writes_its_critical_b_table_beside_the_selection_set(constrained_run):
    """`runs/_selection/<run_id>.gate.json`: what b the gate needs at each
    b + c, for the family threshold and the one-z-higher strong-evidence one,
    so a kept-baseline line can be read without recomputing the binomial."""

    sidecar = constrained_run.parent / "_selection" / f"{constrained_run.name}.gate.json"
    record = json.loads(sidecar.read_text("utf-8"))
    family, strong = record["family"], record["strong_evidence"]
    assert record["strong_extra_z"] == 1.0
    assert strong["z"] == pytest.approx(family["z"] + 1.0)
    assert 0 < strong["alpha"] < family["alpha"] < 0.05
    ns = [n for n, _ in family["critical_b"]]
    assert ns == [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]
    assert [n for n, _ in strong["critical_b"]] == ns
    # The table is written for the configured candidate count (three here),
    # before any iteration has run. Iteration 1 held one of its three as
    # invalid and gated the other two at the two-candidate threshold — lower,
    # never higher, than the table's — with the same +1 z for the padded one.
    assert record["n_candidates"] == 3
    assert family["z"] == pytest.approx(z_threshold_for(3))
    per_candidate = _selection(constrained_run, 1)["per_candidate"]
    constant = per_candidate["iter001_02_constant"]["threshold"]
    padded = per_candidate["iter001_01_padded"]["threshold"]
    assert constant == pytest.approx(z_threshold_for(2)) and constant < family["z"]
    assert padded - constant == pytest.approx(record["strong_extra_z"])


# --- the prompt half of the edit discipline ----------------------------------

PARENT = {
    "rubric_criteria": 6,
    "rule_count": 5,
    "rules_loc": 40,
    "policy_loc": 20,
    "rubric_chars": 3000,
}
INITIAL = {
    "rubric_criteria": 5,
    "rule_count": 4,
    "rules_loc": 30,
    "policy_loc": 20,
    "rubric_chars": 2500,
}


def _agentic(n: int = 6, **kwargs) -> str:
    return " ".join(
        _agentic_instructions(
            "iterations/002/candidates",
            2,
            n,
            "iterations/002/current_filter",
            parent_counts=PARENT,
            initial_counts=INITIAL,
            **kwargs,
        ).split()
    )


def test_the_ranking_mission_states_the_hypothesis_rule_and_the_deletion_limit():
    text = _agentic()

    assert "exactly one line beginning `Hypothesis:`" in text
    assert "diagnosis.json" in text.split("`Hypothesis:`", 1)[1][:300]
    assert "You are writing exactly 6 candidates" in text
    assert "at least 4 distinct mechanisms" in text
    assert "At most one candidate per iteration may delete" in text
    assert "cite the judged pairs that motivate the removal" in text
    assert "Meta-Harness (App. A.2) logged six consecutive regressions" in text


def test_the_caps_are_printed_with_the_parents_and_the_initial_filters_numbers():
    text = _agentic()

    assert "The current filter has 6 rubric criteria, 5 `rule_` functions, 40 rules.py LOC" in text
    # Soft caps against the parent.
    assert "+2 criteria (to 8), +2 rules (to 7)" in text
    # rules.py: 40 + max(15 % = 6, the 35-line floor) = 75. The floor is what
    # keeps this number and the "+2 rules" above it from contradicting each
    # other — two rule bodies is 32 lines at the median, which 46 could not
    # hold. Above a 233-line rules.py the share is the larger of the two again.
    assert "+15% LOC or characters (rules.py to 75 LOC, rubric.md to 3450" in text
    assert "policy.py to" not in text, "policy.py LOC is uncapped"
    # Integer arithmetic on both sides: 3000 + 15 % is 3450 exactly, and
    # `constraints.check_candidate` fires on `value * 100 > base * 115`, so
    # 3450 is the number a candidate may actually reach. (`int(3000 * 1.15)`
    # is 3449 — the float form printed one less than the check allowed.)
    assert "rubric.md to 3450 characters)" in text
    assert "must clear a stricter threshold, one z-unit higher" in text
    assert "α 0.0003 instead of 0.0076" in text
    assert "twice the evidence" not in text
    # Hard caps and floor against the initial filter: 1.5 x 5 = 7.5 -> 7.
    assert "1.5x the initial filter (criteria 7, rules 6, rules.py 45 LOC" in text
    assert "rubric.md 3750 characters). policy.py LOC is not capped" in text
    assert "policy.py 30 LOC" not in text
    # Floors from `ConstraintLimits.floor_for`: criteria 5 - 1; rules
    # max(3, 4 - 5) = 3, and the prompt states the rule as well as the number.
    assert "initial count minus 1 (4 criteria)" in text
    assert "rules never below the initial minus 5, never below 3 (3 rules)" in text
    assert "pruning rules that do not order pairs is allowed" in text


def test_the_gating_sentence_appears_only_when_gating_is_on():
    assert "Only 3 of your 6 candidates are fully scored" in _agentic(candidates_scored=3)
    assert "fully scored" not in _agentic()
    assert "fully scored" not in _agentic(candidates_scored=6), "6 of 6 is not a gate"


def _static_prompt(tmp_path, **attrs) -> str:
    """The static proposer prompt as the optimizer renders it."""

    from meta_filter.proposer import Proposer

    proposer = Proposer(n_candidates=attrs.pop("n_candidates", 3))
    for name, value in attrs.items():
        setattr(proposer, name, value)
    return proposer._build_prompt(
        loaded_filter=a_filter(tmp_path),
        iteration=1,
        current_metrics=stub_metrics(),
        recent_false_negatives=[],
        recent_false_positives=[],
        **REPORTS,
    )


def test_the_static_ranking_prompt_carries_the_block_with_the_fixture_filters_counts(tmp_path):
    prompt = _static_prompt(tmp_path)
    text = " ".join(prompt.split())

    assert "## Edit discipline" in prompt
    assert '"hypothesis": "<the single mechanism this candidate tests' in prompt
    # The fixture filter: one criterion, one rule, 2 LOC of rules, 3 of policy.
    assert "The current filter has 1 rubric criteria, 1 `rule_` functions, 2 rules.py LOC" in text
    assert "+2 criteria (to 3), +2 rules (to 3)" in text
    assert "fully scored" not in text


def test_the_static_prompt_says_how_many_are_scored_when_the_optimizer_told_it(tmp_path):
    prompt = _static_prompt(
        tmp_path, n_candidates=6, candidates_scored=3, initial_complexity=INITIAL
    )
    text = " ".join(prompt.split())
    assert "Only 3 of your 6 candidates are fully scored" in text
    assert "1.5x the initial filter (criteria 7, rules 6" in text


def test_the_prompt_prints_the_rule_floor_the_check_computes_for_a_real_sized_filter():
    """Nine rules: the floor is 9 - 5 = 4, the minimum does not bind, and the
    number in the prompt is `floor_for`'s."""

    from meta_filter.constraints import ConstraintLimits

    parent = {**PARENT, "rule_count": 9}
    text = " ".join(
        render_edit_discipline(
            n_candidates=3, parent_counts=parent, hypothesis_source="the JSON"
        ).split()
    )
    assert ConstraintLimits.DEFAULT.floor_for("rule_count", 9) == 4
    assert "(4 rules)" in text


def test_the_static_prompt_says_a_rename_is_a_deletion(tmp_path):
    text = " ".join(_static_prompt(tmp_path).split())
    assert "Renaming a criterion or a rule counts as a deletion" in text
    assert "keep names stable unless you mean to remove it" in text


def test_render_edit_discipline_falls_back_to_the_parent_as_the_initial_filter():
    text = " ".join(
        render_edit_discipline(
            n_candidates=3, parent_counts=PARENT, hypothesis_source="the JSON"
        ).split()
    )
    # 1.5 x 6 = 9 criteria; criteria floor 6 - 1 = 5; rules max(3, 5 - 5) = 3.
    assert "1.5x the initial filter (criteria 9, rules 7" in text
    assert "(5 criteria)" in text
    assert "(3 rules)" in text
    assert "at least 3 distinct mechanisms" in text


# --- rationale.md carries exactly one Hypothesis: line -------------------------


def _hypothesis_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.lower().lstrip("-*# ").startswith("hypothesis:")
    ]


def test_the_rationale_writer_emits_one_hypothesis_line_from_the_proposers_field(tmp_path):
    _write_rationale_md(
        tmp_path,
        {"name": "x", "hypothesis": "  verification\nis under-weighted ", "notes": "n"},
        ["d1"],
    )
    assert _hypothesis_lines(tmp_path / "rationale.md") == [
        "Hypothesis: verification is under-weighted"
    ]


def test_a_hypothesis_the_agent_wrote_into_rationale_md_survives_the_rewrite(tmp_path):
    (tmp_path / "rationale.md").write_text(
        "# mine\n\n## Hypothesis: the rule fires on tests too\n\nmore prose\n", encoding="utf-8"
    )
    _write_rationale_md(tmp_path, {"name": "x", "notes": "n"}, [])
    assert _hypothesis_lines(tmp_path / "rationale.md") == [
        "Hypothesis: the rule fires on tests too"
    ]


def test_no_stated_hypothesis_writes_no_hypothesis_line(tmp_path):
    _write_rationale_md(tmp_path, {"name": "x", "notes": "n"}, ["d"])
    assert _hypothesis_lines(tmp_path / "rationale.md") == []


def test_a_line_the_agent_wrote_is_kept_and_the_field_is_not_added_beside_it(tmp_path):
    """One idea stated twice (file and field) must not read as two mechanisms."""

    (tmp_path / "rationale.md").write_text("Hypothesis: from the file\n", encoding="utf-8")
    _write_rationale_md(tmp_path, {"name": "x", "hypothesis": "from the field"}, [])
    assert _hypothesis_lines(tmp_path / "rationale.md") == ["Hypothesis: from the file"]


# --- two declared mechanisms survive the rewrite and are rejected -------------


def test_every_hypothesis_line_the_agent_wrote_survives_the_rewrite(tmp_path):
    """Collapsing to the first line is how the two-mechanism rejection was
    unable to fire: `count_mechanisms` ran on a file this writer had already
    normalised to one line."""

    from meta_filter.constraints import count_mechanisms

    (tmp_path / "rationale.md").write_text(
        "# mine\n\n## Hypothesis: the rule fires on tests too\n\nprose\n\n"
        "- Hypothesis: and the review band is too wide\n",
        encoding="utf-8",
    )
    _write_rationale_md(tmp_path, {"name": "x", "hypothesis": "only one", "notes": "n"}, ["d"])
    assert _hypothesis_lines(tmp_path / "rationale.md") == [
        "Hypothesis: the rule fires on tests too",
        "Hypothesis: and the review band is too wide",
    ]
    assert count_mechanisms(tmp_path / "rationale.md") == 2


def test_a_field_carrying_two_markers_is_written_as_two_lines(tmp_path):
    from meta_filter.constraints import count_mechanisms

    _write_rationale_md(
        tmp_path,
        {"name": "x", "hypothesis": "Hypothesis: A is under-weighted. Hypothesis: B never fires"},
        [],
    )
    assert _hypothesis_lines(tmp_path / "rationale.md") == [
        "Hypothesis: A is under-weighted.",
        "Hypothesis: B never fires",
    ]
    assert count_mechanisms(tmp_path / "rationale.md") == 2


def test_a_two_mechanism_candidate_is_rejected_by_the_constraint_check(tmp_path):
    """End to end through `check_candidate`: the writer keeps both lines and
    the precheck's constraint report comes back invalid."""

    from meta_filter.constraints import check_candidate

    base = tmp_path / "base"
    materialize_candidate(base, rubric_md=RUBRIC, rules_py=RULES, policy_py=TRUTH_POLICY)
    cand = tmp_path / "cand"
    materialize_candidate(cand, rubric_md=RUBRIC, rules_py=RULES, policy_py=TRUTH_POLICY)
    (cand / "rationale.md").write_text("Hypothesis: one\nHypothesis: two\n", encoding="utf-8")
    _write_rationale_md(cand, {"name": "cand", "notes": "n"}, [])

    report = check_candidate(cand, baseline_dir=base, initial_dir=base)
    assert report.status == "invalid"
    assert report.mechanism_count == 2
    assert any("2 hypotheses" in r for r in report.reasons)


def test_a_diagnosis_bullet_starting_with_the_marker_is_not_a_hypothesis(tmp_path):
    from meta_filter.constraints import count_mechanisms

    _write_rationale_md(
        tmp_path,
        {"name": "x", "hypothesis": "the real one", "targets": ["Hypothesis: t over u"]},
        ["Hypothesis: the proposer titled its observation this way", "plain observation"],
    )
    text = (tmp_path / "rationale.md").read_text(encoding="utf-8")
    assert _hypothesis_lines(tmp_path / "rationale.md") == ["Hypothesis: the real one"]
    assert "- note: Hypothesis: the proposer titled its observation this way" in text
    assert "- note: Hypothesis: t over u" in text
    assert "- plain observation" in text
    assert count_mechanisms(tmp_path / "rationale.md") == 1


def test_diagnosis_json_hypotheses_are_read_per_candidate(tmp_path):
    (tmp_path / "diagnosis.json").write_text(
        json.dumps(
            {
                "diagnosis": "one line",
                "candidates": [
                    {"name": "alpha", "hypothesis": "A alone", "notes": "na"},
                    {"name": "beta", "notes": "nb"},
                ],
            }
        ),
        encoding="utf-8",
    )
    diagnosis, notes, hypotheses = _read_diagnosis(tmp_path)
    assert diagnosis == ["one line"]
    assert notes == {"alpha": "na", "beta": "nb"}
    assert hypotheses == {"alpha": "A alone", "beta": ""}


# --- iteration 0's retry path: caps gate, ranks do not ------------------------


def test_the_retry_path_applies_the_caps_and_never_rank_gates_without_scores(tmp_path):
    """Leg 1 wrote nothing loadable, so leg 2 is a retry with no judged pair
    behind it. The caps are file-level facts and still gate; a rank cut with
    no score would keep the first N in directory order, which is no rank."""

    run = prepare(tmp_path)
    optimizer = _build(
        run,
        script={0: []},
        iterations=1,
        run_id="retry",
        candidates_scored=2,
    )
    optimizer.proposer.retry_script = {
        0: [
            ("constant", CONSTANT_POLICY),
            ("overgrown", TRUTH_POLICY, OVERGROWN_RUBRIC, RULES),
            ("truth", TRUTH_POLICY),
            ("inverted_truth", INVERTED_TRUTH_POLICY),
        ]
    }
    run_dir = optimizer.run()

    assert optimizer.proposer.revise_calls == 1, "the one retry leg"
    payload = _precheck(run_dir, 0)
    assert payload["candidates_scored"] == 2
    assert set(payload["gated_out"]) == {"iter000_01_overgrown"}
    assert payload["gated_out"]["iter000_01_overgrown"]["reason"].startswith("constraint invalid:")
    assert payload["gated_out"]["iter000_01_overgrown"]["constraint_status"] == "invalid"
    assert set(payload["scored"]) == {
        "iter000_00_constant",
        "iter000_02_truth",
        "iter000_03_inverted_truth",
    }, "three survivors scored although candidates_scored is 2: no scores, no rank"
    assert payload["after"] == {}, "no measurement was written for pairs that do not exist"
    assert set(payload["constraints"]) == set(payload["scored"]) | set(payload["gated_out"])
    validation = _validation(run_dir, 0)
    assert validation["iter000_01_overgrown"]["invalid_reason"].startswith("constraint invalid:")
    assert all(validation[cid]["invalid_reason"] is None for cid in payload["scored"])


# --- five candidates, all scored ----------------------------------------------

# `Truth`, except that every third row (the pool's turn counts cycle 2, 3, 4)
# is read upside down and so sinks to the bottom. Right on most judged pairs
# and wrong on the ones those rows appear in: a fifth per-pair vector, which
# is what the duplicate collapse needs if five candidates are to stay
# standing.
# Distinct from `truth` by mixing the noise axis in, not by reading the
# trajectory: a policy that grades the transcript is invalid
# (`constraints.policy_reads_trajectory`), and what this fixture needs is a
# fifth DISTINCT candidate, not a grading one.
PARTLY_INVERTED_POLICY = """RUBRIC_CRITERIA = ['Truth', 'Noisy']


def decide(trajectory, scores, triggered):
    return {
        'decision': 'REVIEW',
        'score': scores['Truth'] - scores['Noisy'],
        'confidence': 0.5,
    }
"""

FIVE_DISTINCT = [
    ("truth", TRUTH_POLICY),
    ("constant", CONSTANT_POLICY),
    ("inverted_truth", INVERTED_TRUTH_POLICY),
    ("inverted_noisy", INVERTED_NOISY_POLICY),
    ("partly_inverted", PARTLY_INVERTED_POLICY),
]


def test_five_candidates_without_a_scored_cap_are_all_scored(tmp_path):
    """`--candidates 5` and no `--candidates-scored`.

    The rank cut is what `candidates_scored` buys; without it every survivor
    of the caps and the duplicate collapse goes through the full train +
    selection pass. Five distinct behaviours survive the collapse, so
    `validation.json` carries five candidates plus the baseline.
    """

    run = prepare(tmp_path)
    run_dir = _build(
        run,
        script={0: LOSING_FIRST, 1: FIVE_DISTINCT},
        iterations=2,
        run_id="five",
        candidates_scored=None,
    ).run()

    payload = _precheck(run_dir, 1)
    assert payload["candidates_scored"] is None
    assert len(payload["scored"]) == 5, payload["gated_out"]
    assert payload["gated_out"] == {}

    validation = _validation(run_dir, 1)
    assert len(validation) == 6, "five candidates plus the baseline"
    assert sum(entry["invalid_reason"] is None for entry in validation.values()) == 6
    # Every one of them was measured on the hidden pairs too.
    assert len(_selection(run_dir, 1)["per_candidate"]) == 5
