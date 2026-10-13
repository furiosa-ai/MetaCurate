"""A criterion that moves no pair must not be protected by the deletion floor.

A run at the floor (six criteria down to five, `initial - 1`) told which axis
contributes least would be forbidden to delete it — so the proposer sets its
weight to 0.0 instead, the only move left, and the run ends carrying a
criterion whose scores spread over +0/+1/+2 (not a DEAD axis), read by
`decide()` (not an UNREAD one), contributing exactly zero to the ordering,
and still costing one scorer call on every row.

Neither of the other health findings covers that third case, and the floor
would turn the correct repair into a half one.
"""

from __future__ import annotations

from pathlib import Path

from meta_filter.diagnostics.criterion_lift import ZERO_LIFT_ACTION, render_ranking, zero_lift_items
from meta_filter.gate.constraints import check_candidate

RUBRIC = """# Rubric

## Criterion: Alpha

+2: good
-2: bad

## Criterion: Beta

+2: good
-2: bad

## Criterion: Gamma

+2: good
-2: bad
"""


def _filter(directory: Path, criteria: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    sections = "# Rubric\n\n" + "\n".join(
        f"## Criterion: {name}\n\n+2: good\n-2: bad\n" for name in criteria
    )
    (directory / "rubric.md").write_text(sections)
    (directory / "rules.py").write_text("def rule_one(trajectory):\n    return None\n")
    (directory / "policy.py").write_text(
        "RUBRIC_CRITERIA = [\n"
        + "".join(f'    "{n}",\n' for n in criteria)
        + "]\n\n\ndef decide(trajectory, rubric_scores, triggered_rules):\n"
        "    return {'decision': 'ACCEPT', 'score': 1.0}\n"
    )
    (directory / "rationale.md").write_text("one mechanism\n")
    return directory


def test_the_floor_blocks_the_deletion_when_nothing_says_the_axis_is_inert(tmp_path):
    initial = _filter(tmp_path / "initial", ["Alpha", "Beta", "Gamma"])
    baseline = _filter(tmp_path / "baseline", ["Alpha", "Beta"])
    candidate = _filter(tmp_path / "cand", ["Alpha"])

    report = check_candidate(candidate, baseline_dir=baseline, initial_dir=initial)

    assert report.status == "invalid"
    assert any("deletion floor" in r for r in report.reasons)


def test_a_measured_zero_lift_axis_is_credited_back(tmp_path):
    """The same candidate, once the harness has measured Beta at zero lift."""

    initial = _filter(tmp_path / "initial", ["Alpha", "Beta", "Gamma"])
    baseline = _filter(tmp_path / "baseline", ["Alpha", "Beta"])
    candidate = _filter(tmp_path / "cand", ["Alpha"])

    report = check_candidate(
        candidate, baseline_dir=baseline, initial_dir=initial, zero_lift_criteria=("Beta",)
    )

    assert report.status == "ok", report.reasons
    assert not any("deletion floor" in r for r in report.reasons)


def test_the_credit_is_only_for_the_axis_that_was_measured_inert(tmp_path):
    """Deleting a DIFFERENT axis does not earn the slot. The credit tracks the
    axis the measurement named, not the act of deleting something."""

    initial = _filter(tmp_path / "initial", ["Alpha", "Beta", "Gamma"])
    baseline = _filter(tmp_path / "baseline", ["Alpha", "Beta"])
    candidate = _filter(tmp_path / "cand", ["Beta"])  # Alpha removed, Beta kept

    report = check_candidate(
        candidate, baseline_dir=baseline, initial_dir=initial, zero_lift_criteria=("Beta",)
    )

    assert report.status == "invalid"
    assert any("deletion floor" in r for r in report.reasons)


def test_zero_lift_items_reads_exact_zeros_only():
    report = {
        "rows_ranked": 50,
        "rubric_rank_shift": 0.17,
        "criterion_rank_shift": {"Alpha": 0.02, "Beta": 0.0, "Gamma": 0.0001},
        "rule_rank_shift": {"rule_one": 0.0, "rule_two": 0.03},
    }
    criteria, rules = zero_lift_items(report)

    # 0.0001 is a weak axis, which is a judgement; 0 is a fact about the ordering.
    assert criteria == ["Beta"]
    assert rules == ["rule_one"]


def test_the_prompt_names_the_zero_lift_items_and_says_what_to_do():
    report = {
        "rows_ranked": 50,
        "rubric_rank_shift": 0.17,
        "criterion_rank_shift": {"Alpha": 0.02, "Beta": 0.0},
        "rule_rank_shift": {"rule_one": 0.0},
    }
    text = render_ranking(report)

    assert "ZERO LIFT" in text
    assert "Beta" in text and "rule_one" in text
    assert ZERO_LIFT_ACTION.split(".")[0] in text
    # And it must say the floor is not an excuse, because that is what stopped
    # the repair the one time this arose.
    assert "deletion floor does not count" in text


def test_a_report_with_no_zero_lift_item_says_nothing_extra():
    report = {
        "rows_ranked": 50,
        "rubric_rank_shift": 0.17,
        "criterion_rank_shift": {"Alpha": 0.02},
        "rule_rank_shift": {"rule_one": 0.03},
    }
    assert "ZERO LIFT" not in render_ranking(report)


REPORT = {
    "rows_ranked": 50,
    "rubric_rank_shift": 0.2,
    "criterion_rank_shift": {"Alpha": 0.0694, "Beta": 0.0},
    "rule_rank_shift": {"rule_two": 0.0073, "rule_one": 0.0},
}


def test_the_action_tells_every_candidate_to_carry_the_removal():
    """A solo prune candidate cannot win, so the removal must ride along.

    A proposer that reads the ZERO LIFT finding and obeys "one hypothesis
    per candidate" isolates the removal into a candidate of its own — which
    scores b=0, c=0 over the whole exam, exactly as the lift number predicts,
    ties its parent, and loses to any sibling with a real gain. The item then
    survives into the promoted filter. Naming the item is necessary and not
    sufficient; the action has to say WHERE the removal goes.
    """

    text = render_ranking(REPORT)
    assert "Remove them in EVERY candidate you write" in text
    assert "Do NOT give the removal a candidate of its own" in text


def test_the_action_names_the_boundary_against_the_nearest_survivor():
    """`_ranked_float` sorts descending, so the zero sits next to the smallest
    non-zero item, and the bullet above invites pruning NEAR-zero ones. The
    boundary has to be marked against the data, not only asserted in prose."""

    text = render_ranking(REPORT)
    assert "the next-smallest item is rule_two at 0.0073" in text
    assert "must not be removed this way" in text
    assert "Alpha" not in text.split("ZERO LIFT")[1]


def test_the_action_does_not_claim_invariance_it_did_not_measure():
    """The zero is measured by ablating the CURRENT filter over the train rows.
    It is asserted about a DIFFERENT policy's ordering over hidden pairs, so it
    is evidence, not a proof — and a candidate that changes how decide()
    aggregates the rubric can break it."""

    text = render_ranking(REPORT)
    assert "strong evidence the removal is free, not a proof" in text
    assert "re-check that" in text and "inert under YOUR policy" in text
    assert "cannot change your candidate's b/c" not in text


def test_the_action_leaves_the_repair_and_dependency_routes_open():
    """ "Make it matter", a merge, and a precondition all stay real hypotheses."""

    text = render_ranking(REPORT)
    keep = text.split("KEEP it in one candidate only if", 1)[1]
    assert "weight in decide()" in keep and "anchors that separate these rows" in keep
    assert "merge into" in keep
    assert "precondition" in keep


def test_the_one_hypothesis_rule_carves_out_the_zero_lift_removal():
    """The two instructions must not contradict each other in one prompt."""

    from meta_filter.propose.agentic_proposer import _agentic_instructions

    text = _agentic_instructions(
        "candidates", 1, 3, "current_filter", parent_counts={"rubric_criteria": 3}
    )
    assert "One hypothesis per candidate" in text
    carve_out = text.split("One hypothesis per candidate", 1)[1].split(
        "Candidates should test genuinely different", 1
    )[0]
    flat = " ".join(carve_out.split()).lower()
    assert "zero lift" in flat
    assert "fold it into every candidate" in flat
    assert "ties its parent by construction" in flat


def test_a_measured_inert_removal_does_not_consume_the_deletion_slot(tmp_path):
    """The crux. `gate_candidates` holds back every deletion candidate but one,
    UNSCORED. Telling the proposer to fold the removal into EVERY candidate
    made all of them deletion candidates, so obeying the instruction would have
    cost two slots where the bug it fixes cost one."""

    baseline = _filter(tmp_path / "base", ["Alpha", "Beta", "Gamma"])
    # a candidate carrying its own hypothesis AND the mandated removal
    candidate = _filter(tmp_path / "cand", ["Alpha", "Gamma"])
    (candidate / "policy.py").write_text(
        (candidate / "policy.py").read_text() + "\n\nEXTRA_THRESHOLD = 1.0\n"
    )

    report = check_candidate(
        candidate, baseline_dir=baseline, initial_dir=baseline, zero_lift_criteria=("Beta",)
    )
    assert report.deleted == ["Beta"]
    assert report.deleted_inert == ["Beta"]
    assert report.deletion is False, "a measured-inert removal must not use the slot"
    # `precheck.json` has to carry the reason `deletion` is False while
    # `deleted` is not empty, or the run cannot be audited afterwards.
    assert report.to_dict()["deleted_inert"] == ["Beta"]


def test_a_live_removal_still_consumes_the_deletion_slot(tmp_path):
    """The exemption is for the measured name, not for the act of deleting."""

    baseline = _filter(tmp_path / "base", ["Alpha", "Beta", "Gamma"])
    candidate = _filter(tmp_path / "cand", ["Alpha", "Beta"])

    report = check_candidate(
        candidate, baseline_dir=baseline, initial_dir=baseline, zero_lift_criteria=("Beta",)
    )
    assert report.deleted == ["Gamma"]
    assert report.deleted_inert == []
    assert report.deletion is True


def test_the_floor_credit_reaches_rules_not_only_criteria(tmp_path):
    """The finding names rules in the same sentence as criteria and the action
    asks for both. Crediting only criteria would write a cheque the rule floor
    bounces — and under "in EVERY candidate" that bounce invalidates the whole
    iteration, not one candidate."""

    from meta_filter.gate.constraints import ConstraintLimits, check_candidate

    def _rules(directory: Path, names: list[str]) -> None:
        (directory / "rules.py").write_text(
            "\n\n".join(f"def {n}(trajectory):\n    return None" for n in names) + "\n"
        )

    baseline = _filter(tmp_path / "base", ["Alpha", "Beta", "Gamma"])
    _rules(baseline, ["rule_a", "rule_b", "rule_c"])
    candidate = _filter(tmp_path / "cand", ["Alpha", "Beta", "Gamma"])
    _rules(candidate, ["rule_a", "rule_b"])

    limits = ConstraintLimits.DEFAULT
    blocked = check_candidate(candidate, baseline_dir=baseline, initial_dir=baseline, limits=limits)
    credited = check_candidate(
        candidate,
        baseline_dir=baseline,
        initial_dir=baseline,
        limits=limits,
        zero_lift_rules=("rule_c",),
    )
    if any("deletion floor" in r for r in blocked.reasons):
        assert not any("deletion floor" in r for r in credited.reasons)
    assert credited.deleted_inert == ["rule_c"]
    assert credited.deletion is False


def test_the_revision_leg_carries_the_finding(tmp_path):
    """Leg 2 is a fresh process and is the leg that ships. Without the note, a
    removal folded into every candidate reads there as a shrunken filter with
    no upside, and the obvious repair is to put the criterion back."""

    from meta_filter.diagnostics.criterion_lift import zero_lift_note
    from meta_filter.propose.agentic_proposer import _revision_instructions

    note = zero_lift_note(REPORT)
    assert note.startswith("- ZERO LIFT")

    without = _revision_instructions("candidates", 1, "precheck")
    with_note = _revision_instructions("candidates", 1, "precheck", zero_lift_note=note)
    assert "ZERO LIFT" not in without
    assert "ZERO LIFT" in with_note
    assert "Restoring one is not a repair" in with_note


def test_iteration_zero_measures_the_lift_it_would_otherwise_propose_blind(tmp_path):
    """The blind spot is where inherited dead axes live.

    The measurement normally runs at the END of an iteration, on the winner,
    so iteration k's propose call reads iteration k-1's report and iteration 0
    reads nothing. That is the worst place for it: an inherited dead axis
    comes from the seed, no candidate invented it, and iteration 0 is where
    the seed rubric is edited wholesale — a candidate can spend its slot
    raising that criterion's weight instead of removing it.

    Nothing extra is bought: step 1 of the loop already scores the baseline
    before the proposer is called, and ablation re-runs `decide()` on those
    scores.
    """

    import json

    from tests.selection_gate_fixture import prepare
    from tests.test_candidate_gating import LOSING_FIRST, _build

    run = prepare(tmp_path)
    run_dir = _build(run, script={0: LOSING_FIRST}, iterations=1, run_id="seed-lift").run()

    records = [
        json.loads(line)
        for line in (run_dir / "criterion_lift.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    pre = [r for r in records if r.get("stage") == "pre_propose"]
    assert pre, "iteration 0 proposed with no rank-lift measurement at all"
    assert pre[0]["iteration"] == 0
    assert "criterion_rank_shift" in pre[0]
    # one record per iteration for the end-of-loop measurement is preserved:
    # the stage marker is what keeps the two apart.
    assert all("stage" not in r for r in records if r not in pre)
