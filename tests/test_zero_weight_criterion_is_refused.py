"""A criterion declared at weight zero is caught before it is ever scored.

The fourth health finding. A run at the criteria deletion floor that cannot
remove its least useful axis can set the axis's weight to 0.0 instead — the
only move the floor leaves it — and end carrying a criterion scored on every
row, contributing nothing through the weighted sum, and invisible to both
other checks. It is not UNREAD (the policy names it) and not DEAD (its
scores spread over +0/+1/+2), so only the rank-lift MEASUREMENT could name
it, which takes a whole iteration and cannot see iteration 0's own
candidates at all.

This check is static: it fires on the candidate that writes it, in the
precheck, before one row is scored.
"""

from __future__ import annotations

from pathlib import Path

from meta_filter.gate.constraints import (
    ConstraintLimits,
    check_candidate,
    newly_zero_weight_criteria,
    priced_criteria,
    zero_weight_criteria,
)

DECIDE = """

def decide(trajectory, rubric_scores, triggered_rules):
    total = sum(rubric_scores.get(n, 0.0) * RUBRIC_WEIGHTS[n] for n in RUBRIC_CRITERIA)
    worst = min(rubric_scores.values()) if rubric_scores else 0.0
    return {'decision': 'ACCEPT', 'score': total + worst}
"""


def _filter(directory: Path, weights: dict[str, float], extra: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rubric.md").write_text(
        "# Rubric\n\n" + "\n".join(f"## Criterion: {n}\n\n+2: good\n-2: bad\n" for n in weights)
    )
    (directory / "rules.py").write_text("def rule_one(trajectory):\n    return None\n")
    (directory / "policy.py").write_text(
        "RUBRIC_CRITERIA = [\n"
        + "".join(f'    "{n}",\n' for n in weights)
        + "]\n\nRUBRIC_WEIGHTS = {\n"
        + "".join(f'    "{n}": {w},\n' for n, w in weights.items())
        + "}\n"
        + extra
        + DECIDE
    )
    (directory / "rationale.md").write_text("one mechanism\n")
    return directory


LIVE = {"Alpha": 2.0, "Beta": 1.0, "Gamma": 3.0}
ZEROED = {"Alpha": 2.0, "Beta": 0.0, "Gamma": 3.0}


def test_writing_a_zero_weight_criterion_is_invalid(tmp_path):
    baseline = _filter(tmp_path / "base", LIVE)
    candidate = _filter(tmp_path / "cand", ZEROED)

    report = check_candidate(candidate, baseline_dir=baseline, initial_dir=baseline)
    assert report.status == "invalid"
    reason = next(r for r in report.reasons if "weight 0" in r)
    assert "'Beta'" in reason
    assert "one scorer call on every row" in reason


def test_inheriting_one_is_recorded_but_not_charged(tmp_path):
    """Otherwise a run whose seed filter carries the defect invalidates every
    candidate at once, and the iteration has nothing to score. The same rule
    `declared_unread` and `new_dead_axes` already follow."""

    baseline = _filter(tmp_path / "base", ZEROED)
    candidate = _filter(tmp_path / "cand", ZEROED, extra="EXTRA = 1.0\n")

    report = check_candidate(candidate, baseline_dir=baseline, initial_dir=baseline)
    assert report.zero_weight == ["Beta"]
    assert not any("weight 0" in r for r in report.reasons)
    assert report.to_dict()["zero_weight"] == ["Beta"]


def test_the_finding_is_not_a_claim_that_the_criterion_is_unused(tmp_path):
    """`decide()` above also reads `min(rubric_scores.values())`, so Beta can
    still move the score through the worst-criterion channel. The check is
    about the weighted term paying nothing, not about the axis having no
    effect — a policy at a -2 floor really does react to it."""

    from meta_filter.gate.constraints import declared_unread_criteria

    zeroed = _filter(tmp_path / "z", ZEROED)
    assert zero_weight_criteria(zeroed) == ["Beta"]
    assert "Beta" not in declared_unread_criteria(zeroed), "it is read; only its weight is 0"


def test_ambiguous_weights_say_nothing(tmp_path):
    """0 in one table and non-zero in another is not a finding. Conservative
    in the same direction as `declared_unread_criteria`: the safe way to be
    wrong is to stay quiet."""

    directory = _filter(tmp_path / "amb", ZEROED, extra='PENALTIES = {"Beta": 4.0}\n')
    assert zero_weight_criteria(directory) == []


def test_an_unreadable_policy_says_nothing(tmp_path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "policy.py").write_text("RUBRIC_CRITERIA = [\n")  # syntax error
    assert zero_weight_criteria(directory) == []
    assert zero_weight_criteria(tmp_path / "absent") == []


def test_the_floor_never_protects_a_zero_weight_criterion(tmp_path):
    """The deadlock this check would otherwise create.

    A filter at the criteria floor that carries a zero-weight criterion could
    neither delete it (floor) nor keep it (this check, once it is the
    candidate's own doing) — a position with no legal move. The floor credit
    for a measured zero-lift removal does not help at iteration 0, where no
    measurement has landed. So the baseline's STATIC zero-weight declaration
    earns the same credit, with no measurement at all.
    """

    baseline = _filter(tmp_path / "base", ZEROED)
    candidate = _filter(tmp_path / "cand", {"Alpha": 2.0, "Gamma": 3.0})
    initial = _filter(tmp_path / "init", {**LIVE, "Delta": 1.0})

    limits = ConstraintLimits.DEFAULT
    report = check_candidate(candidate, baseline_dir=baseline, initial_dir=initial, limits=limits)
    assert not any("deletion floor" in r for r in report.reasons), report.reasons
    assert report.deleted == ["Beta"]


def test_the_static_credit_reaches_the_floor_but_not_the_deletion_slot(tmp_path):
    """The asymmetry is deliberate, and the two credits answer different
    questions.

    The FLOOR asks "may this be removed at all?" — and a criterion the filter
    itself declares at zero weight must always be removable, or the check
    above traps a candidate with no legal move.

    The SLOT asks "may every candidate remove it at once without the loop
    losing its ability to attribute damage?" — and only the MEASURED zero
    answers that. Zero weight is not proof of harmlessness: these policies
    also read `min(rubric_scores.values())` against a floor, so a zero-weight
    criterion can still move the score, and a removal that moves the ordering
    is exactly the kind the one-per-iteration rule takes slowly.

    When the criterion is both — the usual case —
    the measured credit applies and the removal rides in every candidate.
    """

    baseline = _filter(tmp_path / "base", ZEROED)
    candidate = _filter(tmp_path / "cand", {"Alpha": 2.0, "Gamma": 3.0})
    initial = _filter(tmp_path / "init", {**LIVE, "Delta": 1.0})

    static_only = check_candidate(candidate, baseline_dir=baseline, initial_dir=initial)
    assert static_only.deletion is True, "zero weight alone does not free the slot"
    assert static_only.deleted_inert == []

    measured = check_candidate(
        candidate,
        baseline_dir=baseline,
        initial_dir=initial,
        zero_lift_criteria=("Beta",),
    )
    assert measured.deletion is False, "measured at zero lift, so it rides along"
    assert measured.deleted_inert == ["Beta"]


def test_newly_zero_weight_is_about_the_named_axis_not_the_act(tmp_path):
    baseline = _filter(tmp_path / "base", ZEROED)
    candidate = _filter(tmp_path / "cand", {"Alpha": 0.0, "Beta": 0.0, "Gamma": 3.0})
    assert newly_zero_weight_criteria(candidate, baseline) == ["Alpha"]


# --- what the first version of this check got wrong ---------------------------


DEMOTED = """
RUBRIC_CRITERIA = ["Alpha", "Beta", "Gamma"]
RUBRIC_WEIGHTS = {"Alpha": 2.0, "Beta": 0.0, "Gamma": 3.0}
EXECUTION_CRITERIA = ("Alpha", "Beta")
FLOOR_PENALTIES = {-2.0: 4.0, -1.0: 1.0}


def decide(trajectory, rubric_scores, triggered_rules):
    total = sum(rubric_scores.get(n, 0.0) * RUBRIC_WEIGHTS[n] for n in RUBRIC_CRITERIA)
    floor = min(rubric_scores[n] for n in EXECUTION_CRITERIA)
    return {'decision': 'ACCEPT', 'score': total - FLOOR_PENALTIES.get(floor, 0.0)}
"""


def _raw(directory: Path, policy: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rubric.md").write_text(
        "# Rubric\n\n"
        + "\n".join(f"## Criterion: {n}\n\n+2: good\n-2: bad\n" for n in ("Alpha", "Beta", "Gamma"))
    )
    (directory / "rules.py").write_text("def rule_one(trajectory):\n    return None\n")
    (directory / "policy.py").write_text(policy)
    (directory / "rationale.md").write_text("one mechanism\n")
    return directory


def test_a_demoted_axis_is_not_a_defect(tmp_path):
    """Zero weight plus a use elsewhere is a MECHANISM, not a dead axis.

    A policy can move `Verification Completeness` out of the weighted sum and
    leave it in `EXECUTION_CRITERIA`, where the score still drops 1 point at
    -1 and 7 at -2. Invalidating that candidate, with the remedy "remove the
    criterion", would produce a policy that raises KeyError on the tuple the
    floor still reads.
    """

    directory = _raw(tmp_path / "demoted", DEMOTED)
    assert zero_weight_criteria(directory) == []

    baseline = _filter(tmp_path / "base", LIVE)
    assert (
        check_candidate(directory, baseline_dir=baseline, initial_dir=baseline).status != "invalid"
    )


def test_a_negative_weight_does_not_blind_the_table(tmp_path):
    """`-1.0` parses as UnaryOp, not Constant. Reading only Constant let one
    negative weight hide the whole table — and a hidden BASELINE table is the
    iteration-emptying path below."""

    directory = _filter(tmp_path / "neg", {"Alpha": -1.0, "Beta": 0.0, "Gamma": 3.0})
    assert zero_weight_criteria(directory) == ["Beta"]
    assert priced_criteria(directory) == {"Alpha", "Beta", "Gamma"}


def test_one_unreadable_entry_does_not_blind_the_table(tmp_path):
    """A computed weight skips that entry, not the table."""

    directory = _raw(
        tmp_path / "computed",
        'SCALE = 2.0\nRUBRIC_CRITERIA = ["Alpha", "Beta", "Gamma"]\n'
        'RUBRIC_WEIGHTS = {"Alpha": SCALE, "Beta": 0.0, "Gamma": 3.0}\n' + DECIDE,
    )
    assert zero_weight_criteria(directory) == ["Beta"]
    assert priced_criteria(directory) == {"Beta", "Gamma"}


def test_a_baseline_the_scan_cannot_price_fires_nothing(tmp_path):
    """The fire reads the candidate's scan and the escape hatch reads the
    baseline's. If those disagree, a candidate is charged `invalid` for a
    defect it inherited AND refused the removal the reason tells it to make —
    and because the trigger is a property of the shared baseline, every
    candidate in the iteration flips at once.

    `RUBRIC_CRITERIA = list(RUBRIC_WEIGHTS)` is a real shape, and
    `_declared_in` cannot read it.
    """

    baseline = _raw(
        tmp_path / "base",
        'RUBRIC_WEIGHTS = {"Alpha": 2.0, "Beta": 0.0, "Gamma": 3.0}\n'
        "RUBRIC_CRITERIA = list(RUBRIC_WEIGHTS)\n" + DECIDE,
    )
    assert priced_criteria(baseline) == set(), "the baseline cannot be priced"

    candidate = _filter(tmp_path / "cand", ZEROED)
    assert zero_weight_criteria(candidate) == ["Beta"], "the candidate CAN be priced"
    assert newly_zero_weight_criteria(candidate, baseline) == [], "inherited, not written"
    assert (
        check_candidate(candidate, baseline_dir=baseline, initial_dir=baseline).status != "invalid"
    )
