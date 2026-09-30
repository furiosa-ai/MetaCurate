"""The growth caps and deletion floor (`constraints.py`).

Every cap is exercised against tiny filters built in `tmp_path`, because the
check reads files and nothing else: no import of candidate code, no LLM.
What matters here is that each cap fires on exactly the side of its
threshold the constraint names, that the reason string a proposer will read says
which cap and by how much, and that a merge — a way to prune axes without
the count looking like a deletion — is caught by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_filter import constraints as cx

# --- fixtures: filters small enough to state a threshold in one number ------


def _criterion(name: str, body: str = "Does it?\n\n+2: yes\n-2: no\n") -> str:
    return f"## Criterion: {name}\n\n{body}\n"


def _rubric(*names: str, header: str = "# Rubric\n\nScale +2 / -2.\n\n") -> str:
    return header + "\n".join(_criterion(n) for n in names)


def _rules(*names: str, body_lines: int = 2) -> str:
    """One `rule_*` function per name, `body_lines` non-blank statements each."""

    if not names:
        return ""
    out = ["from __future__ import annotations", ""]
    for name in names:
        out.append(f"def rule_{name}(trajectory):")
        for i in range(body_lines - 1):
            out.append(f"    x{i} = {i}")
        out.append("    return None")
        out.append("")
    return "\n".join(out) + "\n"


def _policy(lines: int = 10) -> str:
    out = ['RUBRIC_CRITERIA = ["Alpha"]']
    for i in range(lines - 1):
        out.append(f"CONST_{i} = {i}")
    return "\n".join(out) + "\n"


def make_filter(
    root: Path,
    name: str,
    *,
    criteria: tuple[str, ...] = ("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"),
    rules: tuple[str, ...] = ("one", "two", "three", "four"),
    rule_body_lines: int = 2,
    policy_lines: int = 10,
    rubric_header: str = "# Rubric\n\nScale +2 / -2.\n\n",
    rationale: str | None = None,
) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "rubric.md").write_text(_rubric(*criteria, header=rubric_header), encoding="utf-8")
    (d / "rules.py").write_text(_rules(*rules, body_lines=rule_body_lines), encoding="utf-8")
    (d / "policy.py").write_text(_policy(policy_lines), encoding="utf-8")
    if rationale is not None:
        (d / "rationale.md").write_text(rationale, encoding="utf-8")
    return d


# The production `soft_floor` gives rules.py 35 lines of headroom whatever the
# baseline, which is larger than these deliberately tiny fixtures. Tests that
# are about the SHARE — that 15 % is exact and fires one over — state it by
# switching the floor off; the floor has its own tests below, on the defaults.
SHARE_ONLY = cx.ConstraintLimits(soft_floor={})


@pytest.fixture()
def initial(tmp_path):
    return make_filter(tmp_path, "initial")


@pytest.fixture()
def baseline(tmp_path):
    """Same shape as the initial filter: iteration 1's parent."""

    return make_filter(tmp_path, "baseline")


def _check(candidate: Path, baseline: Path, initial: Path, **kwargs) -> cx.ConstraintReport:
    return cx.check_candidate(candidate, baseline_dir=baseline, initial_dir=initial, **kwargs)


# --- the happy path -----------------------------------------------------------


def test_an_unchanged_copy_is_ok_with_nothing_fired(tmp_path, baseline, initial):
    report = _check(make_filter(tmp_path, "cand"), baseline, initial)

    assert report.status == "ok"
    assert report.reasons == []
    assert report.deletion is False
    assert report.deleted == [] and report.added == []
    assert report.candidate == report.baseline == report.initial
    assert set(report.candidate) == set(cx.COMPONENTS)


def test_growth_inside_every_soft_cap_is_ok(tmp_path, baseline, initial):
    """+2 rules is exactly the allowance, which is not "exceeded".

    The baseline's rules.py is 1 import + 4 x (def, x0, return) = 13 LOC;
    six one-statement rules are 1 + 6 x (def, return) = 13 LOC too, so the
    headcount moves by the full allowance while `rules_loc` does not move
    at all, and nothing fires.
    """

    cand = make_filter(
        tmp_path,
        "cand",
        rules=("one", "two", "three", "four", "five", "six"),
        rule_body_lines=1,
    )
    report = _check(cand, baseline, initial)

    assert report.candidate["rule_count"] == 6
    assert report.candidate["rules_loc"] == report.baseline["rules_loc"] == 13
    assert report.status == "ok"
    assert report.added == ["rule_five", "rule_six"]


# --- soft caps: needs_strong_evidence, one reason per firing ------------------


def test_three_more_criteria_than_the_baseline_needs_strong_evidence(tmp_path, baseline, initial):
    cand = make_filter(
        tmp_path,
        "cand",
        criteria=("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota"),
    )
    report = _check(cand, baseline, initial)

    assert report.status == "needs_strong_evidence"
    assert "rubric_criteria 9 > baseline 6 + 2 (soft)" in report.reasons
    # rubric_chars grew too (three sections on six), so that soft cap fires
    # as well — both are listed, not just the first.
    assert any(r.startswith("rubric_chars") and "(soft" in r for r in report.reasons)
    assert report.added == ["Eta", "Theta", "Iota"]


def test_two_more_criteria_is_the_allowance_not_a_breach(tmp_path, baseline, initial):
    """The cap is `>`, not `>=`: +2 is what the soft cap grants per iteration."""

    cand = make_filter(
        tmp_path,
        "cand",
        criteria=("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"),
    )
    report = _check(cand, baseline, initial)

    assert not any(r.startswith("rubric_criteria") for r in report.reasons)


def test_three_more_rules_than_the_baseline_needs_strong_evidence(tmp_path):
    """Six rules to nine: over the +2 soft cap, exactly at the x1.5 hard cap.

    Nine one-statement rules are 1 + 9 x 2 = 19 LOC, the same as six
    two-statement ones (1 + 6 x 3), so the headcount is the only firing.
    """

    six = ("one", "two", "three", "four", "five", "six")
    initial = make_filter(tmp_path, "initial", rules=six)
    baseline = make_filter(tmp_path, "baseline", rules=six)
    cand = make_filter(tmp_path, "cand", rules=(*six, "seven", "eight", "nine"), rule_body_lines=1)
    report = _check(cand, baseline, initial)

    assert report.candidate["rules_loc"] == report.baseline["rules_loc"] == 19
    assert report.status == "needs_strong_evidence"
    assert report.reasons == ["rule_count 9 > baseline 6 + 2 (soft)"]
    assert report.added == ["rule_seven", "rule_eight", "rule_nine"]


def test_rules_loc_over_fifteen_percent_needs_strong_evidence(tmp_path, baseline, initial):
    # Baseline rules.py: 1 import + 4 x (def + x0 + return) = 13 LOC. Padding
    # each body to 3 lines gives 1 + 4 x 4 = 17 LOC, +31 %.
    cand = make_filter(tmp_path, "cand", rule_body_lines=3)
    report = _check(cand, baseline, initial, limits=SHARE_ONLY)

    assert report.baseline["rules_loc"] == 13
    assert report.candidate["rules_loc"] == 17
    assert report.status == "needs_strong_evidence"
    assert "rules_loc 17 > baseline 13 + 15% (soft, +31%)" in report.reasons
    assert report.added == [], "more lines inside the same rules is not a new rule"


def test_rules_loc_inside_fifteen_percent_is_ok(tmp_path, baseline, initial):
    """13 LOC -> 14 LOC is +7.7 %: under the cap, nothing fires."""

    cand = make_filter(tmp_path, "cand")
    rules = (cand / "rules.py").read_text(encoding="utf-8")
    (cand / "rules.py").write_text(rules + "EXTRA = 1\n", encoding="utf-8")
    report = _check(cand, baseline, initial)

    assert report.candidate["rules_loc"] == 14
    assert report.status == "ok"


def test_policy_loc_growth_is_measured_but_never_capped(tmp_path, baseline, initial):
    """A policy line costs no LLM call, and a 1.5x hard cap would throw out a
    policy-only candidate however good its ordering. Neither cap reads it;
    the breakdown still does."""

    grown = _check(make_filter(tmp_path, "grown", policy_lines=12), baseline, initial)  # +20 %
    assert grown.status == "ok"
    assert grown.reasons == []
    assert grown.candidate["policy_loc"] == 12

    tripled = _check(make_filter(tmp_path, "tripled", policy_lines=30), baseline, initial)
    assert tripled.status == "ok", "3x the initial policy: no hard cap either"
    assert tripled.candidate["policy_loc"] == 30 and tripled.initial["policy_loc"] == 10


def test_rubric_chars_over_fifteen_percent_needs_strong_evidence(tmp_path, baseline, initial):
    """A rubric that grows only in prose, with the same six criteria."""

    cand = make_filter(tmp_path, "cand", rubric_header="# Rubric\n\n" + ("padding " * 10) + "\n\n")
    report = _check(cand, baseline, initial)

    assert report.candidate["rubric_criteria"] == 6
    assert report.status == "needs_strong_evidence"
    assert report.reasons == [
        f"rubric_chars {report.candidate['rubric_chars']} > baseline "
        f"{report.baseline['rubric_chars']} + 15% (soft, +21%)"
    ]


# --- hard caps and the deletion floor: invalid ---------------------------------


def test_more_than_one_and_a_half_times_the_initial_rules_is_invalid(tmp_path, initial):
    # A baseline that has already crept to 6 rules: from there, 7 is +1 (soft
    # cap fine) but 7 > 1.5 x 4 = 6 against the initial, so the run cap fires.
    baseline = make_filter(
        tmp_path, "baseline", rules=("one", "two", "three", "four", "five", "six")
    )
    cand = make_filter(
        tmp_path,
        "cand",
        rules=("one", "two", "three", "four", "five", "six", "seven"),
        rule_body_lines=1,
    )
    report = _check(cand, baseline, initial)

    assert report.status == "invalid"
    assert "rule_count 7 > 1.5 x initial 4 (hard)" in report.reasons
    assert not any("rule_count" in r and "(soft)" in r for r in report.reasons)


def test_exactly_one_and_a_half_times_the_initial_is_still_allowed(tmp_path, initial):
    baseline = make_filter(tmp_path, "baseline", rules=("one", "two", "three", "four", "five"))
    cand = make_filter(
        tmp_path,
        "cand",
        rules=("one", "two", "three", "four", "five", "six"),
        rule_body_lines=1,
    )
    report = _check(cand, baseline, initial)

    assert not any("(hard)" in r for r in report.reasons)


def test_the_hard_cap_is_skipped_for_a_component_the_initial_filter_lacked(tmp_path):
    """1.5 x 0 is 0; a filter that starts with no rules may still gain some."""

    initial = make_filter(tmp_path, "initial", rules=())
    baseline = make_filter(tmp_path, "baseline", rules=())
    cand = make_filter(tmp_path, "cand", rules=("one",), rule_body_lines=1)
    report = _check(cand, baseline, initial, limits=SHARE_ONLY)

    assert report.initial["rule_count"] == report.initial["rules_loc"] == 0
    assert not any("(hard)" in r for r in report.reasons)
    # Growth from nothing is unbounded in relative terms, so the LOC soft cap
    # does fire, and that is the right reading: it needs evidence, not a ban.
    assert report.status == "needs_strong_evidence"


def test_dropping_two_criteria_below_the_initial_count_is_invalid(tmp_path, baseline, initial):
    cand = make_filter(tmp_path, "cand", criteria=("Alpha", "Beta", "Gamma", "Delta"))
    report = _check(cand, baseline, initial)

    assert report.status == "invalid"
    assert "rubric_criteria 4 < initial 6 - 1 (deletion floor)" in report.reasons
    assert report.deletion is True
    assert report.deleted == ["Epsilon", "Zeta"]


def test_dropping_one_criterion_is_a_deletion_but_not_a_floor_breach(tmp_path, baseline, initial):
    cand = make_filter(tmp_path, "cand", criteria=("Alpha", "Beta", "Gamma", "Delta", "Epsilon"))
    report = _check(cand, baseline, initial)

    assert report.status == "ok"
    assert report.deletion is True
    assert report.deleted == ["Zeta"]
    assert report.added == []


def test_dropping_below_three_rules_is_invalid_on_a_four_rule_initial(tmp_path, baseline, initial):
    """The rule floor is `max(3, initial - 5)`. From four rules that
    is the minimum of 3, so one deletion is allowed and two are not."""

    cand = make_filter(tmp_path, "cand", rules=("one", "two"))
    report = _check(cand, baseline, initial)

    assert report.status == "invalid"
    assert "rule_count 2 < 3 (deletion floor minimum)" in report.reasons
    assert report.deleted == ["rule_three", "rule_four"]
    one_gone = _check(
        make_filter(tmp_path, "one", rules=("one", "two", "three")), baseline, initial
    )
    assert one_gone.status == "ok" and one_gone.deletion is True


NINE_RULES = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def test_a_nine_rule_initial_may_prune_to_four_but_not_three(tmp_path):
    """Pruning 9 -> 5 rule functions would be forbidden by `initial - 1`;
    `initial - 5` allows it and stops one step later."""

    initial = make_filter(tmp_path, "initial", rules=NINE_RULES)
    baseline = make_filter(tmp_path, "baseline", rules=NINE_RULES)
    assert cx.ConstraintLimits.DEFAULT.floor_for("rule_count", 9) == 4

    five = _check(make_filter(tmp_path, "five", rules=NINE_RULES[:5]), baseline, initial)
    assert five.status == "ok" and five.deletion is True
    four = _check(make_filter(tmp_path, "four", rules=NINE_RULES[:4]), baseline, initial)
    assert four.status == "ok"
    three = _check(make_filter(tmp_path, "three", rules=NINE_RULES[:3]), baseline, initial)
    assert three.status == "invalid"
    assert "rule_count 3 < initial 9 - 5 (deletion floor)" in three.reasons


def test_the_rule_floor_never_takes_the_last_deletion_from_a_tiny_filter(tmp_path):
    """A one-rule initial filter would sit under a floor of 3 forever; the
    minimum is capped at `initial - 1`, so it may still delete its rule."""

    limits = cx.ConstraintLimits.DEFAULT
    assert limits.floor_for("rule_count", 1) == 0
    assert limits.floor_for("rule_count", 2) == 1
    assert limits.floor_for("rule_count", 3) == 2
    assert limits.floor_for("rule_count", 4) == 3
    assert limits.floor_for("rubric_criteria", 6) == 5
    assert limits.floor_for("rules_loc", 40) is None

    initial = make_filter(tmp_path, "initial", rules=("one",))
    baseline = make_filter(tmp_path, "baseline", rules=("one",))
    report = _check(make_filter(tmp_path, "cand", rules=()), baseline, initial)
    assert report.status == "ok"
    assert report.deleted == ["rule_one"]


def test_the_floor_is_measured_against_the_initial_not_the_parent(tmp_path, initial):
    """One-at-a-time pruning is how a run would farm the complexity term.

    The parent already gave up one criterion; the candidate gives up one
    more. Against the parent that is a single deletion; against the initial
    it is two, and the floor says so.
    """

    baseline = make_filter(
        tmp_path, "baseline", criteria=("Alpha", "Beta", "Gamma", "Delta", "Epsilon")
    )
    cand = make_filter(tmp_path, "cand", criteria=("Alpha", "Beta", "Gamma", "Delta"))
    report = _check(cand, baseline, initial)

    assert report.status == "invalid"
    assert report.deleted == ["Epsilon"], "names are relative to the parent"
    assert "rubric_criteria 4 < initial 6 - 1 (deletion floor)" in report.reasons


# --- deletion by name, including merges ---------------------------------------


def test_a_merge_of_two_criteria_into_one_is_a_deletion_by_name(tmp_path, baseline, initial):
    """Count says -1; the names say two gone, one new. The doc calls it a
    deletion, and the names are what let the loop enforce that."""

    cand = make_filter(
        tmp_path, "cand", criteria=("Alpha", "Beta", "Gamma", "Delta", "EpsilonZeta")
    )
    report = _check(cand, baseline, initial)

    assert report.status == "ok", "one fewer criterion is within the floor"
    assert report.deletion is True
    assert report.deleted == ["Epsilon", "Zeta"]
    assert report.added == ["EpsilonZeta"]


def test_a_rename_reads_as_one_deleted_and_one_added(tmp_path, baseline, initial):
    cand = make_filter(
        tmp_path, "cand", criteria=("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Omega")
    )
    report = _check(cand, baseline, initial)

    assert report.deletion is True
    assert report.deleted == ["Zeta"]
    assert report.added == ["Omega"]
    assert report.candidate["rubric_criteria"] == report.baseline["rubric_criteria"]


def test_a_merge_of_two_rules_into_one_is_a_deletion_by_name(tmp_path, baseline, initial):
    cand = make_filter(tmp_path, "cand", rules=("one", "two", "three_four"))
    report = _check(cand, baseline, initial)

    assert report.deletion is True
    assert report.deleted == ["rule_three", "rule_four"]
    assert report.added == ["rule_three_four"]


def test_criterion_names_follow_the_lines_complexity_counts(tmp_path):
    """Both heading forms `_count_rubric_criteria` accepts yield a name."""

    rubric = tmp_path / "rubric.md"
    rubric.write_text(
        "# R\n\n## Criterion: Alpha\n\nbody\n\ncriterion: Beta\n\n## Criterion Gamma\n\n"
        "### not a criterion\n\nCRITERION: Delta\n",
        encoding="utf-8",
    )

    assert cx.criterion_names(rubric) == ["Alpha", "Beta", "Gamma", "Delta"]


def test_rule_names_are_the_top_level_rule_functions_only(tmp_path):
    rules = tmp_path / "rules.py"
    rules.write_text(
        "def rule_a(t):\n    def rule_inner(x):\n        return x\n    return None\n\n"
        "def helper(t):\n    return None\n\n"
        "def rule_b(t):\n    return None\n",
        encoding="utf-8",
    )

    assert cx.rule_names(rules) == ["rule_a", "rule_b"]


def test_a_rules_file_that_does_not_parse_has_no_rule_names(tmp_path):
    rules = tmp_path / "rules.py"
    rules.write_text("def rule_a(:\n", encoding="utf-8")

    assert cx.rule_names(rules) == []


# --- the rationale: one mechanism per candidate --------------------------------


def test_two_hypothesis_lines_in_the_rationale_is_invalid(tmp_path, baseline, initial):
    cand = make_filter(
        tmp_path,
        "cand",
        rationale=(
            "# Candidate\n\n"
            "Hypothesis: the Beta criterion under-weights verification.\n\n"
            "- **Hypothesis**: rule_two fires on the wrong action.\n"
        ),
    )
    report = _check(cand, baseline, initial)

    assert report.mechanism_count == 2
    assert report.status == "invalid"
    assert "rationale declares 2 hypotheses; one mechanism per candidate" in report.reasons


def test_one_hypothesis_line_is_ok(tmp_path, baseline, initial):
    cand = make_filter(tmp_path, "cand", rationale="## Hypothesis: one thing\n\nDetail.\n")
    report = _check(cand, baseline, initial)

    assert report.mechanism_count == 1
    assert report.status == "ok"
    assert report.reasons == []


def test_a_rationale_without_a_hypothesis_line_is_ok_with_count_zero(tmp_path, baseline, initial):
    cand = make_filter(tmp_path, "cand", rationale="# Candidate\n\nWe changed Beta.\n")
    report = _check(cand, baseline, initial)

    assert report.mechanism_count == 0
    assert report.status == "ok"


def test_no_rationale_at_all_is_ok_with_count_none(tmp_path, baseline, initial):
    report = _check(make_filter(tmp_path, "cand"), baseline, initial)

    assert report.mechanism_count is None
    assert report.status == "ok"


def test_the_word_hypothesis_in_prose_does_not_count(tmp_path, baseline, initial):
    """Only a `Hypothesis:` label at the start of a line is a declaration."""

    cand = make_filter(
        tmp_path,
        "cand",
        rationale=(
            "Hypothesis: Beta is miscalibrated.\n\n"
            "Our earlier hypothesis: that rules mattered, was wrong. The\n"
            "alternative Hypothesis is not pursued here.\n"
        ),
    )
    report = _check(cand, baseline, initial)

    assert report.mechanism_count == 1


# --- precedence, limits, serialisation -----------------------------------------


def test_status_is_the_worst_firing_and_every_firing_is_listed(tmp_path, baseline, initial):
    """A soft breach plus a floor breach reads `invalid` with both reasons."""

    cand = make_filter(
        tmp_path,
        "cand",
        criteria=("Alpha", "Beta", "Gamma", "Delta"),  # floor: invalid
        rule_body_lines=3,  # rules.py +50 %: needs_strong_evidence
    )
    report = _check(cand, baseline, initial, limits=SHARE_ONLY)

    assert report.status == "invalid"
    assert any("(deletion floor)" in r for r in report.reasons)
    assert any(r.startswith("rules_loc") and "(soft" in r for r in report.reasons)


def test_soft_firing_alone_never_escalates_to_invalid(tmp_path, baseline, initial):
    cand = make_filter(
        tmp_path,
        "cand",
        rule_body_lines=3,
        rubric_header="# Rubric\n\n" + ("padding " * 10) + "\n\n",
    )
    report = _check(cand, baseline, initial, limits=SHARE_ONLY)

    assert report.status == "needs_strong_evidence"
    assert len(report.reasons) == 2


def test_the_limits_are_overridable(tmp_path, baseline, initial):
    tight = cx.ConstraintLimits(
        soft_add={"rubric_criteria": 0},
        hard_ratio=1.0,
        deletion_floor={"rubric_criteria": 0},
    )
    plus_one = make_filter(
        tmp_path, "plus", criteria=("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta")
    )
    minus_one = make_filter(
        tmp_path, "minus", criteria=("Alpha", "Beta", "Gamma", "Delta", "Epsilon")
    )

    assert _check(plus_one, baseline, initial, limits=tight).status == "invalid"  # 7 > 1.0 x 6
    assert _check(minus_one, baseline, initial, limits=tight).status == "invalid"  # 5 < 6 - 0
    assert _check(plus_one, baseline, initial).status == "ok"


def test_the_default_limits_are_the_documented_table():
    limits = cx.ConstraintLimits.DEFAULT

    assert limits.soft_add == {
        "rubric_criteria": 2,
        "rule_count": 2,
        "rules_loc": 0.15,
        "rubric_chars": 0.15,
    }
    assert "policy_loc" not in limits.soft_add
    assert limits.hard_ratio == 1.5
    assert limits.hard_exempt == frozenset({"policy_loc"})
    assert limits.deletion_floor == {"rubric_criteria": 1, "rule_count": 5}
    assert limits.floor_min == {"rule_count": 3}
    payload = json.loads(json.dumps(limits.to_dict()))
    assert payload["hard_ratio"] == 1.5
    assert payload["hard_exempt"] == ["policy_loc"]
    assert payload["floor_min"] == {"rule_count": 3}


def test_the_report_round_trips_through_json(tmp_path, baseline, initial):
    cand = make_filter(
        tmp_path,
        "cand",
        criteria=("Alpha", "Beta", "Gamma", "Delta", "EpsilonZeta"),
        rationale="Hypothesis: merge.\nHypothesis: and another.\n",
    )
    report = _check(cand, baseline, initial)
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["status"] == "invalid"
    assert payload["deletion"] is True
    assert payload["deleted"] == ["Epsilon", "Zeta"]
    assert payload["added"] == ["EpsilonZeta"]
    assert payload["mechanism_count"] == 2
    assert set(payload["candidate"]) == set(payload["baseline"]) == set(cx.COMPONENTS)
    assert payload["initial"]["rubric_criteria"] == 6


def test_a_missing_candidate_directory_reports_zeros_and_a_floor_breach(
    tmp_path, baseline, initial
):
    """A candidate the proposer deleted is not scoreable, and the caps say so
    too rather than raising into the run."""

    gone = tmp_path / "gone"
    assert not gone.exists()
    report = _check(gone, baseline, initial)

    assert report.candidate == dict.fromkeys(cx.COMPONENTS, 0)
    assert report.status == "invalid"
    assert report.deleted == [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
        "Zeta",
        "rule_one",
        "rule_two",
        "rule_three",
        "rule_four",
    ]


# --- the 15 % cap is integer arithmetic, so it fires exactly where it says ----


def test_soft_cap_prints_the_number_the_check_allows():
    assert cx.soft_cap(100, 0.15) == 115
    assert cx.soft_cap(3000, 0.15) == 3450
    assert cx.soft_cap(20, 0.15) == 23
    assert cx.soft_cap(13, 0.15) == 14
    # `int(3000 * 1.15)` is 3449 and `int(20 * 1.15)` may be 22: the float
    # forms printed a cap one below the one the check let through.
    assert int(3000 * 1.15) == 3449


# policy.py LOC is the one component a test can set to an exact line count,
# so the integer-arithmetic check re-caps it explicitly (the default caps it
# by nothing).
POLICY_RECAPPED = cx.ConstraintLimits(
    soft_add={**cx.ConstraintLimits.DEFAULT.soft_add, "policy_loc": 0.15}
)


@pytest.mark.parametrize(
    ("base", "at_cap", "over_cap"),
    [(100, 115, 116), (3000, 3450, 3451), (20, 23, 24)],
)
def test_a_relative_cap_is_exact_at_the_number_it_prints_and_fires_one_over(
    tmp_path, base, at_cap, over_cap
):
    baseline = make_filter(tmp_path, "baseline", policy_lines=base)
    initial = make_filter(tmp_path, "initial", policy_lines=base)

    at = make_filter(tmp_path, "at", policy_lines=at_cap)
    over = make_filter(tmp_path, "over", policy_lines=over_cap)

    at_report = _check(at, baseline, initial, limits=POLICY_RECAPPED)
    assert at_report.candidate["policy_loc"] == at_cap
    assert not any(r.startswith("policy_loc") for r in at_report.reasons), at_report.reasons

    over_report = _check(over, baseline, initial, limits=POLICY_RECAPPED)
    assert over_report.status == "needs_strong_evidence"
    expected = f"policy_loc {over_cap} > baseline {base} + 15% (soft"
    assert any(r.startswith(expected) for r in over_report.reasons), over_report.reasons
    assert _check(over, baseline, initial).status == "ok", "the default does not cap policy_loc"


# --- the explicit run cap (`--complexity-ceiling`) ----------------------------
#
# The hard cap is a ratio of the initial filter, so a run started from a
# deliberately small filter would be held to a small multiple of small for its
# whole length: a one-criterion seed puts its ceiling at 1.5 and makes the most
# ordinary proposal there is — add a criterion — `invalid` in iteration 0. An
# explicit cap replaces the ratio, and moves nothing else.


def test_a_one_criterion_seed_refuses_the_first_candidate_that_grows_it(tmp_path):
    seed = make_filter(tmp_path, "seed", criteria=("Alpha",), rules=())
    cand = make_filter(tmp_path, "cand", criteria=("Alpha", "Beta"), rules=())

    report = _check(cand, seed, seed)

    assert report.status == "invalid"
    assert any("rubric_criteria 2 > 1.5 x initial 1 (hard)" in r for r in report.reasons)


CAPS = {"rubric_criteria": 9, "rule_count": 13, "rules_loc": 522, "rubric_chars": 8785}


def test_an_explicit_cap_lifts_that_refusal(tmp_path):
    seed = make_filter(tmp_path, "seed", criteria=("Alpha",), rules=())
    cand = make_filter(tmp_path, "cand", criteria=("Alpha", "Beta"), rules=())

    report = _check(cand, seed, seed, hard_caps=CAPS)

    # The hard refusal is gone. The per-iteration SOFT cap on rubric_chars is
    # not, and must not be: it is measured against the parent and raises the
    # acceptance bar rather than refusing the candidate.
    assert report.status == "needs_strong_evidence"
    assert not any("hard" in r for r in report.reasons)
    assert all("rubric_chars" in r and "soft" in r for r in report.reasons)


def test_an_explicit_cap_still_refuses_above_its_number(tmp_path):
    seed = make_filter(tmp_path, "seed", criteria=("Alpha",), rules=())
    ten = tuple(f"C{i}" for i in range(10))
    cand = make_filter(tmp_path, "cand", criteria=ten, rules=())

    report = _check(cand, seed, seed, hard_caps=CAPS)

    assert report.status == "invalid"
    assert any("rubric_criteria 10 > run cap 9 (hard)" in r for r in report.reasons)


def test_an_explicit_cap_does_not_move_the_deletion_floor(tmp_path):
    """The floor keeps reading the INITIAL filter."""

    seed = make_filter(tmp_path, "seed", criteria=("Alpha",), rules=())
    cand = make_filter(tmp_path, "cand", criteria=("Alpha",), rules=())

    report = _check(cand, seed, seed, hard_caps=CAPS)

    assert report.status == "ok"
    assert not any("floor" in r for r in report.reasons)


# --- the absolute floor under a relative cap --------------------------------


def test_a_tiny_rules_file_still_has_room_for_two_rule_bodies(tmp_path):
    """A baseline whose rules.py the loop had emptied would get
    `2 * 15 // 100 == 0` extra lines, so every
    candidate that wrote a rule was over the soft cap — while `rule_count`
    allowed +2 rules in the same breath."""

    empty = _rules()
    base = make_filter(tmp_path, "base")
    (base / "rules.py").write_text(empty, encoding="utf-8")
    initial_dir = make_filter(tmp_path, "init")

    limits = cx.ConstraintLimits.DEFAULT
    assert limits.soft_max_for("rules_loc", 0) == 35
    assert limits.soft_max_for("rules_loc", 2) == 37

    # One median-sized rule body (16 lines) from an emptied rules.py.
    cand = make_filter(tmp_path, "cand", rules=("only",), rule_body_lines=16)
    report = cx.check_candidate(cand, baseline_dir=base, initial_dir=initial_dir)

    assert report.candidate["rules_loc"] <= 35
    assert not any(r.startswith("rules_loc") for r in report.reasons), report.reasons


def test_the_share_takes_over_once_it_is_worth_more_than_the_floor(tmp_path):
    limits = cx.ConstraintLimits.DEFAULT

    assert limits.soft_max_for("rules_loc", 233) == 268  # floor: 233 + 35
    assert limits.soft_max_for("rules_loc", 234) == 269  # share: 234 + 35 (234*15//100)
    assert limits.soft_max_for("rules_loc", 3000) == 3450  # share alone


def test_the_floor_says_so_when_it_is_the_thing_that_bound(tmp_path):
    # A roomy initial filter so the run hard cap is not what fires, and three
    # rules so the deletion floor's minimum of 3 is met; `rule_count` fires
    # soft alongside, which is the same status.
    initial_dir = make_filter(tmp_path, "init", rule_body_lines=8)
    base = make_filter(tmp_path, "base")
    (base / "rules.py").write_text(_rules(), encoding="utf-8")
    cand = make_filter(tmp_path, "cand", rules=("a", "b", "c"), rule_body_lines=16)

    report = cx.check_candidate(cand, baseline_dir=base, initial_dir=initial_dir)

    assert report.candidate["rules_loc"] > 35

    assert report.status == "needs_strong_evidence"
    assert any("the floor under 15%" in r for r in report.reasons), report.reasons


def test_the_proposer_prompt_never_promises_rules_it_has_no_room_for(tmp_path):
    """`+2 rules (to 2), rules.py to 2 LOC` is the contradiction a proposer
    prompt would print. Two rules must fit in the lines the same prompt allows."""

    limits = cx.ConstraintLimits.DEFAULT
    median_rule_body = 16

    for base_loc in (0, 2, 9, 13, 185, 212):
        room = limits.soft_max_for("rules_loc", base_loc) - base_loc
        added_rules = limits.soft_max_for("rule_count", 1) - 1
        assert room >= added_rules * median_rule_body, (base_loc, room, added_rules)


# --- policy.py composes; it does not grade ---------------------------------

_COMPOSING_POLICY = """
RUBRIC_CRITERIA = ["Alpha"]


def decide(trajectory, rubric_scores, triggered_rules):
    return {"decision": "ACCEPT", "score": 50.0 + 20.0 * rubric_scores.get("Alpha", 0.0)}
"""

_GRADING_POLICY = """
RUBRIC_CRITERIA = ["Alpha"]


def _post_edit_tests(trajectory) -> int:
    turns = getattr(trajectory, "turns", []) or []
    return sum("pytest" in str(getattr(t, "content", "")) for t in turns)


def decide(trajectory, rubric_scores, triggered_rules):
    score = 50.0 + 20.0 * rubric_scores.get("Alpha", 0.0)
    return {"decision": "ACCEPT", "score": score + 4.0 * _post_edit_tests(trajectory)}
"""


def _with_policy(tmp_path, name: str, policy: str) -> Path:
    d = make_filter(tmp_path, name)
    (d / "policy.py").write_text(policy, encoding="utf-8")
    return d


def test_a_policy_that_only_composes_is_fine(tmp_path, baseline, initial):
    cand = _with_policy(tmp_path, "composing", _COMPOSING_POLICY)

    assert cx.policy_reads_trajectory(cand) is None
    assert _check(cand, baseline, initial).status == "ok"


def test_a_policy_that_grades_the_trajectory_is_invalid(tmp_path, baseline, initial):
    """An empty rules.py and a policy.py that walks the turns. `policy_loc` is the one uncapped
    component, so this is deterministic logic bought outside every budget."""

    cand = _with_policy(tmp_path, "grading", _GRADING_POLICY)

    reason = cx.policy_reads_trajectory(cand)
    assert reason is not None and "belongs in rules.py" in reason

    report = _check(cand, baseline, initial)
    assert report.status == "invalid"
    assert any("reads the trajectory in decide()" in r for r in report.reasons), report.reasons


def test_every_shipped_seed_composes():
    filters = Path(__file__).resolve().parents[1] / "filters"
    seeds = [s for s in sorted(filters.glob("initial_*")) if (s / "policy.py").exists()]
    assert seeds, "no shipped filters found"

    graders = {s.name for s in seeds if cx.policy_reads_trajectory(s) is not None}
    assert graders == set(), graders
