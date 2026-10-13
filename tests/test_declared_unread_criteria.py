"""A criterion declared in RUBRIC_CRITERIA that `decide()` never reads.

The solver buys one LLM call per (row, criterion in `RUBRIC_CRITERIA`). A
criterion the policy declares and never reads is therefore paid for on every
row and cannot move the ordering by a single pair — the purest waste the
rubric can express, and invisible in every number the loop reports.

The fixture policy's decide() is `30 + Task Grounding*10 + Edit
Justification*6 + Verification Completeness*2 + Error Recovery`: Query/PR
Quality and Gold-Patch Relation were dropped from the arithmetic and left in
the declaration, so the scorer keeps buying two cells per row for nothing.

Two things are pinned: what the detector says about real policies (the
initial filter, which takes a mean over the whole mapping, must come back
empty), and WHO pays — introducing the mismatch is `invalid`, inheriting one
fires nothing at all.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from meta_filter.diagnostics.criterion_stats import (
    DEAD_AXIS_ACTION,
    REMOVAL_LICENCE,
    UNREAD_AXIS_ACTION,
    criterion_stats,
    render_criterion_health,
    render_inert_axes,
)
from meta_filter.gate import constraints as cx
from meta_filter.gate.precheck import PrecheckResult, render_precheck_block
from meta_filter.propose.proposer import _format_loop_telemetry, render_edit_discipline

REPO = Path(__file__).resolve().parents[1]
FIXTURE_FILTERS = Path(__file__).parent / "fixtures" / "filters"

CRITERIA = (
    "Task Grounding",
    "Query/PR Quality",
    "Gold-Patch Relation",
    "Edit Justification",
    "Verification Completeness",
    "Error Recovery",
)


# --- filters small enough that the policy is the only thing under test -------


def _rubric(*names: str) -> str:
    body = "\n".join(f"## Criterion: {n}\n\nDoes it?\n\n+2: yes\n-2: no\n" for n in names)
    return "# Rubric\n\nScale +2 / -2.\n\n" + body


_RULES = """from __future__ import annotations


def rule_only(trajectory):
    return None
"""


def _weights_policy(read: tuple[str, ...], declared: tuple[str, ...] = CRITERIA) -> str:
    """A weights dict keyed by criterion name, applied one cell at a time.

    The shape the check has to get right in both directions: the names are
    string literals in a MODULE constant, not in `decide()`, and the cell
    lookup inside the helper is keyed by a loop variable. Neither is a
    generic read, and every name in `read` is read.
    """

    entries = "\n".join(f'    "{name}": 1.0,' for name in read)
    declaration = "\n".join(f'    "{name}",' for name in declared)
    return f"""from __future__ import annotations

RUBRIC_CRITERIA = [
{declaration}
]

WEIGHTS = {{
{entries}
}}


def _criterion(rubric_scores, name):
    return float((rubric_scores or {{}}).get(name, 0.0))


def decide(trajectory, rubric_scores, triggered_rules):
    score = 30.0
    for name, weight in WEIGHTS.items():
        score += _criterion(rubric_scores, name) * weight
    return {{"decision": "ACCEPT", "score": score, "confidence": 0.5,
            "reasoning_summary": "", "evidence": []}}
"""


def make_filter(
    root: Path,
    name: str,
    *,
    policy: str,
    criteria: tuple[str, ...] = CRITERIA,
) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "rubric.md").write_text(_rubric(*criteria), encoding="utf-8")
    (d / "rules.py").write_text(_RULES, encoding="utf-8")
    (d / "policy.py").write_text(policy, encoding="utf-8")
    return d


def _check(candidate: Path, baseline: Path, initial: Path | None = None):
    return cx.check_candidate(candidate, baseline_dir=baseline, initial_dir=initial or baseline)


# --- the two real policies ----------------------------------------------------


def test_the_initial_filter_reads_every_criterion_through_its_mean():
    """`filters/initial_ranking` averages `rubric_scores.values()`.

    Nothing there names a criterion, so a literal scan alone would call all
    six unread and hand the proposer six deletions to make. The generic-use
    valve is what stops it, and this is the case it exists for.
    """

    initial = REPO / "tests" / "fixtures" / "filters" / "six_criteria"

    assert cx.declared_criteria(initial) == list(CRITERIA)
    assert cx.declared_unread_criteria(initial) == []


def test_a_policy_that_stopped_reading_two_criteria_is_caught(tmp_path):
    """A filter that keeps paying for two axes it has stopped using."""

    d = make_filter(
        tmp_path,
        "two_unread",
        policy=(FIXTURE_FILTERS / "policy_two_unread_criteria.py.txt").read_text(encoding="utf-8"),
    )

    assert cx.declared_criteria(d) == list(CRITERIA)
    # In declaration order, so the list reads against rubric.md.
    assert cx.declared_unread_criteria(d) == ["Query/PR Quality", "Gold-Patch Relation"]


def test_a_weights_dict_policy_that_reads_every_criterion_is_clean(tmp_path):
    d = make_filter(tmp_path, "weights", policy=_weights_policy(CRITERIA))

    assert cx.declared_unread_criteria(d) == []


def test_a_weights_dict_that_drops_one_key_leaves_that_criterion_unread(tmp_path):
    read = tuple(n for n in CRITERIA if n != "Query/PR Quality")
    d = make_filter(tmp_path, "weights", policy=_weights_policy(read))

    assert cx.declared_unread_criteria(d) == ["Query/PR Quality"]


# --- a key the scan cannot enumerate is not evidence of anything ------------


def _keyed_by(prelude: str, iterable: str) -> str:
    """A policy that reads all six criteria through a helper, keyed by a name
    drawn from `iterable`. Whether that is a real read or an unanswerable
    question is exactly what `prelude` decides.
    """

    declaration = "\n".join(f'    "{n}",' for n in CRITERIA)
    return (
        "from __future__ import annotations\n\n"
        f"RUBRIC_CRITERIA = [\n{declaration}\n]\n\n\n"
        f"{prelude}"
        "def _criterion(rubric_scores, name):\n"
        "    return float((rubric_scores or {}).get(name, 0.0))\n\n\n"
        "def decide(trajectory, rubric_scores, triggered_rules):\n"
        "    score = 30.0\n"
        f"    for name in {iterable}:\n"
        "        score += _criterion(rubric_scores, name)\n"
        '    return {"decision": "ACCEPT", "score": score}\n'
    )


_UNENUMERABLE_KEYS = {
    # Dropping `_axes` silently and keeping going would report the six names
    # the scan could not read as six findings.
    "a module constant built by a call": (
        "def _axes():\n    return list(RUBRIC_CRITERIA)\n\n\nAXES = _axes()\n\n\n",
        "AXES",
    ),
    # The same hole one step further out: nothing in this module holds the
    # names at all.
    "an imported key list": ("from rules import AXES  # noqa: F401\n\n\n", "AXES"),
    # And through a local, which the module-constant walk never sees.
    "a local built from the trajectory": ("", "trajectory.metadata['axes']"),
}


@pytest.mark.parametrize("label", sorted(_UNENUMERABLE_KEYS))
def test_a_key_the_scan_cannot_enumerate_reports_nothing(tmp_path, label):
    """A policy that reads every criterion must not be flagged on every criterion.

    The helper forwards whatever its caller passed, and the scan trusted the
    forwarded parameter because "the call site's literals are collected from
    the same reachable code". They are not, when the call site's key came from
    somewhere the walk never read: then `literals` is empty, nothing looks
    generic, and all six declared criteria are reported unread. That is a
    proposer being told to delete the whole rubric.
    """

    prelude, iterable = _UNENUMERABLE_KEYS[label]
    d = make_filter(
        tmp_path, f"unresolvable_{label.replace(' ', '_')}", policy=_keyed_by(prelude, iterable)
    )

    assert cx.declared_unread_criteria(d) == []


def test_the_same_shape_over_an_enumerable_list_still_reads_every_criterion(tmp_path):
    """The trap half: `AXES = list(RUBRIC_CRITERIA)` was always clean, which is
    what made the call form a trap rather than a uniform blind spot."""

    d = make_filter(
        tmp_path, "resolvable", policy=_keyed_by("AXES = list(RUBRIC_CRITERIA)\n\n\n", "AXES")
    )

    assert cx.declared_unread_criteria(d) == []


def test_a_comprehension_key_is_scoped_to_its_comprehension(tmp_path):
    """A policy can key a helper from a comprehension over a weights dict
    while the enclosing `decide()` rebinds the same spelling to a rule name.
    A comprehension has its own scope, and folding the two
    together would let the rule name — which this scan cannot enumerate —
    disqualify a key it can.
    """

    declaration = "\n".join(f'    "{n}",' for n in CRITERIA)
    entries = "\n".join(f'    "{n}": 1.0,' for n in CRITERIA if n != "Query/PR Quality")
    policy = (
        "from __future__ import annotations\n\n"
        f"RUBRIC_CRITERIA = [\n{declaration}\n]\n\n"
        f"WEIGHTS = {{\n{entries}\n}}\n\n\n"
        "def _criterion(rubric_scores, name):\n"
        "    return float((rubric_scores or {}).get(name, 0.0))\n\n\n"
        "def decide(trajectory, rubric_scores, triggered_rules):\n"
        "    evidence = []\n"
        "    for rule in triggered_rules or []:\n"
        '        name = rule.get("name", "?")\n'
        "        evidence.append(name)\n"
        "    values = [_criterion(rubric_scores, name) for name in WEIGHTS]\n"
        '    return {"decision": "ACCEPT", "score": 30.0 + sum(values)}\n'
    )
    d = make_filter(tmp_path, "comprehension", policy=policy)

    assert cx.declared_unread_criteria(d) == ["Query/PR Quality"]


# --- the valve: anything the scan cannot follow reads everything --------------


_GENERIC_BODIES = {
    "iterates the mapping": "    for value in rubric_scores:\n        score += 1.0\n",
    "values": "    score += sum(rubric_scores.values())\n",
    "items": "    for _k, v in rubric_scores.items():\n        score += v\n",
    "keys": "    score += len(list(rubric_scores.keys()))\n",
    "handed to a builtin": "    score += len(rubric_scores)\n",
    # A key drawn from somewhere this scan never read. `('a', 'b')` would not
    # belong here: those two keys ARE literals in `decide()`, so the scan can
    # enumerate them, and the right answer for
    # that policy is that all six declared criteria go unread.
    "key the scan cannot enumerate": (
        "    for name in trajectory.metadata['axes']:\n        score += rubric_scores[name]\n"
    ),
    "enumerates the declaration": (
        "    for name in RUBRIC_CRITERIA:\n        score += rubric_scores.get(name, 0.0)\n"
    ),
}


@pytest.mark.parametrize("label", sorted(_GENERIC_BODIES))
def test_generic_consumption_makes_every_declared_criterion_read(tmp_path, label):
    declaration = "\n".join(f'    "{n}",' for n in CRITERIA)
    policy = (
        "from __future__ import annotations\n\n"
        f"RUBRIC_CRITERIA = [\n{declaration}\n]\n\n\n"
        "def decide(trajectory, rubric_scores, triggered_rules):\n"
        "    score = 30.0\n"
        f"{_GENERIC_BODIES[label]}"
        '    return {"decision": "ACCEPT", "score": score}\n'
    )
    d = make_filter(tmp_path, f"generic_{label.replace(' ', '_')}", policy=policy)

    assert cx.declared_unread_criteria(d) == []


def _helper_policy(call: str) -> str:
    declaration = "\n".join(f'    "{n}",' for n in CRITERIA)
    reads = "\n".join(f"    score += {call.format(name=n)}" for n in CRITERIA)
    return (
        "from __future__ import annotations\n\n"
        f"RUBRIC_CRITERIA = [\n{declaration}\n]\n\n\n"
        "def _criterion(rubric_scores, name):\n"
        "    return float((rubric_scores or {}).get(name, 0.0))\n\n\n"
        "def decide(trajectory, rubric_scores, triggered_rules):\n"
        "    score = 30.0\n"
        f"{reads}\n"
        '    return {"decision": "ACCEPT", "score": score}\n'
    )


def test_a_helper_called_by_keyword_still_resolves_its_literal_key(tmp_path):
    """`_criterion(rubric_scores=..., name="X")` reads X like the positional form."""

    d = make_filter(
        tmp_path,
        "kwargs",
        policy=_helper_policy('_criterion(rubric_scores=rubric_scores, name="{name}")'),
    )

    assert cx.declared_unread_criteria(d) == []


def test_unpacking_the_mapping_into_a_helper_is_generic(tmp_path):
    """`_criterion(*rubric_scores)` spreads the keys; nothing here is one cell."""

    declaration = "\n".join(f'    "{n}",' for n in CRITERIA)
    policy = (
        "from __future__ import annotations\n\n"
        f"RUBRIC_CRITERIA = [\n{declaration}\n]\n\n\n"
        "def _criterion(a, b):\n"
        "    return float(a) + float(b)\n\n\n"
        "def decide(trajectory, rubric_scores, triggered_rules):\n"
        '    return {"score": _criterion(*rubric_scores)}\n'
    )
    d = make_filter(tmp_path, "starred", policy=policy)

    assert cx.declared_unread_criteria(d) == []


def test_an_unreadable_policy_says_nothing(tmp_path):
    """No decide, no parse, no declaration — three ways to have no opinion."""

    no_decide = make_filter(tmp_path, "no_decide", policy='RUBRIC_CRITERIA = ["Task Grounding"]\n')
    broken = make_filter(tmp_path, "broken", policy="def decide(a, b, c:\n")
    computed = make_filter(
        tmp_path,
        "computed",
        policy=(
            "RUBRIC_CRITERIA = [n for n in ('Task Grounding',)]\n\n\n"
            "def decide(trajectory, rubric_scores, triggered_rules):\n"
            '    return {"score": 1.0}\n'
        ),
    )

    assert cx.declared_unread_criteria(no_decide) == []
    assert cx.declared_unread_criteria(broken) == []
    assert cx.declared_unread_criteria(computed) == []


# --- which parameter is the mapping -------------------------------------------


def _mapping_policy(signature: str, body: str) -> str:
    declaration = "\n".join(f'    "{n}",' for n in CRITERIA)
    return (
        "from __future__ import annotations\n\n"
        f"RUBRIC_CRITERIA = [\n{declaration}\n]\n\n\n"
        f"def decide({signature}):\n"
        f"{body}"
    )


def test_a_keyword_only_mapping_is_still_the_mapping(tmp_path):
    """Falling back to the second POSITIONAL parameter would guess
    `triggered_rules` as the mapping of a `decide()` that takes
    `rubric_scores` by keyword. Nothing would then look like a generic read of
    the real mapping, and a policy that averages all six criteria would be
    reported as reading none of them.
    """

    d = make_filter(
        tmp_path,
        "kwonly",
        policy=_mapping_policy(
            "trajectory, triggered_rules, *, rubric_scores",
            "    score = 30.0 + sum((rubric_scores or {}).values())\n"
            '    return {"decision": "ACCEPT", "score": score}\n',
        ),
    )

    assert cx.declared_unread_criteria(d) == []


def test_a_keyword_only_mapping_is_still_read_criterion_by_criterion(tmp_path):
    """...and the finding is still available when it is a real one."""

    reads = "".join(
        f'    score += float((rubric_scores or {{}}).get("{n}", 0.0))\n'
        for n in CRITERIA
        if n != "Query/PR Quality"
    )
    d = make_filter(
        tmp_path,
        "kwonly_partial",
        policy=_mapping_policy(
            "trajectory, triggered_rules, *, rubric_scores",
            f'    score = 30.0\n{reads}    return {{"decision": "ACCEPT", "score": score}}\n',
        ),
    )

    assert cx.declared_unread_criteria(d) == ["Query/PR Quality"]


def test_a_second_parameter_decide_never_loads_is_not_guessed_at(tmp_path):
    """The guess is only worth taking when the body uses what was guessed.

    With no parameter this scan can show to be the mapping there is no read to
    call generic, so every criterion whose name is not a literal would be
    reported unread — a wrong guess is strictly worse here than no answer.
    """

    d = make_filter(
        tmp_path,
        "no_mapping",
        policy=_mapping_policy(
            "trajectory, unused, third",
            '    return {"decision": "ACCEPT", "score": 30.0 + float(third or 0)}\n',
        ),
    )

    assert cx.declared_unread_criteria(d) == []


def test_a_renamed_mapping_is_taken_when_decide_actually_reads_it(tmp_path):
    """`solver.py` calls `decide` positionally, so a renamed second parameter
    really is the mapping — the fixtures in this repo's own tests use
    `scores`. The guess stays; only the unused guess goes.
    """

    d = make_filter(
        tmp_path,
        "renamed",
        policy=_mapping_policy(
            "trajectory, scores, triggered_rules",
            "    score = 30.0 + sum((scores or {}).values())\n"
            '    return {"decision": "ACCEPT", "score": score}\n',
        ),
    )

    assert cx.declared_unread_criteria(d) == []


# --- who pays -----------------------------------------------------------------


def test_introducing_an_unread_criterion_is_invalid(tmp_path):
    baseline = make_filter(tmp_path, "baseline", policy=_weights_policy(CRITERIA))
    candidate = make_filter(
        tmp_path,
        "cand",
        policy=_weights_policy(tuple(n for n in CRITERIA if n != "Query/PR Quality")),
    )

    report = _check(candidate, baseline, baseline)

    assert report.status == "invalid"
    assert (
        "policy.py no longer reads declared criterion Query/PR Quality — remove it "
        "from RUBRIC_CRITERIA and rubric.md, or read it"
    ) in report.reasons
    assert report.declared_unread == ["Query/PR Quality"]


def test_inheriting_an_unread_criterion_fires_nothing(tmp_path):
    """The defect is the parent's; the candidate is only told about it.

    A run whose initial filter already carries an unread axis would otherwise
    hold every candidate in the run to the stricter bar for something none of
    them did.
    """

    unread = _weights_policy(tuple(n for n in CRITERIA if n != "Query/PR Quality"))
    baseline = make_filter(tmp_path, "baseline", policy=unread)
    candidate = make_filter(tmp_path, "cand", policy=unread)

    report = _check(candidate, baseline, baseline)

    assert report.status == "ok"
    assert report.reasons == []
    # Reported all the same: silence here would let a run carry them forever.
    assert report.declared_unread == ["Query/PR Quality"]


def test_removing_the_declaration_and_the_rubric_section_is_clean(tmp_path):
    """The fix the prompt asks for: gone from policy.py AND from rubric.md.

    It has to come out clean and it has to keep the simplification licence,
    or the loop would be telling a proposer to make an edit its own gate
    then charges for.
    """

    kept = tuple(n for n in CRITERIA if n != "Query/PR Quality")
    baseline = make_filter(tmp_path, "baseline", policy=_weights_policy(CRITERIA))
    candidate = make_filter(
        tmp_path,
        "cand",
        policy=_weights_policy(kept, declared=kept),
        criteria=kept,
    )

    report = _check(candidate, baseline, baseline)

    assert report.status == "ok"
    assert report.reasons == []
    assert report.declared_unread == []
    assert report.deleted == ["Query/PR Quality"]
    assert report.simplification is True


def test_a_criterion_the_baseline_did_not_declare_is_not_an_introduction(tmp_path):
    """`no longer reads` means what it says: the baseline has to have read it.

    A candidate that adds an axis is bounded by the growth caps and the new
    dead-axis rule; this check is about the arithmetic quietly walking away
    from a criterion that stays declared.
    """

    baseline = make_filter(tmp_path, "baseline", policy=_weights_policy(CRITERIA))
    grown = CRITERIA + ("Regression Risk",)
    candidate = make_filter(
        tmp_path,
        "cand",
        policy=_weights_policy(CRITERIA, declared=grown),
        criteria=grown,
    )

    report = _check(candidate, baseline, baseline)

    assert report.declared_unread == ["Regression Risk"]
    assert not any("no longer reads" in reason for reason in report.reasons)


# --- the readouts -------------------------------------------------------------


def _wide_stats(n_criteria: int):
    """Enough live axes that the block overflows a tight character budget."""

    columns = {
        f"Criterion Number {k}": [float((i * (k + 1)) % 5 - 2) for i in range(50)]
        for k in range(n_criteria)
    }
    return criterion_stats(
        [SimpleNamespace(rubric_scores={c: v[i] for c, v in columns.items()}) for i in range(50)]
    )


def _stats_with_dead_axis():
    columns = {
        "Task Grounding": [float(i % 5 - 2) for i in range(50)],
        "Error Recovery": [float((i * 3) % 5 - 2) for i in range(50)],
        "Gold-Patch Relation": [0.0] * 50,
    }
    return criterion_stats(
        [SimpleNamespace(rubric_scores={c: v[i] for c, v in columns.items()}) for i in range(50)]
    )


def test_the_report_carries_the_dead_axes_it_measured(tmp_path):
    baseline = make_filter(tmp_path, "baseline", policy=_weights_policy(CRITERIA))
    candidate = make_filter(tmp_path, "cand", policy=_weights_policy(CRITERIA))
    report = _check(candidate, baseline, baseline)
    assert report.dead_axes == []

    cx.assess_rubric_health(report, _stats_with_dead_axis(), _stats_with_dead_axis())

    # Inherited, so nothing fires — and it is still named.
    assert report.dead_axes == ["Gold-Patch Relation"]
    assert report.new_dead_axes == []
    assert report.status == "ok"
    assert report.to_dict()["dead_axes"] == ["Gold-Patch Relation"]


def test_the_precheck_block_names_both_readouts_under_an_ok_candidate():
    constraint = cx.ConstraintReport(
        status="ok",
        reasons=[],
        deletion=False,
        deleted=[],
        added=[],
        mechanism_count=1,
        candidate={},
        baseline={},
        initial={},
        declared_unread=["Query/PR Quality", "Gold-Patch Relation"],
        dead_axes=["Gold-Patch Relation"],
    )
    block = render_precheck_block({"c0": PrecheckResult(candidate_id="c0", constraint=constraint)})

    assert "declared but never read by policy.py: Query/PR Quality, Gold-Patch Relation" in block
    assert "one scorer call per row each, and no effect on the ordering" in block
    assert "dead axes (one level on almost every row): Gold-Patch Relation" in block


def test_the_precheck_block_names_them_on_a_candidate_that_does_not_load():
    constraint = cx.ConstraintReport(
        status="ok",
        reasons=[],
        deletion=False,
        deleted=[],
        added=[],
        mechanism_count=1,
        candidate={},
        baseline={},
        initial={},
        declared_unread=["Query/PR Quality"],
    )
    block = render_precheck_block(
        {
            "c0": PrecheckResult(
                candidate_id="c0", error="policy.py: SyntaxError", constraint=constraint
            )
        }
    )

    assert "NOT SCOREABLE" in block
    assert "declared but never read by policy.py: Query/PR Quality" in block


def test_the_health_block_names_a_dead_axis_and_states_both_options():
    text = render_criterion_health(_stats_with_dead_axis(), threshold=0.7)

    assert "Gold-Patch Relation: 100% of scores sit at +0 — DEAD AXIS" in text
    assert "- dead: Gold-Patch Relation" in text
    assert DEAD_AXIS_ACTION in text
    # Both options, named: rewrite the anchors, or remove it.
    assert "rewrite its anchors so they separate these trajectories" in text
    assert "judge the defect from the issue, the diff and the observed checks" in text
    assert "Or remove it from rubric.md and RUBRIC_CRITERIA" in text
    assert REMOVAL_LICENCE in text


def test_the_health_block_names_an_unread_criterion_the_scores_cannot_show():
    """Live scores, spread over five levels, and the policy ignores them.

    Nothing in `criterion_stats` can see this: the axis separates the pool
    perfectly and the ordering never consults it. Only policy.py knows.
    """

    text = render_criterion_health(
        _stats_with_dead_axis(), threshold=0.7, declared_unread=["Error Recovery"]
    )

    assert "Error Recovery" in text
    assert "DECLARED BUT UNREAD: policy.py lists it in RUBRIC_CRITERIA" in text
    assert "- dead: Gold-Patch Relation" in text
    assert "- declared but unread: Error Recovery" in text
    assert text.count(REMOVAL_LICENCE) == 1, "one licence, however many axes"


def test_an_unread_axis_is_told_to_read_it_not_to_rewrite_it():
    """The repair differs by defect, and the wrong one costs a whole edit.

    A declared-but-unread axis already separates trajectories — its scores
    spread over several levels — so "rewrite it so it separates trajectories" is
    an instruction to fix the half that is not broken, and the rewritten axis
    goes on costing a scorer call per row because `decide()` still does not
    look. The cheap repair is the one the invalid reason has always named:
    read it. It is free, `policy_loc` being in neither growth cap.
    """

    text = render_criterion_health(
        _stats_with_dead_axis(),
        threshold=0.7,
        declared_unread=["Error Recovery"],
    )
    unread_line = next(line for line in text.splitlines() if "declared but unread:" in line)
    dead_line = next(line for line in text.splitlines() if "- dead:" in line)

    assert "read it in decide()" in unread_line
    assert "reading it is free" in unread_line
    assert "Rewriting its anchors fixes nothing here" in unread_line
    # ...and the dead axis is still told the opposite thing, which is right
    # for it: one level on every row IS an anchor defect.
    assert "rewrite its anchors" in dead_line
    assert "read it in decide()" not in dead_line


def test_the_removal_option_never_asks_for_more_than_the_floor_allows():
    """`remove it` for two axes, on a six-criterion filter with a floor of
    five, is a candidate the precheck rejects as
    `rubric_criteria 4 < initial 6 - 1 (deletion floor)`, so the block names
    the floor.
    """

    line = render_inert_axes(["Gold-Patch Relation"], ["Query/PR Quality"], removals=1, floor=5)

    assert "remove 1 of them (never below 5 criteria)" in line
    assert "for each of them" not in line.lower()


def test_at_the_floor_removal_is_not_offered_at_all():
    line = render_inert_axes(["Gold-Patch Relation"], ["Query/PR Quality"], removals=0, floor=5)

    assert "remove it from rubric.md" not in line
    assert "none of them may be removed" in line
    assert "deletion floor (5 criteria) is already reached" in line
    assert REMOVAL_LICENCE not in line
    # The repairs that ARE available are still stated.
    assert "rewrite its anchors" in line
    assert "read it in decide()" in line


def test_the_removals_left_reach_the_block_from_the_caller():
    """The count is the caller's, not `n_criteria`: `stats` counts what was
    SCORED — the policy's declaration — while the deletion floor is checked
    against rubric.md, and the two need not agree."""

    stats = _stats_with_dead_axis()

    assert "remove 2 of them (never below 4 criteria)" in render_criterion_health(
        stats, threshold=0.7, removals_left=2, criteria_floor=4
    )
    assert "none of them may be removed" in render_criterion_health(
        stats, threshold=0.7, removals_left=0, criteria_floor=6
    )
    # Without a floor the option is stated, uncounted.
    assert REMOVAL_LICENCE in render_criterion_health(stats, threshold=0.7)


def test_the_licence_is_worded_the_way_the_proposer_prompt_words_it():
    """ "accepted whenever it loses no selection pairs and does
    not use the iteration's one deletion slot" reads as two acceptance
    conditions, the second of which a removal plainly violates. The
    edit-discipline block has always had the unambiguous form.
    """

    discipline = render_edit_discipline(
        n_candidates=5,
        parent_counts={
            "rubric_criteria": 6,
            "rule_count": 5,
            "rules_loc": 100,
            "policy_loc": 80,
            "rubric_chars": 3000,
        },
        initial_counts=None,
        candidates_scored=None,
        hypothesis_source="the candidate's `hypothesis` field in your JSON",
    )

    # The prompt's own copy is line-wrapped; the wording is what is compared.
    assert "it does not use the iteration's one deletion" in " ".join(discipline.split())
    assert "it does not use the iteration's one deletion" in REMOVAL_LICENCE
    assert "deletion slot" not in REMOVAL_LICENCE
    assert "loses no selection pairs and does not use" not in REMOVAL_LICENCE


def test_the_health_block_says_nothing_about_inert_axes_when_none_are(tmp_path):
    columns = {
        "Task Grounding": [float(i % 5 - 2) for i in range(50)],
        "Error Recovery": [float((i * 3) % 5 - 2) for i in range(50)],
    }
    stats = criterion_stats(
        [SimpleNamespace(rubric_scores={c: v[i] for c, v in columns.items()}) for i in range(50)]
    )

    text = render_criterion_health(stats, threshold=0.7)

    assert "inert axes" not in text
    assert DEAD_AXIS_ACTION not in text
    assert UNREAD_AXIS_ACTION not in text


# --- a block with no unread tag -----------------------------------------------

# A criterion-health block for `_stats_with_dead_axis()` at threshold 0.7 with
# no DECLARED BUT UNREAD tag. Pinned as a literal, so the test states the
# contract instead of restating the implementation.
PLAIN_BLOCK = "\n".join(
    [
        "criterion health over 50 rows, 3 criteria:",
        "  * Gold-Patch Relation: 100% of scores sit at +0 — DEAD AXIS: scorer gives one "
        "level on every row (anchors do not separate these trajectories); sharpen what "
        "divides adjacent levels or remove it",
        "      score levels: +0:100%",
        "  * Error Recovery: 20% of scores sit at -2",
        "      score levels: -2:20%, -1:20%, +0:20%, +1:20%, +2:20%",
        "  * Task Grounding: 20% of scores sit at -2",
        "      score levels: -2:20%, -1:20%, +0:20%, +1:20%, +2:20%",
        "  * effective rank 1.6 of 2 live axes",
        "  * |rho| between every pair of live axes (`!` marks >= 0.70, the pairs that "
        "largely re-measure one thing):",
        "           ER    TG",
        "    ER     -  0.50 ",
        "    TG  0.50     - ",
        "    legend: ER=Error Recovery, TG=Task Grounding",
        "  * most correlated pair: Error Recovery vs Task Grounding (|rho|=0.50)",
    ]
)


def test_the_telemetry_header_describes_the_block_under_it():
    """A block with no unread tag gets a header that does not mention one.

    The header is in `_format_loop_telemetry`, which is reached in every run —
    so a header that names DECLARED BUT UNREAD unconditionally would describe
    a tag the block below it never shows.
    """

    with_unread = _format_loop_telemetry(
        {
            "criterion_health": render_criterion_health(
                _stats_with_dead_axis(), threshold=0.7, declared_unread=["Error Recovery"]
            )
        }
    )
    without = _format_loop_telemetry({"criterion_health": PLAIN_BLOCK})

    assert "a DEAD axis and a criterion DECLARED BUT UNREAD both waste" in with_unread
    assert "measured over its own scores; a DEAD axis wastes every scorer call it costs" in without
    assert "DECLARED BUT UNREAD" not in without


def test_the_check_reads_files_and_never_imports_the_candidate(tmp_path):
    """A policy that raises on import is still measured.

    `check_candidate`'s contract, and the reason the whole detector is an
    AST walk: the precheck runs on candidates that do not load.
    """

    baseline = make_filter(tmp_path, "baseline", policy=_weights_policy(CRITERIA))
    candidate = make_filter(
        tmp_path,
        "cand",
        policy=_weights_policy(tuple(n for n in CRITERIA if n != "Query/PR Quality"))
        + "\n\nraise RuntimeError('boom')\n",
    )
    shutil.copy(baseline / "rules.py", candidate / "rules.py")

    report = _check(candidate, baseline, baseline)

    assert report.status == "invalid"
    assert report.declared_unread == ["Query/PR Quality"]


# --- end to end: the finding reaches the prompt --------------------------------

# Two criteria declared, one read: `TrainBoost` is scored on every row and the
# score never reaches the ordering.
UNREAD_INITIAL_POLICY = """RUBRIC_CRITERIA = ['Noisy', 'TrainBoost']


def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': scores['Noisy'], 'confidence': 0.5}
"""


def _prompts(tmp_path: Path, *, run_id: str) -> str:
    from tests.selection_gate_fixture import build_optimizer, prepare

    run = prepare(tmp_path)
    (run.initial / "policy.py").write_text(UNREAD_INITIAL_POLICY, encoding="utf-8")
    optimizer = build_optimizer(
        run,
        script={},
        iterations=2,
        run_id=run_id,
    )
    optimizer.run()
    return "\n".join(optimizer.proposer.prompts)


def test_a_run_tells_the_proposer_about_the_unread_criterion(tmp_path):
    text = _prompts(tmp_path, run_id="unread_on")

    assert "DECLARED BUT UNREAD" in text
    assert "inert axes" in text
    assert "read it in decide()" in text


def test_a_constant_mutated_after_assignment_reports_nothing(tmp_path):
    """Half an enumeration is the one state the literal set may not be in.

    A weights dict assigned five literals and then given its sixth key by a
    module-level subscript leaves the walk holding five names and blind to the
    sixth, so a key built from it would read as covering fewer criteria than it
    does. The scan says nothing rather than something wrong.
    """

    body = ",\n".join(f'    "{n}": 1.0' for n in CRITERIA[:-1])
    policy = (
        "RUBRIC_CRITERIA = [\n"
        + "".join(f'    "{n}",\n' for n in CRITERIA)
        + "]\n\n"
        + "WEIGHTS = {\n"
        + body
        + ",\n}\n"
        + f'WEIGHTS["{CRITERIA[-1]}"] = 2.0\n\n\n'
        + "def decide(trajectory, rubric_scores, triggered_rules):\n"
        + "    scores = rubric_scores or {}\n"
        + "    total = sum(scores.get(k, 0.0) * w for k, w in WEIGHTS.items())\n"
        + '    return {"score": total, "decision": "REVIEW", "evidence": []}\n'
    )
    d = make_filter(tmp_path, "mutated", policy=policy)

    assert cx.declared_unread_criteria(d) == []
