"""The loop measures what a candidate's rules fire on, instead of asking.

`meta_filter.rule_tool` gives the proposer a free way to check a draft rule
before it commits to one, and across two probe iterations the proposer called
it zero times. The reason is structural: its deliverable is candidate
directories, and the prompt promises the loop will run each candidate and
report back. That promise holds for the rubric and not for the rules — the
precheck reports concordance, redundancy and score, and in all three a rule
that fires on nothing and a rule that fires and does not help are the same
number. So the measurement moves into the precheck, where the rows already
are and the rules cost nothing to run.

Everything below builds real filters and real trajectories and executes the
rules: a mocked rule execution would assert that the plumbing is connected
and say nothing about whether a rule fires, which is the whole question.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from meta_filter import precheck as pc
from meta_filter.filter_loader import load_filter
from meta_filter.schemas import Trajectory, Turn
from meta_filter.solver import ScorerCost

ROOT = Path(__file__).resolve().parents[1]
INITIAL_RANKING = ROOT / "tests" / "fixtures" / "filters" / "six_criteria"

# Rule sources appended to a copy of the initial filter. Each one is a whole
# rule under the `rules.py` contract — `dict` when it fires, `None` when it
# declines — so what the precheck measures is the rule itself.
DEAD = '''

def rule_probe_dead(trajectory):
    """Reads a field these rows do not carry, so it never fires."""

    return None
'''

SATURATED = '''

def rule_probe_saturated(trajectory):
    """Fires on everything, so it orders nothing."""

    return {"name": "rule_probe_saturated", "verdict": "REJECT", "evidence": ["always"]}
'''

HALF = '''

def rule_probe_half(trajectory):
    """Fires on the even-numbered rows only."""

    if int(trajectory.trajectory_id.removeprefix("t")) % 2:
        return None
    return {"name": "rule_probe_half", "verdict": "REJECT", "evidence": ["even row"]}
'''

RAISES = '''

def rule_probe_raises(trajectory):
    """Raises on every row, the way a draft reading a missing field does."""

    raise ValueError("no such field")
'''


class _FlatScorer:
    """Scores every criterion the same everywhere. No LLM, no cache, no cost.

    The concordance these tests produce is meaningless and unread: nothing
    here is about the score, only about what the rules did beside it.
    """

    def score(self, trajectory, criteria, rubric_md=""):
        return {c: 1.0 for c in criteria}, [], ScorerCost(llm_calls=0, llm_tokens=0)


def _traj(tid: str) -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task=f"task for {tid}",
        turns=[
            Turn(role="assistant", action="grep", content=f"grep -rn thing {tid}"),
            Turn(role="assistant", action="apply_patch", content="apply_patch <<EOF"),
            Turn(role="assistant", action="pytest", content="pytest -q"),
        ],
        metadata={"instance_id": tid, "exit_status": "Submitted", "submission": "diff --git a b"},
    )


@pytest.fixture()
def rows():
    return {f"t{i}": _traj(f"t{i}") for i in range(4)}


@pytest.fixture()
def pairs():
    return [("t0", "t1"), ("t2", "t3")]


def _make_filter(dest: Path, *extra_rules: str) -> Path:
    """A copy of the shipped ranking filter, with rule sources appended."""

    shutil.copytree(INITIAL_RANKING, dest, ignore=shutil.ignore_patterns("__pycache__"))
    if extra_rules:
        with (dest / "rules.py").open("a", encoding="utf-8") as fh:
            fh.write("".join(extra_rules))
    return dest


def _candidate(tmp_path: Path, name: str, *extra_rules: str) -> Path:
    root = tmp_path / "candidates"
    root.mkdir(exist_ok=True)
    return _make_filter(root / name, *extra_rules)


def _write_diagnosis(tmp_path: Path, **rule_check_by_slug: str) -> None:
    payload = {
        "diagnosis": "the rules never fire",
        "candidates": [
            {"name": slug, "hypothesis": "h", "notes": "n", "rule_check": claim}
            for slug, claim in rule_check_by_slug.items()
        ],
    }
    (tmp_path / "candidates" / "diagnosis.json").write_text(json.dumps(payload), encoding="utf-8")


def _run(tmp_path, parent_dir, candidates, rows, pairs):
    return pc.run_precheck(
        candidates,
        parent_filter=load_filter(parent_dir),
        trajectories_by_id=rows,
        truth_pairs=pairs,
        scorer=_FlatScorer(),
        lazy_scoring=False,
        initial_dir=parent_dir,
    )


def _firing(result, name: str) -> pc.RuleFiring:
    match = [r for r in result.rule_firing.rules if r.name == name]
    assert match, f"{name} was not measured at all"
    return match[0]


def _reasons(result) -> list[str]:
    return list(result.constraint.reasons)


# --- the measurement itself -------------------------------------------------


def test_every_rule_is_measured_with_its_own_denominator(tmp_path, rows, pairs):
    """`3` alone is unreadable; `3 of 4 precheck rows` is the claim."""

    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_half", HALF)
    _write_diagnosis(tmp_path, half="rule_probe_half 7/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.rule_firing.n_rows == len(rows)
    half = _firing(result, "rule_probe_half")
    assert (half.fires, half.n_rows) == (2, 4)
    assert half.inherited is False
    # The parent's own rules are measured too, under the same denominator.
    assert _firing(result, "rule_nemotron_no_verification").inherited is True


def test_the_measurement_reaches_precheck_json(tmp_path, rows, pairs):
    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_half", HALF)
    _write_diagnosis(tmp_path, half="rule_probe_half 2/20")

    _run(tmp_path, parent, [cand], rows, pairs)
    payload = json.loads((cand / "precheck.json").read_text(encoding="utf-8"))

    block = payload["rule_firing"]
    assert block["n_rows"] == 4
    measured = {r["name"]: r["fires"] for r in block["rules"]}
    assert measured["rule_probe_half"] == 2
    assert block["rule_check_status"] == "verified"


# --- the two degenerate shapes ----------------------------------------------


def test_a_new_dead_rule_raises_the_bar(tmp_path, rows, pairs):
    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_dead", DEAD)
    _write_diagnosis(tmp_path, dead="rule_probe_dead 0/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert "rule_probe_dead fired on 0 of 4 precheck rows — it cannot order anything" in _reasons(
        result
    )
    assert result.constraint.status == "needs_strong_evidence"


def test_a_dead_rule_is_never_invalid(tmp_path, rows, pairs):
    """0 firings on 4 rows is not proof of 0 on the pool.

    `invalid` would drop the candidate outright on evidence that cannot
    support it; `needs_strong_evidence` says exactly what was seen.
    """

    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_dead", DEAD)
    _write_diagnosis(tmp_path, dead="rule_probe_dead 0/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.constraint.status != "invalid"


def test_a_new_saturated_rule_raises_the_bar(tmp_path, rows, pairs):
    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_sat", SATURATED)
    _write_diagnosis(tmp_path, sat="rule_probe_saturated 20/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert "rule_probe_saturated fired on 4 of 4 precheck rows — it separates no pair" in _reasons(
        result
    )
    assert result.constraint.status == "needs_strong_evidence"


def test_an_inherited_dead_rule_is_reported_but_not_charged(tmp_path, rows, pairs):
    """The candidate did not write it, so it does not raise the candidate's bar.

    It is still measured and still printed, because a proposer only ever
    shown its own new defects carries an inherited one to the end of the run.
    """

    parent = _make_filter(tmp_path / "parent", DEAD)
    cand = _candidate(tmp_path, "iter000_01_kept", DEAD, HALF)
    _write_diagnosis(tmp_path, kept="rule_probe_half 2/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    dead = _firing(result, "rule_probe_dead")
    assert (dead.fires, dead.inherited) == (0, True)
    assert not [r for r in _reasons(result) if "rule_probe_dead" in r]


def test_a_raising_rule_is_recorded_and_the_others_still_measured(tmp_path, rows, pairs):
    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_boom", RAISES, HALF)
    _write_diagnosis(tmp_path, boom="rule_probe_half 2/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    boom = _firing(result, "rule_probe_raises")
    assert boom.raised == len(rows)
    assert boom.error_types == ("ValueError",)
    assert _firing(result, "rule_probe_half").fires == 2
    assert result.rule_firing.error == ""


# --- the proposer's own claim ------------------------------------------------


def test_a_matching_rule_check_passes_and_is_named(tmp_path, rows, pairs):
    """A claim that holds is reported, so the signal is visible when it works."""

    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_half", HALF)
    # The denominator differs on purpose: the proposer samples the train pool
    # through `rule_tool`, the precheck runs over the pair rows.
    _write_diagnosis(tmp_path, half="rule_probe_half 2/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.rule_firing.rule_check_status == "verified"
    assert result.rule_firing.rule_check_verified == ["rule_probe_half"]
    assert not [r for r in _reasons(result) if "rule_check" in r]


def test_a_differing_count_on_a_rule_that_does_fire_is_not_a_mismatch(tmp_path, rows, pairs):
    """Two honest samples of the same rule disagree on k and agree on the rule.

    The proposer counts over rows it drew itself; the precheck counts over
    its own. Requiring the two numbers to be equal would fail the candidate
    that actually ran the checker and pass the one that guessed a number
    matching the precheck by luck.
    """

    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_half", HALF)
    _write_diagnosis(tmp_path, half="rule_probe_half 4/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.rule_firing.rule_check_status == "verified"
    assert result.rule_firing.rule_check_verified == ["rule_probe_half"]
    assert not [r for r in _reasons(result) if "rule_check" in r]


def test_claiming_a_rule_fires_when_it_fires_on_nothing_raises_the_bar(tmp_path, rows, pairs):
    """No sampling explains a rule reported as live that fires on no row."""

    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_dead", DEAD)
    _write_diagnosis(tmp_path, dead="rule_probe_dead 3/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.rule_firing.rule_check_status == "mismatch"
    assert (
        "rule_check claims rule_probe_dead fired 3 time(s); measured 0 of 4 precheck rows"
        in _reasons(result)
    )
    assert result.constraint.status == "needs_strong_evidence"


def test_a_rule_check_carrying_its_own_evidence_still_parses(tmp_path, rows, pairs):
    """The counts are read out of the sentence, not matched against it.

    A proposer that reports what it measured and then says where it saw it
    has done more work than one that reports the bare count, and an anchored
    match threw exactly that candidate away.
    """

    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_half", HALF)
    _write_diagnosis(tmp_path, half="rule_probe_half 2/20; fired on traj_540bcc8f at 375 bytes.")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.rule_firing.rule_check_status == "verified"
    assert result.rule_firing.rule_check_verified == ["rule_probe_half"]


def test_declaring_no_rules_is_a_claim_and_is_checked(tmp_path, rows, pairs):
    """A candidate that keeps no rules has no count to give, and says so."""

    parent = _make_filter(tmp_path / "parent", DEAD)
    cand = _candidate(tmp_path, "iter000_01_none")
    (cand / "rules.py").write_text("", encoding="utf-8")
    _write_diagnosis(tmp_path, none="No rules remain; the inherited one was dead.")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.rule_firing.rule_check_status == "verified"
    assert not [r for r in _reasons(result) if "rule_check" in r]


def test_declaring_no_rules_while_defining_one_raises_the_bar(tmp_path, rows, pairs):
    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_half", HALF)
    _write_diagnosis(tmp_path, half="none")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.rule_firing.rule_check_status == "mismatch"
    assert [
        r
        for r in _reasons(result)
        if "says this candidate has no rules" in r and "rule_probe_half" in r
    ]
    assert result.constraint.status == "needs_strong_evidence"


def test_a_rule_check_naming_a_rule_that_does_not_exist_raises_the_bar(tmp_path, rows, pairs):
    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_half", HALF)
    _write_diagnosis(tmp_path, half="rule_probe_ghost 3/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert [r for r in _reasons(result) if "rule_probe_ghost" in r and "no rule of that name" in r]
    assert result.constraint.status == "needs_strong_evidence"


def test_a_missing_rule_check_raises_the_bar_when_rules_changed(tmp_path, rows, pairs):
    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_half", HALF)
    _write_diagnosis(tmp_path, other="rule_probe_half 2/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.rule_firing.rule_check_status == "absent"
    assert pc._NO_RULE_CHECK_REASON in _reasons(result)


def test_an_unchanged_rules_py_owes_no_rule_check(tmp_path, rows, pairs):
    """Nothing is asked of a candidate that did not touch the rules."""

    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_rubric_only")
    (cand / "rubric.md").write_text(
        (cand / "rubric.md").read_text(encoding="utf-8") + "\nA trailing note.\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.rule_firing.rules_changed is False
    assert result.rule_firing.rule_check_status == "not_required"
    assert pc._NO_RULE_CHECK_REASON not in _reasons(result)
    assert result.rule_firing.rules, "an unchanged rules.py is still measured"


def test_a_malformed_rule_check_is_treated_as_absent(tmp_path, rows, pairs):
    """Free text written by a model; every malformation is data, not an error."""

    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_half", HALF)
    _write_diagnosis(tmp_path, half="I ran the checker and it looked fine")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    assert result.rule_firing.rule_check_status == "unparseable"
    assert [r for r in _reasons(result) if "could not be read" in r]
    assert result.constraint.status == "needs_strong_evidence"


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "rule_x", "rule_x 3", "rule_x three/20", "4/20", "I did not check."],
)
def test_parsing_a_rule_check_never_raises(raw):
    assert pc.parse_rule_check(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("rule_x 3/20, nonsense", {"rule_x": 3}),
        ("rule_x 3/20; fired on traj_abc at 375 bytes.", {"rule_x": 3}),
        ("rule_a 3/20 and rule_b 0/20 (dead, so I dropped it)", {"rule_a": 3, "rule_b": 0}),
        ("none", {}),
        ("No rules remain; the inherited one was dead.", {}),
    ],
)
def test_parsing_reads_counts_out_of_the_sentence(raw, expected):
    assert pc.parse_rule_check(raw) == expected


def test_parsing_a_well_formed_rule_check_ignores_the_denominator(tmp_path):
    assert pc.parse_rule_check("rule_a 1/20, rule_b 2/41") == {"rule_a": 1, "rule_b": 2}


def test_an_unreadable_diagnosis_file_yields_no_claims(tmp_path):
    (tmp_path / "diagnosis.json").write_text("{not json", encoding="utf-8")
    assert pc.read_rule_checks(tmp_path) == {}
    assert pc.read_rule_checks(tmp_path / "nowhere") == {}


# --- what the proposer is shown ---------------------------------------------


def test_the_precheck_block_prints_the_counts_with_their_denominator(tmp_path, rows, pairs):
    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_half", HALF)
    _write_diagnosis(tmp_path, half="rule_probe_half 2/20")

    results = _run(tmp_path, parent, [cand], rows, pairs)
    block = pc.render_precheck_block(results)

    assert "rules, over the 4 precheck rows" in block
    assert "rule_probe_half: fired on 2 of 4" in block
    assert "rule_check in diagnosis.json matches the measurement" in block


def test_a_rule_that_raises_on_every_row_says_so_rather_than_saying_it_never_fired(
    tmp_path, rows, pairs
):
    """Zero firings and total failure want opposite repairs.

    A rule that raises fires zero times, so the "cannot order anything"
    reason is literally true and points at the predicate — when what is
    wrong is the code. The reason has to name the exception instead.
    """

    parent = _make_filter(tmp_path / "parent")
    cand = _candidate(tmp_path, "iter000_01_boom", RAISES)
    _write_diagnosis(tmp_path, boom="rule_probe_raises 0/20")

    result = _run(tmp_path, parent, [cand], rows, pairs)[cand.name]

    firing = _firing(result, "rule_probe_raises")
    assert firing.fires == 0 and firing.raised == firing.n_rows
    reasons = _reasons(result)
    assert any("raised on all" in r and "rule_probe_raises" in r for r in reasons), reasons
    assert not any("cannot order anything" in r for r in reasons), reasons


# A parser that wants the count immediately after the name cannot read many of
# the `rule_check` strings proposers actually write. These are the shapes they
# take, kept verbatim so the next narrowing of the parser fails here instead of
# in a run.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "rule_broad_sprawl fired 1/20 and ordered 3/3 split judged pairs correctly",
            {"rule_broad_sprawl": 1},
        ),
        (
            "rule_repeated_editing fired on 3 of 20 sampled train rows and ordered 3 of 4",
            {"rule_repeated_editing": 3},
        ),
        (
            "rule_repeated_editing unchanged; current telemetry measured 13/50",
            {"rule_repeated_editing": 13},
        ),
        (
            "rule_no_test_invocation fired on 14/49 train rows across fixed batches",
            {"rule_no_test_invocation": 14},
        ),
        ("No deterministic rules. Synthetic policy inputs produced 11 distinct score levels.", {}),
        ("No rule_* functions; deterministic penalties intentionally removed.", {}),
        ("rule_tool confirmed there are no deterministic rules to check.", {}),
        (
            "No deterministic rule is defined; rule_tool correctly refused because there is "
            "nothing to replay.",
            {},
        ),
    ],
)
def test_parsing_reads_the_shapes_the_full_runs_actually_wrote(raw, expected):
    assert pc.parse_rule_check(raw) == expected


def test_a_count_belonging_to_a_later_clause_is_not_read_as_the_firing_count():
    """`fired 1/20 and ordered 3/3` is one firing count and one hit rate."""

    claims = pc.parse_rule_check("rule_x fired 1/20 and ordered 3/3 split judged pairs")
    assert claims == {"rule_x": 1}


def test_an_unnamed_count_is_attributed_only_when_one_rule_could_own_it():
    anchored = "Replayed on 50/50 train rows; fired 2, raised 0; judged-pair credit 4/5."
    assert pc.parse_rule_check(anchored, ["rule_only"]) == {"rule_only": 2}
    assert pc.parse_rule_check(anchored, ["rule_a", "rule_b"]) is None
    assert pc.parse_rule_check(anchored) is None


def test_an_unnamed_string_with_several_counts_and_no_anchor_is_unreadable():
    """Three readings of one rule; picking one invented a mismatch.

    `rule_runaway_sprawl` reported 0/20 on a random sample and 3/50 on the
    full pool. Both are true, the precheck measures a third set, and a parser
    that grabbed the first number faulted the candidate for the sample it was
    honest about.
    """

    raw = "random sample 0/20; targeted widening 1/6 with 1/1 pair credit; full-pool check 3/50"
    assert pc.parse_rule_check(raw, ["rule_runaway_sprawl"]) is None
