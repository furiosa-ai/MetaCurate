"""The proposer prompt may only describe what the loop computes.

The loop runs one objective — `pair_concordance`, adjudicated by a two-order
pair judge — and no part of it produces a per-row verdict. So the prompt must
not speak of false negatives, false positives or accept sets, must render a
misordered pair as one mistake naming both trajectories, and must omit any
section that could only ever read "(none)" — a proposer reads an empty
mistakes section as "the filter made no mistakes there".
"""

from __future__ import annotations

import pytest

from meta_filter.propose.agentic_proposer import _agentic_instructions
from tests.proposer_prompt_fixture import render

# Every one of these names something the loop does not compute, or names a
# pair mistake as if it were a per-row verdict.
PER_ROW_DIALECT = (
    "false negative",
    "false positive",
    "FN/FP",
    "fn_count",
    "review_rate",
    "must-accept",
    "must-reject",
    "hand-picked by the judge",
)

PARENT = {
    "rubric_criteria": 6,
    "rule_count": 5,
    "rules_loc": 40,
    "policy_loc": 20,
    "rubric_chars": 3000,
}

RANKING_STATE = {
    "pair_concordance": 0.742,
    "n_truth_pairs": 31,
    "n_tied_pairs": 2,
    "redundancy_penalty": 0.0,
    "complexity_score": 17.5,
    "min_distinct_levels": 3,
}


@pytest.fixture
def ranking_prompt(tmp_path) -> str:
    """A mid-run ranking prompt with every section populated."""

    return render(tmp_path, iteration=1, ranking_state=RANKING_STATE)


# --- the prompt speaks in pairs --------------------------------------------


@pytest.mark.parametrize("phrase", PER_ROW_DIALECT)
def test_no_per_row_dialect_survives_in_the_prompt(ranking_prompt, phrase):
    assert phrase.lower() not in ranking_prompt.lower(), phrase


def test_the_first_iteration_says_plainly_that_no_pair_has_been_judged(tmp_path):
    """ "(none)" under a mistakes heading reads as "no mistakes"."""

    prompt = " ".join(render(tmp_path, iteration=0, pairs=False).split())

    assert "no judged pairs yet" in prompt
    assert "the first truth set is built after this iteration's candidates" in prompt


def test_a_misordered_pair_is_one_block_naming_both_trajectories(ranking_prompt):
    """One inversion is one mistake, and both rows belong to it."""

    assert "### Misordered pairs" in ranking_prompt
    block = ranking_prompt.split("### Misordered pair 00", 1)[1].split("### Misordered pair 01", 1)[
        0
    ]

    assert "the judge preferred traj_under_a over traj_over_a" in block
    assert "this filter scored traj_over_a above traj_under_a (gap 19.00)" in block
    assert 'the judge\'s rationale: "B verified the fix and A did not"' in block
    # Both trajectories' evidence, side by side, under one heading.
    for tid in ("traj_under_a", "traj_over_a"):
        assert f"- task: fix the parser in {tid}" in block
    assert block.count("- rubric_scores: {'Task Grounding': 2.0}") == 2


def test_the_pair_block_never_relabels_an_ordering_as_a_verdict(ranking_prompt):
    assert "critic=REJECT" not in ranking_prompt
    assert "predicted=ACCEPT" not in ranking_prompt
    assert "pair_misorder_vs:" not in ranking_prompt


def test_state_json_carries_the_terms_the_selector_reads(ranking_prompt):
    import json

    state = json.loads(ranking_prompt.split("<STATE_JSON>", 1)[1].split("</STATE_JSON>", 1)[0])

    assert state["pair_concordance"] == 0.742
    assert state["n_truth_pairs"] == 31
    assert state["n_tied_pairs"] == 2
    assert state["redundancy_penalty"] == 0.0
    assert state["complexity_score"] == 17.5
    assert state["min_distinct_levels"] == 3
    for dead in ("fn_count", "fp_count", "review_rate", "loss", "must_accept_ids"):
        assert dead not in state, dead


def test_sections_the_loop_cannot_fill_are_absent(ranking_prompt):
    """A heading whose body is always "(none)" reads as "the filter made no
    mistakes there"; the pair judge produces no per-row verdicts to fill these."""

    for heading in (
        "### Curated trajectories",
        "### Where the judge said the last filter was fooled",
        "### Historical must-accept regression cases",
        "### Historical must-reject regression cases",
        "### Recent false negatives",
        "### Recent false positives",
    ):
        assert heading not in ranking_prompt, heading


def test_the_rubric_critic_survives_but_is_marked_uncharged(ranking_prompt):
    """A real second reading of the rubric; its weight in the objective is 0."""

    assert "### RubricCritic report (DIAGNOSTIC — weight 0 in the score)" in ranking_prompt
    assert "rubric_quality: 0.420" in ranking_prompt


def test_the_rule_critic_does_not_advise(ranking_prompt):
    """Its precision is measured against per-row verdicts the loop does not
    have, so it can only read low. A rule is measured by `criterion_lift`'s
    per-rule shift over the judged pairs instead, which the prompt carries; the
    critic still writes `v2_critics/rule_critic.json` for post-hoc reading."""

    assert "### RuleCritic report" not in ranking_prompt
    assert "Per-rule stats" not in ranking_prompt


def test_the_score_trajectory_names_concordance_and_flags_the_diagnostics(
    ranking_prompt,
):
    table = ranking_prompt.split("### Score trajectory", 1)[1]

    assert "concordance" in table.split("### Recent winning edits")[0]
    assert "* diagnostic only — weight 0 in the score" in table


def test_the_selector_reason_loses_the_counts_it_did_not_select_on(ranking_prompt):
    """A stored selector reason may carry FN/FP/review counts and a `loss`,
    none of which the loop selects on. The score comparison in front of them
    is kept verbatim."""

    line = next(
        li for li in ranking_prompt.splitlines() if li.strip().startswith("selector reason:")
    )

    assert "score 0.715 > baseline 0.620" in line
    assert "complexity=21.51" in line
    assert "loss" not in line
    assert "FN=" not in line


def test_the_output_schema_asks_for_pairs_to_reorder(ranking_prompt):
    schema = ranking_prompt.split("Output format", 1)[1].split("META_FILTER::PROPOSER", 1)[0]

    assert "misordered pairs" in schema
    assert "reorder" in schema
    assert "counterexample" not in schema


def test_the_schema_promises_only_evidence_the_prompt_actually_carries(
    ranking_prompt,
):
    """The four kinds of evidence reach the prompt inside the pair blocks, so
    the schema names them there rather than pointing at a separate pack."""

    schema = ranking_prompt.split("Output format", 1)[1].split("META_FILTER::PROPOSER", 1)[0]

    assert "evidence pack" not in schema
    for promised in ("rubric scores", "rules it", "action histogram", "reasoning"):
        assert promised in schema, promised


# --- the agentic mission ------------------------------------------------------


def test_the_mission_argues_about_ordering_not_an_accept_set():
    ranking = _agentic_instructions("c", 1, 3, "cur", parent_counts=PARENT)

    assert "Accept-set composition" not in ranking
    assert "a wrong ACCEPT ships bad training data" not in ranking
    assert "Top-of-ordering composition" in ranking
    assert "Confirmation on the high side is a design invariant" in ranking
    assert "a row you rank too LOW" in ranking


def test_the_mission_does_not_price_an_edit_in_scorer_calls():
    """The proposer does not pay for scoring, so the mission must not steer it
    towards cache hits; the self-contained-criterion advice stays."""

    # Line-wrapped prose, so compare on collapsed whitespace.
    ranking = " ".join(_agentic_instructions("c", 1, 3, "cur", parent_counts=PARENT).split())

    assert "The loop pays for scoring; you do not. Do not optimise for cache hits." in ranking
    assert "evaluated exactly like a rules edit, by the precheck and then by the full pass" in (
        ranking
    )
    assert "Write each rubric criterion SELF-CONTAINED" in ranking
    for gone in (
        "BYTE-IDENTICAL is not re-scored at all",
        "re-scores that criterion on every row in the pool",
        "pays no scorer calls at all",
        '"N cells paid for, M reused"',
    ):
        assert gone not in ranking


# --- and the numbers in it are the loop's own --------------------------------


def test_the_optimizer_hands_over_the_pair_numbers_it_measured(tmp_path):
    """STATE_JSON cannot read them off `current_metrics`.

    What the proposer is given there is `_stub_metrics` — zeros for score and
    agreement by design, so nothing stale is quoted as live. The pair terms
    therefore have to be passed separately, and the guard terms have to be
    right on iteration 0, before any pair exists.
    """

    from tests.test_critic_evidence_is_judge_derived import (
        _InvertingProposer,
        _ranking_run,
    )

    class _Capturing(_InvertingProposer):
        def __init__(self) -> None:
            super().__init__()
            self.states: list[dict] = []

        def propose(self, **kwargs):
            self.states.append(kwargs.get("ranking_state"))
            return super().propose(**kwargs)

    proposer = _Capturing()
    _ranking_run(tmp_path, proposer=proposer, iterations=2)

    first, second = proposer.states[0], proposer.states[1]

    # Iteration 0: the guard is known, nothing is measured yet.
    assert first["min_distinct_levels"] >= 1
    assert first["redundancy_threshold"] == 0.7
    assert "pair_concordance" not in first

    # Iteration 1: the previous winner's own numbers.
    assert 0.0 <= second["pair_concordance"] <= 1.0
    assert second["n_truth_pairs"] > 0
    assert second["n_tied_pairs"] >= 0
    assert second["complexity_score"] > 0
