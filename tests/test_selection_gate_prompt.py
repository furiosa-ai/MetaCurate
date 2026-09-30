"""The prompts never call the train score "what selection uses".

A run picks its winner by a paired McNemar test on a hidden pair set; the
train-pair concordance the proposer is shown is evidence, not the objective.
Every prompt that describes selection — the system prompt, the agentic
mission, the precheck step and the between-legs precheck block — has to say
so.
"""

from __future__ import annotations

from meta_filter.agentic_proposer import _agentic_instructions
from meta_filter.precheck import render_precheck_block
from meta_filter.proposer import build_proposer_system
from tests.proposer_prompt_fixture import render
from tests.selection_gate_fixture import build_optimizer, prepare
from tests.test_ranking_proposer_prompt_speaks_in_pairs import RANKING_STATE

TRAIN_SCORE_CLAIMS = (
    "same formula selection uses",
    "exact formula selection uses",
    "exactly what selection uses",
)

PARENT = {
    "rubric_criteria": 6,
    "rule_count": 5,
    "rules_loc": 40,
    "policy_loc": 20,
    "rubric_chars": 3000,
}


def _agentic() -> str:
    return _agentic_instructions(
        "iterations/001/candidates",
        1,
        3,
        "iterations/001/current_filter",
        parent_counts=PARENT,
    )


def test_the_static_prompt_says_the_pairs_are_evidence(tmp_path):
    prompt = render(tmp_path, ranking_state=RANKING_STATE)
    assert "cannot see" in prompt
    assert "These pairs are evidence, not the exam" in prompt
    for claim in TRAIN_SCORE_CLAIMS:
        assert claim not in prompt


def test_the_system_prompt_describes_the_paired_gate():
    system = build_proposer_system()
    assert "cannot see" in system
    assert "b = pairs it gets right that the current filter gets wrong, c = the reverse" in system
    assert "`iterations/*/selection.json` reports b, c, z and p" in system
    assert "necessary but not sufficient" in system
    assert "formula the precheck reports" in system
    for claim in TRAIN_SCORE_CLAIMS:
        assert claim not in system


def test_the_agentic_mission_and_precheck_step_describe_the_gate():
    text = _agentic()
    assert "What selection measures: pairwise CONCORDANCE on a FIXED set of trajectory" in text
    assert "cannot see" in text
    assert "necessary but not sufficient" in text
    assert "formula the precheck reports" in text
    for claim in TRAIN_SCORE_CLAIMS:
        assert claim not in text


def test_the_precheck_block_calls_its_number_the_precheck_formula():
    block = render_precheck_block({})
    assert "formula the precheck reports" in block
    assert "cannot see" in block
    assert "same formula selection uses" not in block


def test_the_optimizer_hands_the_proposer_the_gate_wording(tmp_path):
    run = prepare(tmp_path)
    optimizer = build_optimizer(run, script={}, iterations=1, run_id="gated")
    assert "cannot see" in optimizer.proposer.system_prompt
    assert "formula the precheck reports" in optimizer.proposer.system_prompt
    assert "exactly what selection uses" not in optimizer.proposer.system_prompt
