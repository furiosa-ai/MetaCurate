"""`--freeze rules,policy`: the rubric-only channel.

The flag reaches three places and each is pinned here: the CLI parses it and
refuses an unknown target; the optimizer records it in `config.yaml` and the
precheck invalidates a candidate that changed a frozen file; and the
edit-discipline block names the frozen files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from meta_filter.propose.agentic_proposer import _agentic_instructions
from meta_filter.propose.proposer import render_edit_discipline
from tests.proposer_prompt_fixture import render
from tests.selection_gate_fixture import (
    BASELINE_POLICY,
    RUBRIC,
    TRUTH_POLICY,
    build_optimizer,
    prepare,
)
from tests.test_candidate_gating import PADDED_RULES, RichProposer

ROOT = Path(__file__).resolve().parents[1]

COUNTS = {
    "rubric_criteria": 6,
    "rule_count": 9,
    "rules_loc": 348,
    "policy_loc": 100,
    "rubric_chars": 5000,
}


def _cli(*extra: str, tmp_path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "meta_filter.cli",
            "run",
            "--run-id",
            "x",
            "--llm-provider",
            "mock",
            "--iterations",
            "1",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--memory-dir",
            str(tmp_path / "mem"),
            "--selection-data",
            "data/demo_selection.jsonl",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_the_cli_refuses_an_unknown_target(tmp_path):
    proc = _cli("--freeze", "rules,tests", tmp_path=tmp_path)
    assert proc.returncode != 0
    flat = " ".join(proc.stdout.split() + proc.stderr.split())
    assert "unknown --freeze target 'tests'" in flat


def test_the_optimizer_records_the_freeze_and_the_precheck_refuses_a_frozen_edit(tmp_path):
    run = prepare(tmp_path)
    script = {
        # Iteration 0 disputes every pair so iteration 1 has a precheck.
        0: [("inverted", BASELINE_POLICY.replace("scores['Noisy']", "-scores['Noisy']"))],
        1: [
            # A better policy: under the freeze it may not change.
            ("truth", TRUTH_POLICY),
            # Rules padded by one line: frozen too.
            ("padded", BASELINE_POLICY, RUBRIC, PADDED_RULES),
            # Rubric-only edit of a kept criterion's prose: allowed.
            ("rubric_only", BASELINE_POLICY, RUBRIC.replace("at a glance", "on first read")),
        ],
    }
    optimizer = build_optimizer(run, script=script, iterations=2, run_id="frozen")
    optimizer.proposer = RichProposer(script, run.judge)
    optimizer.config.candidates_per_iteration = 3
    optimizer.config.freeze = ("rules", "policy")
    optimizer.proposer.frozen_files = ("rules", "policy")
    run_dir = optimizer.run()

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text("utf-8"))
    assert cfg["freeze"] == ["rules", "policy"]

    precheck = yaml.safe_load((run_dir / "iterations/001/precheck.json").read_text("utf-8"))
    gated = precheck["gated_out"]
    assert (
        gated["iter001_00_truth"]["reason"] == "constraint invalid: frozen file changed: policy.py"
    )
    assert "frozen file changed: rules.py" in gated["iter001_01_padded"]["reason"]
    assert "iter001_02_rubric_only" not in gated
    assert precheck["constraints"]["iter001_00_truth"]["frozen_changed"] == ["policy.py"]
    assert precheck["scored"] == ["iter001_02_rubric_only"]


def test_without_the_flag_nothing_is_frozen_and_config_says_so(tmp_path):
    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="free"
    )
    run_dir = optimizer.run()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text("utf-8"))
    assert cfg["freeze"] is None


def test_the_edit_discipline_block_names_the_frozen_files_only_when_frozen():
    on = render_edit_discipline(
        n_candidates=3,
        parent_counts=COUNTS,
        hypothesis_source="x",
        frozen=("rules", "policy"),
        score_tool=True,
    )
    assert "**Frozen files.** This run freezes `rules.py`, `policy.py`" in on
    assert "so only `rubric.md` may" in on
    assert "a smaller filter buys no exception" in on
    assert "Score every rewritten or new criterion with the criterion-scoring tool" in on

    no_tool = render_edit_discipline(
        n_candidates=3, parent_counts=COUNTS, hypothesis_source="x", frozen=("rules",)
    )
    assert "This run freezes `rules.py`:" in no_tool
    assert "only `rubric.md`, `policy.py` may" in no_tool
    assert "criterion-scoring tool" not in no_tool

    off = render_edit_discipline(n_candidates=3, parent_counts=COUNTS, hypothesis_source="x")
    assert "Frozen files" not in off


def test_the_discipline_block_carries_rubric_health_and_simplification():
    gated = render_edit_discipline(n_candidates=3, parent_counts=COUNTS, hypothesis_source="x")
    assert "**Rubric health.**" in gated
    assert "NEW dead axis (>= 95% of" in gated
    assert "**Growth needs evidence; making the filter smaller does not.**" in gated
    assert "loses no\nselection pair (b >= c)" in gated
    assert "it is not rejected as a no-op\nwhen b = c = 0" in gated
    # Clearing the gate is not promotion: a proposer that reads it as such
    # spends a slot on a b = c = 0 prune that can never win.
    assert "CLEARS THE GATE" in gated
    assert "Clearing the gate is not promotion" in gated


def test_the_static_prompt_carries_both_blocks_from_the_proposer_attributes(tmp_path):
    text = render(tmp_path, proposer_attrs={"frozen_files": ("rules", "policy")})
    assert "**Frozen files.** This run freezes `rules.py`, `policy.py`" in text
    assert "**Rubric health.**" in text


def test_the_agentic_mission_carries_the_frozen_sentence():
    text = _agentic_instructions(
        "iterations/001/candidates",
        1,
        3,
        "iterations/001/current_filter",
        parent_counts=COUNTS,
        frozen=("rules", "policy"),
    )
    assert "**Frozen files.** This run freezes `rules.py`, `policy.py`" in text
    assert "**Rubric health.**" in text
    off = _agentic_instructions("c", 1, 3, "cur", parent_counts=COUNTS)
    assert "Frozen files" not in off
