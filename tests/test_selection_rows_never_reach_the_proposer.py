"""The hidden selection rows stay hidden (the exposure guard).

The gate's whole value is that the proposer never saw the exam. The optimizer
proves it two ways: the CLI and the constructor refuse a selection file that
overlaps train or probe, and after every proposer leg the saved prompt is
grepped for every selection id and the run is aborted — not warned — if one
is there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from meta_filter.core.llm_scorer import sanitize_trajectory_id
from meta_filter.gate.selection_set import (
    SelectionSet,
    selection_row_ids_in_text,
    selection_set_path,
)
from meta_filter.llm.base import LLMResponse
from tests.selection_gate_fixture import (
    TRAIN_ONLY_POLICY,
    TRUTH_POLICY,
    ScriptedProposer,
    build_optimizer,
    prepare,
)


def _saved_prompts(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("iterations/*/proposer/prompt.txt"))


def _plain(output: str) -> str:
    """Typer renders errors in a wrapped rich panel; flatten it for substring checks."""

    return " ".join(output.replace("│", " ").replace("─", " ").split())


def test_no_saved_prompt_or_proposer_argument_names_a_selection_row(tmp_path):
    run = prepare(tmp_path)
    script = {0: [("truth", TRUTH_POLICY)], 1: [("trainonly", TRAIN_ONLY_POLICY)]}
    optimizer = build_optimizer(run, script=script, iterations=2, run_id="hidden")

    run_dir = optimizer.run()

    prompts = _saved_prompts(run_dir)
    assert len(prompts) == 2
    for path in prompts:
        text = path.read_text(encoding="utf-8")
        assert len(text) > 200, "the prompt is the full argument dump, not a stub"
        assert selection_row_ids_in_text(text, run.selection_ids) == []
    # Negative control for the grep itself: the train-side pair dump of
    # iteration 0 is full of train ids and the same call finds them there.
    pair_dump = (run_dir / "iterations" / "000" / "ranking_pairs.jsonl").read_text("utf-8")
    assert selection_row_ids_in_text(pair_dump, run.train_ids)
    assert selection_row_ids_in_text(pair_dump, run.selection_ids) == []
    # The proposer's own record of what it was handed, including the
    # revision leg where the precheck block is rendered.
    proposer = optimizer.proposer
    assert isinstance(proposer, ScriptedProposer)
    assert proposer.revise_calls >= 1
    for prompt in proposer.prompts:
        assert selection_row_ids_in_text(prompt, run.selection_ids) == []

    # None of the train-side artifacts carry a selection row either.
    for name in ("ranking_pairs.jsonl", "critic_outputs", "v2_critics"):
        for path in run_dir.glob(f"iterations/*/{name}/**/*"):
            if path.is_file():
                assert selection_row_ids_in_text(path.read_text("utf-8"), run.selection_ids) == []
        for path in run_dir.glob(f"iterations/*/{name}"):
            if path.is_file():
                assert selection_row_ids_in_text(path.read_text("utf-8"), run.selection_ids) == []


def test_the_selection_set_is_written_once_with_the_planned_pair_count(tmp_path):
    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="built", selection_pairs=40
    )

    run_dir = optimizer.run()

    path = selection_set_path(run_dir)
    assert path.is_file()
    saved = SelectionSet.load(path)
    assert sorted(saved.row_ids) == sorted(run.selection_ids)
    # Truths are distinct so every planned pair resolves; 40 of C(16, 2)=120.
    assert saved.provenance["n_planned"] == 40
    assert len(saved.pairs) == 40
    assert saved.tied == [] and saved.unresolved == []
    assert saved.provenance["stratified"] is True
    assert saved.provenance["judge_model"] == "truth-judge"
    # Every judged pair lies on selection rows and on nothing else.
    for hi, lo in saved.pairs:
        assert hi in run.selection_ids and lo in run.selection_ids


def test_a_prompt_that_names_a_selection_row_aborts_the_run(tmp_path):
    run = prepare(tmp_path)
    leaked = run.selection_ids[3]

    class _LeakingProposer(ScriptedProposer):
        def propose(self, **kwargs):
            result = super().propose(**kwargs)
            result.prompt += f"\nby the way, consider {leaked}"
            return result

    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="leak"
    )
    optimizer.proposer = _LeakingProposer({0: [("truth", TRUTH_POLICY)]}, run.judge)

    with pytest.raises(RuntimeError, match="hidden selection set is exposed") as caught:
        optimizer.run()
    assert leaked in str(caught.value)
    # Nothing was selected: the run died on the guard, before scoring.
    assert not (optimizer.store.run_dir / "iterations" / "000" / "selection.json").exists()


def test_a_selection_file_overlapping_train_is_refused_at_construction(tmp_path):
    run = prepare(tmp_path)
    overlapping = tmp_path / "overlap.jsonl"
    lines = run.selection.read_text("utf-8").splitlines()
    first_train = run.train.read_text("utf-8").splitlines()[0]
    overlapping.write_text("\n".join([*lines, first_train]) + "\n", encoding="utf-8")
    run.selection = overlapping

    with pytest.raises(ValueError, match="disjoint from train and probe"):
        build_optimizer(run, script={}, iterations=1, run_id="overlap")


def test_the_cli_refuses_an_overlapping_selection_file_before_building_anything(tmp_path):
    from typer.testing import CliRunner

    from meta_filter.cli import app

    run = prepare(tmp_path)
    overlapping = tmp_path / "overlap.jsonl"
    overlapping.write_text(
        run.selection.read_text("utf-8") + run.train.read_text("utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--train",
            str(run.train),
            "--initial-filter",
            str(run.initial),
            "--llm-provider",
            "mock",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--selection-data",
            str(overlapping),
        ],
    )
    assert result.exit_code != 0
    assert "shared with train" in _plain(result.output)
    assert not (tmp_path / "runs").exists()


def test_the_cli_refuses_a_missing_selection_file(tmp_path):
    from typer.testing import CliRunner

    from meta_filter.cli import app

    run = prepare(tmp_path)
    missing = CliRunner().invoke(
        app,
        [
            "run",
            "--train",
            str(run.train),
            "--initial-filter",
            str(run.initial),
            "--llm-provider",
            "mock",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--selection-data",
            str(tmp_path / "nope.jsonl"),
        ],
    )
    assert missing.exit_code != 0 and "does not exist" in _plain(missing.output)


def test_the_selection_file_itself_carries_raw_ids_for_the_grep_to_find(tmp_path):
    """If the file were pseudonymised, `selection_row_ids_in_text` would be blind."""

    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="raw"
    )
    run_dir = optimizer.run()
    payload = json.loads((selection_set_path(run_dir)).read_text("utf-8"))
    assert set(payload["row_ids"]) == set(run.selection_ids)


# --- the Codex read audit: a read of an answer key is a leak ------------------


@dataclass
class _AuditedResponse(LLMResponse):
    """`CodexResponse`'s audit field, without the rest of the harness."""

    reads_outside_workspace: tuple[str, ...] = ()


class _ReadingProposer(ScriptedProposer):
    """The fixture proposer, reporting the reads a Codex agent would have."""

    reads: tuple[str, ...] = ()

    def propose(self, **kwargs):
        result = super().propose(**kwargs)
        result.response = _AuditedResponse(
            text="{}", prompt_tokens=0, completion_tokens=0, reads_outside_workspace=self.reads
        )
        return result


def _reading_optimizer(tmp_path: Path, reads: tuple[str, ...], run_id: str):
    run = prepare(tmp_path)
    script = {0: [("truth", TRUTH_POLICY)]}
    optimizer = build_optimizer(run, script=script, iterations=1, run_id=run_id)
    proposer = _ReadingProposer(script, run.judge)
    proposer.reads = reads
    optimizer.proposer = proposer
    return optimizer


def _cell_name(trajectory_id: str) -> str:
    """A per-criterion cache filename for `trajectory_id`, as the scorer writes it."""

    return f"{sanitize_trajectory_id(trajectory_id)}__crit_ab12cd34_0123456789__view_x.json"


@pytest.mark.parametrize(
    "where", ["selection", "pair_judge_cache", "scorer_cache_selection_cell", "scorer_cache_dir"]
)
def test_a_recorded_read_under_an_answer_key_aborts_the_run(tmp_path, where):
    """Codex cannot block reads, so the audit list is checked after the fact.

    Under the scorer cache the rule is per cell: a hidden (selection) row's
    cell is the exam, and so is the directory itself, whose listing names
    every row ever scored. Train-row cells are exempt (next test).
    """

    optimizer = _reading_optimizer(tmp_path, (), "leak-" + where)
    run_dir = optimizer.store.run_dir
    run_ids = json.loads(Path(optimizer.config.selection_data).read_text().splitlines()[0])
    # The fixture scorer has no cache; give it the attribute the audit reads.
    scorer_cache = tmp_path / "scorer_cache"
    optimizer.solver.rubric_scorer.cache_dir = scorer_cache
    target = {
        "selection": selection_set_path(run_dir),
        "pair_judge_cache": Path(optimizer.config.pair_judge_cache_dir) / "some" / "verdict.json",
        "scorer_cache_selection_cell": (
            scorer_cache / "per_criterion" / _cell_name(run_ids["trajectory_id"])
        ),
        "scorer_cache_dir": scorer_cache / "per_criterion",
    }[where]
    # Recorded the way the audit records it: absolute, and possibly via a
    # relative hop the harness never normalised.
    optimizer.proposer.reads = (
        str(target),
        str(target.parent / ".." / target.parent.name / target.name),
    )

    with pytest.raises(RuntimeError, match="hidden selection set is exposed") as caught:
        optimizer.run()
    assert str(target) in str(caught.value)
    assert not (run_dir / "iterations" / "000" / "selection.json").exists()


def test_a_train_rows_scorer_cell_read_invalidates_the_proposal(tmp_path, caplog):
    """A train cell is not the exam, so the run is not aborted as exposed; but
    any read outside the workspace invalidates that iteration's proposal."""

    run = prepare(tmp_path)
    script = {0: [("truth", TRUTH_POLICY)]}
    optimizer = build_optimizer(run, script=script, iterations=1, run_id="train-cell")
    scorer_cache = tmp_path / "scorer_cache"
    optimizer.solver.rubric_scorer.cache_dir = scorer_cache
    proposer = _ReadingProposer(script, run.judge)
    proposer.reads = (str(scorer_cache / "per_criterion" / _cell_name(run.train_ids[0])),)
    optimizer.proposer = proposer

    with pytest.raises(RuntimeError, match="proposer failed"):
        optimizer.run()
    assert "read 1 path(s) outside its workspace" in caplog.text
    assert "hidden selection set is exposed" not in caplog.text


def test_a_read_elsewhere_outside_the_workspace_invalidates_the_proposal(tmp_path, caplog):
    optimizer = _reading_optimizer(
        tmp_path, (str(tmp_path / "somewhere_else" / "notes.md"),), "harmless"
    )
    with pytest.raises(RuntimeError, match="proposer failed"):
        optimizer.run()
    assert "read 1 path(s) outside its workspace" in caplog.text
    assert "notes.md" in caplog.text


def test_a_response_without_an_audit_field_passes(tmp_path):
    """The static proposer and the Claude Code harness report no reads."""

    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="plain"
    )
    assert not hasattr(optimizer.proposer.client, "reads_outside_workspace")
    assert (optimizer.run() / "iterations" / "000" / "selection.json").exists()
