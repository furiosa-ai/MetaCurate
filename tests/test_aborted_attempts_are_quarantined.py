"""No lineage may share directories with a dead one.

A run id reused across relaunches would accumulate several generations of
candidates in `iterations/000/candidates/` — one per launch. Candidate
collection is protected (it snapshots what pre-exists), but the agentic
proposer reads the run directory with tools, and telling the live lineage
from the dead ones would take reading validation.json.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path

from meta_filter.core.solver import HeuristicRubricScorer, Solver
from meta_filter.gate.selector import Selector
from meta_filter.gate.validator import Validator
from meta_filter.loop.optimizer import Optimizer, OptimizerConfig
from meta_filter.loop.resume import rebuild
from meta_filter.propose.proposer import Proposer
from tests.demo_pool import IdJudgeBackend, selection_rows

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "tests" / "fixtures" / "filters" / "five_criteria"


class _Client:
    model = "stub"

    def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        from meta_filter.llm.base import LLMResponse

        return LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0)


class _StopAfterCollecting(Proposer):
    """Records what the run dir looked like when propose() was entered."""

    def __init__(self):
        self.client = _Client()
        self.seen_candidate_dirs: list[str] = []

    def propose(self, **kwargs):
        root = kwargs["candidates_root"]
        self.seen_candidate_dirs = sorted(
            p.name for p in root.parent.parent.parent.glob("iterations/*/candidates/*")
        )
        raise KeyboardInterrupt


def _optimizer(runs_dir, proposer, *, resume=False) -> Optimizer:
    return Optimizer(
        initial_filter_dir=SEED,
        train_data=ROOT / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=runs_dir,
        memory_dir=Path(tempfile.mkdtemp()) / "memory",
        proposer=proposer,
        selector=Selector(),
        solver=Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}, critic_backend=IdJudgeBackend()),
        config=OptimizerConfig(
            iterations=2,
            candidates_per_iteration=1,
            resume=resume,
            selection_data=selection_rows(runs_dir.parent),
            pair_judge_cache_dir=runs_dir.parent / "pair_cache",
        ),
        run_id="reused-id",
    )


def _plant_stale_attempt(run_dir: Path) -> Path:
    """What a killed launch leaves behind: a partial iteration 0, no summary."""

    stale = run_dir / "iterations" / "000" / "candidates" / "iter000_00_stale_ghost"
    stale.mkdir(parents=True)
    for name in ("rubric.md", "rules.py", "policy.py"):
        shutil.copyfile(SEED / name, stale / name)
    return stale


def test_a_relaunch_under_a_reused_run_id_starts_from_a_clean_directory(tmp_path):
    run_dir = tmp_path / "runs" / "reused-id"
    _plant_stale_attempt(run_dir)

    proposer = _StopAfterCollecting()
    try:
        _optimizer(tmp_path / "runs", proposer).run()
    except KeyboardInterrupt:
        pass

    assert "iter000_00_stale_ghost" not in proposer.seen_candidate_dirs, (
        "the proposer investigates the run directory with tools; a dead "
        "lineage's candidates must not be there to read"
    )
    quarantined = list(run_dir.glob("_aborted/*/000/candidates/iter000_00_stale_ghost"))
    assert quarantined, "the stale attempt is kept as evidence, not deleted"


def test_two_aborted_attempts_do_not_collide(tmp_path):
    run_dir = tmp_path / "runs" / "reused-id"

    for _ in range(2):
        _plant_stale_attempt(run_dir)
        proposer = _StopAfterCollecting()
        try:
            _optimizer(tmp_path / "runs", proposer).run()
        except KeyboardInterrupt:
            pass

    ghosts = list(run_dir.glob("_aborted/*/000/candidates/iter000_00_stale_ghost"))
    assert len(ghosts) == 2, "each aborted attempt gets its own numbered slot"


def test_resume_starts_from_the_winner_not_the_iterations_starting_filter(tmp_path):
    """`iterations/k/current_filter` is snapshotted at the *top* of iteration k,
    so falling back to it resumes from the filter the completed iteration
    started with — silently discarding the winning edits its summary row
    proves it made."""

    run_dir = tmp_path / "runs" / "reused-id"
    it0 = run_dir / "iterations" / "000"

    # The filter iteration 0 started from — NOT what it produced.
    start_filter = it0 / "current_filter"
    start_filter.mkdir(parents=True)
    for name in ("rubric.md", "rules.py", "policy.py"):
        shutil.copyfile(SEED / name, start_filter / name)

    # The winner, with a recognisably different rubric.
    winner = it0 / "candidates" / "iter000_01_the_winner"
    winner.mkdir(parents=True)
    for name in ("rules.py", "policy.py"):
        shutil.copyfile(SEED / name, winner / name)
    (winner / "rubric.md").write_text(
        (SEED / "rubric.md").read_text(encoding="utf-8") + "\n## The Winning Criterion\n",
        encoding="utf-8",
    )
    (it0 / "selection.json").write_text(
        json.dumps({"selected_candidate": "iter000_01_the_winner"}), encoding="utf-8"
    )

    run_dir.joinpath("summary.csv").parent.mkdir(parents=True, exist_ok=True)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["iteration", "score"])
        writer.writeheader()
        writer.writerow({"iteration": 0, "score": 0.9})
    # iterations/001 was never snapshotted — the run died first.

    state = rebuild(run_dir)

    assert state is not None
    assert state.start_iteration == 1
    assert state.filter_dir == winner
    assert "The Winning Criterion" in (state.filter_dir / "rubric.md").read_text(encoding="utf-8")
