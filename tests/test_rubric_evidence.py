"""The criterion-evidence blocks the proposer is shown, and the artifacts
behind them.

The proposer prompt carries the pairwise |rho| matrix between live axes and
the criterion-signal table; the run writes `criterion_lift.jsonl` and
`rationale_themes.json` whether or not a block renders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_filter.criterion_stats import criterion_stats, render_criterion_health
from tests.selection_gate_fixture import RUBRIC, RULES, build_optimizer, prepare
from tests.test_candidate_gating import INVERTED_NOISY_POLICY, RichProposer

# A two-criterion baseline: the |rho| matrix needs two live axes to print.
TWO_AXIS_POLICY = """RUBRIC_CRITERIA = ['Noisy', 'TrainBoost']


def decide(trajectory, scores, triggered):
    return {
        'decision': 'REVIEW',
        'score': scores['Noisy'] + scores['TrainBoost'],
        'confidence': 0.5,
    }
"""

MATRIX_MARKER = "|rho| between every pair of live axes"
SIGNAL_MARKER = "criterion signal (last winner, over up to"


# --- the health block on its own -------------------------------------------------


@pytest.fixture
def stats() -> dict:
    rows = []
    for i in range(20):
        rows.append(
            type("D", (), {"rubric_scores": {"Alpha": float(i % 5) - 2, "Beta": float(i % 3) - 1}})
        )
    return criterion_stats(rows)


def test_the_health_block_carries_the_matrix(stats):
    block = render_criterion_health(stats, threshold=0.7)
    assert MATRIX_MARKER in block and "legend:" in block
    assert "most correlated pair" in block


def test_the_dead_axis_reason_is_named():
    rows = [type("D", (), {"rubric_scores": {"Alpha": float(i), "Dead": 0.0}}) for i in range(20)]
    reasons = {"Dead": {"reason": "one_level", "fields": []}}
    block = render_criterion_health(criterion_stats(rows), threshold=0.7, dead_reasons=reasons)
    assert "DEAD AXIS: scorer gives one level on every row" in block


# --- through a run ---------------------------------------------------------------


def _run(tmp_path: Path, *, run_id: str) -> tuple[Path, list[str]]:
    run = prepare(tmp_path)
    (run.initial / "policy.py").write_text(TWO_AXIS_POLICY, encoding="utf-8")
    script = {
        0: [("inverted", INVERTED_NOISY_POLICY, RUBRIC, RULES)],
        1: [("inverted_again", INVERTED_NOISY_POLICY, RUBRIC, RULES)],
    }
    optimizer = build_optimizer(
        run,
        script=script,
        iterations=2,
        run_id=run_id,
    )
    optimizer.proposer = RichProposer(script, run.judge)
    run_dir = optimizer.run()
    # Every prompt the proposer was handed, not just the last one written to
    # `proposer/prompt.txt` — the revision leg overwrites that file and its
    # prompt carries only the precheck block.
    return run_dir, list(optimizer.proposer.prompts)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory) -> tuple[Path, list[str]]:
    return _run(tmp_path_factory.mktemp("full"), run_id="ev_full")


def test_the_prompts_carry_the_matrix_and_the_signal_table(full_run):
    prompts = "\n".join(full_run[1])
    assert MATRIX_MARKER in prompts
    assert SIGNAL_MARKER in prompts


def test_the_run_writes_the_artifacts(full_run):
    run_dir = full_run[0]
    lift = [
        json.loads(line)
        for line in (run_dir / "criterion_lift.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert any("single_criterion_concordance" in record for record in lift)
    assert (run_dir / "iterations" / "000" / "rationale_themes.json").is_file()


# --- the themes block, which the mock run never fills ----------------------------


def test_the_themes_block_is_written_and_shown(tmp_path, monkeypatch):
    """The mock judge's rationales match no theme, so the block is empty in
    the run above; this drives the renderer directly instead."""

    import meta_filter.rationale_themes as rt

    monkeypatch.setattr(rt, "rationale_themes", lambda *a, **k: {"themes": ["x"]})
    monkeypatch.setattr(rt, "render_rationale_themes", lambda report: "THEMES BLOCK")

    class _Judge:
        def presentations_for(self, *_a):
            return []

        def rationale_for(self, *_a):
            return "the patch was not run against the failing test"

    run = prepare(tmp_path)
    optimizer = build_optimizer(run, script={}, iterations=1, run_id="themes")
    block = optimizer._write_rationale_themes(
        0,
        winner_scores={},
        truth_pairs=[("a", "b")],
        judge=_Judge(),
        traj_by_id={},
        rubric_md=RUBRIC,
        criteria=["Noisy"],
    )
    written = optimizer.store.iteration_dir(0) / "rationale_themes.json"
    assert json.loads(written.read_text("utf-8")) == {"themes": ["x"]}
    assert block == "THEMES BLOCK"
