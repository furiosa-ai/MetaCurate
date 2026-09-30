"""The offline demo from the README runs end to end on the mock provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meta_filter.cli import app

ROOT = Path(__file__).resolve().parents[1]
CAP = "rubric_criteria=9,rule_count=13,rules_loc=522,rubric_chars=8785"


@pytest.mark.integration
def test_the_mock_demo_runs_and_writes_every_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--llm-provider",
            "mock",
            "--solver-backend",
            "llm",
            "--selection-data",
            "data/demo_selection.jsonl",
            "--complexity-ceiling",
            CAP,
            "--iterations",
            "2",
            "--solver-cache-dir",
            str(tmp_path / "scorer_cache"),
            "--judge-cache-dir",
            str(tmp_path / "judge_cache"),
            "--pair-judge-cache-dir",
            str(tmp_path / "pair_judge_cache"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--run-id",
            "demo",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    run_path = tmp_path / "runs" / "demo"

    assert (run_path / "config.yaml").exists()
    assert (run_path / "summary.csv").exists()
    assert (run_path / "train_pool.metadata_free.jsonl").exists()
    assert (tmp_path / "runs" / "_selection" / "demo.json").exists()

    for k in range(2):
        it_dir = run_path / "iterations" / f"{k:03d}"
        for name in ("validation.json", "selection.json", "report.md", "diff.md", "ranking.json"):
            assert (it_dir / name).exists(), name
        assert (it_dir / "current_filter").is_dir()

        selection = json.loads((it_dir / "selection.json").read_text(encoding="utf-8"))
        winner_dir = it_dir / "candidates" / selection["selected_candidate"]
        for name in ("rubric.md", "rules.py", "policy.py"):
            assert (winner_dir / name).exists()
        assert len(list((it_dir / "candidates").glob("iter*_baseline"))) == 1

        v2_dir = it_dir / "v2_critics"
        assert (v2_dir / "rubric_critic.json").exists()
        assert (v2_dir / "rule_critic.json").exists()

    header = (run_path / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
    for col in ("score", "critic_agreement", "gate_decision", "sel_concordance"):
        assert col in header
