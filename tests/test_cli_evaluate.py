"""`meta_filter evaluate` runs one filter over a file and prints its metrics."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from meta_filter.cli import app

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_prints_the_filters_decision_counts():
    data = ROOT / "data" / "demo_train.jsonl"
    n_rows = sum(1 for line in data.read_text(encoding="utf-8").splitlines() if line.strip())

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--filter",
            str(ROOT / "filters" / "initial_blank_wired_v2"),
            "--data",
            str(data),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    metrics = json.loads(result.output)
    assert metrics["total"] == n_rows
    assert metrics["accepted"] + metrics["rejected"] + metrics["reviewed"] == n_rows
    # No critic was consulted, so nothing can have agreed or disagreed with one.
    assert metrics["critic_accept_total"] == metrics["critic_reject_total"] == 0
