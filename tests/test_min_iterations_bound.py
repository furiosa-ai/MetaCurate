"""`--min-iterations` cannot exceed `--iterations` when patience is armed.

The floor is meaningless above the iteration count: the early stop could
never fire and nothing would say so. The CLI refuses the pair.
"""

from __future__ import annotations

from pathlib import Path

from tests.selection_gate_fixture import prepare


def _plain(output: str) -> str:
    return " ".join(output.replace("│", " ").replace("─", " ").split())


def _invoke(tmp_path: Path, run, *extra: str):
    from typer.testing import CliRunner

    from meta_filter.cli import app

    return CliRunner().invoke(
        app,
        [
            "run",
            "--train",
            str(run.train),
            "--initial-filter",
            str(run.initial),
            "--selection-data",
            str(run.selection),
            "--llm-provider",
            "mock",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--memory-dir",
            str(tmp_path / "memory"),
            # Keep every cache inside the test's own directory.
            "--judge-cache-dir",
            str(tmp_path / "caches" / "judge"),
            "--pair-judge-cache-dir",
            str(tmp_path / "caches" / "pair_judge"),
            "--solver-cache-dir",
            str(tmp_path / "caches" / "scorer"),
            *extra,
        ],
    )


def test_the_cli_refuses_a_floor_above_the_iteration_count(tmp_path):
    run = prepare(tmp_path)
    result = _invoke(tmp_path, run, "--iterations", "3", "--patience", "3", "--min-iterations", "4")
    assert result.exit_code != 0
    text = _plain(result.output)
    assert "--min-iterations 4 exceeds --iterations 3" in text
    assert not (tmp_path / "runs").exists()


def test_the_floor_may_equal_the_iteration_count(tmp_path):
    """The check is `>`; a run whose floor is its whole length is legal."""

    run = prepare(tmp_path)
    result = _invoke(tmp_path, run, "--iterations", "1", "--patience", "1", "--min-iterations", "1")
    assert result.exit_code == 0, result.output


def test_without_patience_the_default_floor_does_not_block_a_short_run(tmp_path):
    """`min_iterations` defaults to 4 and `iterations` to 3: unconditional,
    the check would reject every plain run."""

    run = prepare(tmp_path)
    result = _invoke(tmp_path, run, "--iterations", "1")
    assert result.exit_code == 0, result.output
