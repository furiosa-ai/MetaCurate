"""The subset scan can score through the loop's own solver, not just the CLI.

A scan whose model string differs from the loop's (the CLI alias `sonnet`)
files its per-criterion cells under a namespace no run has filled, so it
re-buys the pool at full price and agrees with the loop only by accident.
`--scorer-engine openai_compatible` builds the client the way `cli.py` builds
the loop's solver. What these tests pin is the part that is silent when wrong:
the cache key the scan writes under, and the fact that the default path is
the CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meta_filter.core.llm_scorer import LLMRubricScorer, _criterion_cache_key
from meta_filter.core.schemas import Trajectory
from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient
from meta_filter.llm.cost_meter import LEDGER, MeteredClient
from scripts.build_ranking_subset import build_parser, build_scorer_client, main

# The loop's solver model in the four run cells, and the suffix its cached
# cells carry.
LUNA = "openai/gpt-5.6-luna"
LUNA_SUFFIX = "__model_openai-gpt-5.6-luna"

SECTION = "## Criterion: Verification Completeness\n+2 ran the test"
CRITERION = "Verification Completeness"
HEADER = "# Rubric\nScore each criterion."


@pytest.fixture
def openai_env(monkeypatch):
    """A key that is never spent: no test here calls `generate`."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-real-one")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def test_openai_engine_scores_under_the_model_the_loop_bought(openai_env):
    """The whole point of the flag: `MeteredClient.model` must proxy through to
    `LLMRubricScorer._model_name()`, or the scan writes a namespace of its own.

    A wrapper that swallowed `.model` would return "", the key would lose its
    suffix entirely, and every one of the pool's cells would be a miss — with
    nothing in the log to say so, because a cache miss is not an error.
    """

    client = build_scorer_client("openai_compatible", model=LUNA, auth="api_key")
    scorer = LLMRubricScorer(client=client, cache_dir=Path("/tmp/unused-cache-dir"))

    assert client.model == LUNA
    assert scorer._model_name() == LUNA

    key = _criterion_cache_key(
        SECTION, CRITERION, Trajectory(trajectory_id="row:1"), scorer._model_name(), "", HEADER
    )
    loop_key = _criterion_cache_key(
        SECTION, CRITERION, Trajectory(trajectory_id="row:1"), LUNA, "", HEADER
    )

    assert key == loop_key
    assert key.endswith(LUNA_SUFFIX)


def test_openai_engine_bills_the_scan_to_the_solver_role(openai_env):
    """`LEDGER.columns()` is what the scan logs, so the meter has to be the
    global one and the role has to be the column the loop reports spend in."""

    client = build_scorer_client("openai_compatible", model=LUNA, auth="api_key")

    assert isinstance(client, MeteredClient)
    assert client.role == "solver"
    assert client.ledger is LEDGER
    assert "cost_solver" in LEDGER.columns()


def test_claude_code_is_the_default_and_needs_no_openai_key(monkeypatch):
    """An invocation that passes no --scorer-engine at all must keep shelling
    out to the CLI."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    parser = build_parser()

    assert parser.get_default("scorer_engine") == "claude_code"

    client = build_scorer_client(
        parser.get_default("scorer_engine"),
        model=parser.get_default("scorer_model"),
        auth=parser.get_default("scorer_auth"),
    )

    assert isinstance(client, ClaudeCodeHeadlessClient)
    assert not isinstance(client, MeteredClient)
    assert (client.model, client.auth) == ("sonnet", "api_key")
    scorer = LLMRubricScorer(client=client, cache_dir=Path("/tmp/unused-cache-dir"))
    assert scorer._model_name() == "sonnet"


def test_the_flag_offers_both_engines_and_nothing_else():
    help_text = build_parser().format_help()

    assert "--scorer-engine" in help_text
    action = next(a for a in build_parser()._actions if a.dest == "scorer_engine")
    assert tuple(action.choices) == ("claude_code", "openai_compatible")


def test_the_claude_alias_is_refused_before_the_scan_spends_anything(tmp_path, monkeypatch):
    """`sonnet` is a CLI alias, not an endpoint slug. Left at the default it
    would key every cell under `__model_sonnet`, which no run has ever filled:
    a full-price re-scan of the pool that still cannot be compared to the loop.
    Stop at argument-parsing time, before a pool is even read."""

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_ranking_subset",
            "--filter-dir",
            str(tmp_path / "filter"),
            "--name",
            "probe_subset",
            "--pool",
            str(tmp_path / "pool.jsonl"),
            "--superset-dir",
            str(tmp_path / "superset"),
            "--readme",
            str(tmp_path / "README.md"),
            "--note",
            "n",
            "--out",
            str(tmp_path / "out"),
            "--scorer-engine",
            "openai_compatible",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert "openai/gpt-5.6-luna" in str(excinfo.value)
    # Nothing was scored, packaged or even opened.
    assert not (tmp_path / "out").exists()
