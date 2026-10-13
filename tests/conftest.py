"""Shared test setup: no test writes a cache into the repository."""

from __future__ import annotations

import os

# Typer forces a colour terminal when these are set (GitHub Actions sets the
# first), and the ANSI codes then split the CLI error text the tests match.
# Read at import time, so they must go before anything imports typer.
for _var in ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS"):
    os.environ.pop(_var, None)

from functools import partial  # noqa: E402

import pytest  # noqa: E402

import meta_filter.loop.optimizer as optimizer_module  # noqa: E402
from meta_filter.critics import RubricCritic, RuleCritic  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _critic_caches_in_tmp(tmp_path_factory):
    """The optimizer builds its rubric and rule critics with their default
    `data/` cache directories; point them at a session temporary one.

    Session-scoped so module-scoped run fixtures are covered too.
    """

    root = tmp_path_factory.mktemp("critic_caches")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            optimizer_module,
            "RubricCritic",
            partial(RubricCritic, cache_dir=root / "rubric_critic_cache"),
        )
        patch.setattr(
            optimizer_module,
            "RuleCritic",
            partial(RuleCritic, cache_dir=root / "rule_critic_cache"),
        )
        yield
