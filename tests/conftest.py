"""Shared test setup: no test writes a cache into the repository."""

from __future__ import annotations

from functools import partial

import pytest

import meta_filter.optimizer as optimizer_module
from meta_filter.critics import RubricCritic, RuleCritic


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
