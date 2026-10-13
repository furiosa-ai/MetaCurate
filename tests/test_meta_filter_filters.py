"""Filter loading, validation, complexity."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from meta_filter.core.filter_loader import FilterValidationError, load_filter, materialize_candidate
from meta_filter.gate.complexity import measure_filter_complexity

INITIAL = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "filters" / "five_criteria"


def test_initial_filter_loads():
    loaded = load_filter(INITIAL)
    assert loaded.policy_module.decide is not None
    assert any(fn.__name__ == "rule_no_turns" for fn in loaded.rule_functions)
    assert len(loaded.rubric_md) > 0


def test_invalid_candidate_rejected(tmp_path):
    materialize_candidate(
        tmp_path,
        rubric_md="# r",
        rules_py="def rule_broken(",  # syntax error
        policy_py="def decide(*_a, **_kw): return {}",
    )
    with pytest.raises(FilterValidationError):
        load_filter(tmp_path)


def test_missing_decide_rejected(tmp_path):
    materialize_candidate(
        tmp_path,
        rubric_md="# r",
        rules_py="",
        policy_py="X = 1\n",
    )
    with pytest.raises(FilterValidationError):
        load_filter(tmp_path)


def test_policy_may_import_sibling_rules_module(tmp_path):
    materialize_candidate(
        tmp_path,
        rubric_md="# r",
        rules_py="def _helper(t):\n    return 7\n\ndef rule_one(t):\n    return None\n",
        policy_py=(
            "from rules import _helper\n\n"
            "def decide(*a, **k):\n    return {'decision': 'REVIEW', 'h': _helper(None)}\n"
        ),
    )
    loaded = load_filter(tmp_path)
    assert loaded.policy_module.decide()["h"] == 7
    # The plain name must not leak out of the load.
    assert "rules" not in sys.modules


def test_complexity_grows_with_more_rules(tmp_path):
    base = tmp_path / "base"
    grown = tmp_path / "grown"
    materialize_candidate(
        base,
        rubric_md="# rubric\nCriterion: A\n",
        rules_py="def rule_one(t):\n    return None\n",
        policy_py="def decide(*a, **k):\n    return {'decision': 'REVIEW'}\n",
    )
    materialize_candidate(
        grown,
        rubric_md="# rubric\nCriterion: A\nCriterion: B\n",
        rules_py=textwrap.dedent(
            """
            def rule_one(t):
                return None

            def rule_two(t):
                return None
            """
        ).strip()
        + "\n",
        policy_py="def decide(*a, **k):\n    return {'decision': 'REVIEW'}\n",
    )
    _, s_base = measure_filter_complexity(base)
    _, s_grown = measure_filter_complexity(grown)
    assert s_grown > s_base
