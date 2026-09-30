"""The proposer must be told the objective the selector actually applies.

A proposer told to minimize a loss while selection maximizes a score
optimizes the wrong thing, and a term shown with a 0.00 coefficient is as
misleading as a term not shown at all: it reliably spends part of the
reasoning budget on something that cannot move the score. The formula is
therefore rendered from the weights that apply it, live terms are named and
dead ones are absent.
"""

from __future__ import annotations

from pathlib import Path

from meta_filter.metrics import DEFAULT_SCORE_WEIGHTS
from meta_filter.proposer import PROPOSER_SYSTEM, build_proposer_system


def test_the_objective_is_stated_as_maximization():
    assert "MAXIMIZE" in PROPOSER_SYSTEM
    assert "minimize a scalar loss" not in PROPOSER_SYSTEM


def test_every_charged_term_is_named():
    """A term the proposer cannot see is a term it cannot optimize."""

    for term in ("pair_concordance", "complexity"):
        assert term in PROPOSER_SYSTEM, term


def test_no_zero_weight_term_is_shown():
    """And a term it cannot move is a term it should not be shown."""

    for term in ("rubric_quality", "rule_quality", "review_rate", "cost /"):
        assert term not in PROPOSER_SYSTEM, term
    assert "0.00 *" not in PROPOSER_SYSTEM


def test_the_objective_renders_exactly_the_live_terms():
    text = build_proposer_system(redundancy_penalty=0.5, redundancy_threshold=0.7)
    formula = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("  score =") or line.startswith("        ")
    ]
    assert formula == [
        "score = pair_concordance",
        "- 0.15 * min(1, complexity / 120)",
    ]


def test_a_disabled_redundancy_penalty_is_not_rendered():
    text = build_proposer_system(redundancy_penalty=0.0)
    assert "redundancy_penalty" not in text


def test_the_stated_weights_track_the_applied_weights():
    """Rendered, not written out — otherwise the two drift apart again."""

    custom = {
        **DEFAULT_SCORE_WEIGHTS,
        "lambda_review": 0.99,
        "lambda_cost": 0.20,
        "cost_budget": 500.0,
    }
    text = build_proposer_system(custom)
    assert "0.99 * review_rate" in text
    assert "cost / 500" in text
    # The default rendering must not leak into the overridden one.
    assert "0.60 * review_rate" not in text


# --- the degeneracy guard, stated as the code applies it ---------------------


def test_the_prompt_names_the_actual_floor_and_the_actual_penalty():
    """The floor scales with the pool (`ranking.default_min_distinct_levels`),
    and zeroing is only half the charge: `DEGENERACY_PENALTY` comes off the
    score as well, so a degenerate candidate cannot win on the secondary
    terms. A proposer told only "zeroed" can read the guard as a cap worth
    trading a little spread against."""

    from meta_filter.ranking import DEGENERACY_PENALTY

    prompt = build_proposer_system(min_distinct_levels=3)

    assert "fewer than 3 DISTINCT values" in prompt
    assert f"{DEGENERACY_PENALTY:.3f} is subtracted" in prompt
    assert "zeroed outright" not in prompt


def test_the_floor_rendered_is_the_floor_passed():
    assert "fewer than 20 DISTINCT" in build_proposer_system(min_distinct_levels=20)


def test_an_unstated_floor_drops_the_paragraph_rather_than_guessing_one():
    """Same rule as the zero-weight terms: a wrong number is worse than none."""

    prompt = build_proposer_system()

    assert "Degeneracy guard" not in prompt
    assert "zeroed outright" not in prompt


def test_the_optimizer_hands_over_the_floor_it_will_apply(tmp_path):
    """The prompt and the selector must not be able to name different numbers."""

    from meta_filter.optimizer import Optimizer, OptimizerConfig
    from meta_filter.proposer import Proposer
    from meta_filter.ranking import default_min_distinct_levels
    from meta_filter.selector import Selector
    from meta_filter.solver import HeuristicRubricScorer, Solver
    from meta_filter.validator import Validator

    root = Path(__file__).resolve().parents[1]
    optimizer = Optimizer(
        initial_filter_dir=root / "filters" / "initial_ranking_nometa",
        train_data=root / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=Proposer(),
        selector=Selector(),
        solver=Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}),
        config=OptimizerConfig(iterations=1, selection_data=root / "data" / "demo_selection.jsonl"),
        run_id="floor-test",
    )
    expected = default_min_distinct_levels(len(optimizer.train))

    assert f"fewer than {expected} DISTINCT" in optimizer.proposer.system_prompt


def test_an_explicit_floor_overrides_the_pool_derived_one(tmp_path):
    from meta_filter.optimizer import Optimizer, OptimizerConfig
    from meta_filter.proposer import Proposer
    from meta_filter.selector import Selector
    from meta_filter.solver import HeuristicRubricScorer, Solver
    from meta_filter.validator import Validator

    root = Path(__file__).resolve().parents[1]
    optimizer = Optimizer(
        initial_filter_dir=root / "filters" / "initial_ranking_nometa",
        train_data=root / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=Proposer(),
        selector=Selector(),
        solver=Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}),
        config=OptimizerConfig(
            iterations=1,
            min_distinct_levels=11,
            selection_data=root / "data" / "demo_selection.jsonl",
        ),
        run_id="floor-override",
    )

    assert "fewer than 11 DISTINCT" in optimizer.proposer.system_prompt
