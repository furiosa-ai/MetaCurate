"""A finished run must be able to say what it was configured to do.

Two runs that differ in one setting — say `--score-weight
lambda_complexity=0` — must not write byte-identical `config.yaml` files, or
the distinguishing configuration survives only in whatever launched them.
`loss_weights` is not the selection objective, so recording it alone would be
actively misleading: the score weights have to be written too.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from meta_filter.optimizer import Optimizer, OptimizerConfig
from meta_filter.proposer import Proposer, ProposerResult
from meta_filter.selector import Selector
from meta_filter.solver import HeuristicRubricScorer, Solver
from meta_filter.validator import Validator
from tests.demo_pool import IdJudgeBackend, selection_rows

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "tests" / "fixtures" / "filters" / "five_criteria"


class _StubClient:
    model = "stub"

    def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        from meta_filter.llm.base import LLMResponse

        return LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0)


class _NoOpProposer(Proposer):
    def __init__(self) -> None:
        self.client = _StubClient()

    def propose(self, **kwargs):
        from meta_filter.llm.base import LLMResponse

        return ProposerResult(
            prompt="",
            response=LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0),
            candidate_dirs=[],
        )


def _written_config(tmp_path: Path, config: OptimizerConfig) -> dict:
    config.selection_data = selection_rows(tmp_path)
    config.pair_judge_cache_dir = tmp_path / "pair_cache"
    optimizer = Optimizer(
        initial_filter_dir=INITIAL,
        train_data=ROOT / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=_NoOpProposer(),
        selector=Selector(),
        solver=Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}, critic_backend=IdJudgeBackend()),
        config=config,
        run_id="cfg-test",
    )
    optimizer.run()
    return yaml.safe_load((tmp_path / "runs" / "cfg-test" / "config.yaml").read_text())


def test_the_score_weight_override_reaches_config_yaml(tmp_path):
    """A run's one distinguishing setting has to survive into the run tree."""

    cfg = _written_config(
        tmp_path,
        OptimizerConfig(
            iterations=1,
            candidates_per_iteration=1,
            score_weights={"lambda_complexity": 0.0},
        ),
    )

    assert cfg["score_weights"]["lambda_complexity"] == 0.0, (
        "the distinguishing override must be recorded, not just the loss weight"
    )


def test_defaults_are_recorded_too_not_only_overrides(tmp_path):
    """A launch line that mentions nothing still has to pin the run down.

    Recording only what the CLI passed would leave a default-valued run
    looking identical to one whose default later changed.
    """

    cfg = _written_config(tmp_path, OptimizerConfig(iterations=1, candidates_per_iteration=1))

    from meta_filter.metrics import DEFAULT_SCORE_WEIGHTS

    assert cfg["score_weights"] == DEFAULT_SCORE_WEIGHTS
    assert cfg["max_pairs"] == 4000
    assert cfg["pair_budget"] is not None


def test_two_runs_differing_only_in_score_weight_write_different_configs(tmp_path):
    """Two such runs must be distinguishable on disk."""

    plain = _written_config(
        tmp_path / "a", OptimizerConfig(iterations=1, candidates_per_iteration=1)
    )
    flat = _written_config(
        tmp_path / "b",
        OptimizerConfig(
            iterations=1,
            candidates_per_iteration=1,
            score_weights={"lambda_complexity": 0.0},
        ),
    )

    assert plain != flat, "the two runs must not serialize identically"
    assert plain["score_weights"]["lambda_complexity"] != flat["score_weights"]["lambda_complexity"]


def test_the_ranking_knobs_are_recorded(tmp_path):

    cfg = _written_config(
        tmp_path,
        OptimizerConfig(
            iterations=1,
            candidates_per_iteration=1,
            max_pairs=19900,
            truth_includes_consensus=False,
            consensus_audit_share=0.25,
        ),
    )

    assert cfg["max_pairs"] == 19900
    assert cfg["truth_includes_consensus"] is False
    assert cfg["consensus_audit_share"] == 0.25


def test_loss_weights_still_recorded_and_still_separate(tmp_path):
    """The old field keeps its meaning; the new one does not replace it."""

    cfg = _written_config(
        tmp_path,
        OptimizerConfig(
            iterations=1,
            candidates_per_iteration=1,
            loss_weights={"lambda_complexity": 0.05},
            score_weights={"lambda_complexity": 0.0},
        ),
    )

    assert cfg["loss_weights"]["lambda_complexity"] == 0.05
    assert cfg["score_weights"]["lambda_complexity"] == 0.0
