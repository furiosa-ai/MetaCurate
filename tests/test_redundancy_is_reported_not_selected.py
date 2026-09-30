"""The redundancy penalty is measured everywhere and charged once, at the end.

`|rho|` between two rubric criteria is a property of the scored pool: it does
not exist until every row has been scored on every criterion. A proposer
writing candidates therefore has nothing to optimize against it — it cannot
know what its new criterion will correlate with until an iteration after it
ships one. Charged in selection, a criterion added to an existing rubric
would pay for its first contact with the incumbents before it had a round to
differentiate itself.

So the penalty is computed, logged, persisted and shown to the proposer, but
kept out of the score selection sorts on and out of the objective the
proposer is told to maximize. It is charged to `score_with_redundancy`, the
number the filter the run would deploy is reported with.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from meta_filter.agentic_proposer import _agentic_instructions
from meta_filter.criterion_stats import charge_redundancy
from meta_filter.llm.base import LLMResponse
from meta_filter.optimizer import Optimizer, OptimizerConfig
from meta_filter.precheck import PrecheckResult, render_precheck_block
from meta_filter.proposer import Proposer, ProposerResult, build_proposer_system
from meta_filter.selector import Selector
from meta_filter.solver import ScorerCost, Solver
from meta_filter.validator import Validator
from tests.demo_pool import selection_rows

ROOT = Path(__file__).resolve().parents[1]


def test_the_charge_lands_only_in_the_reported_number():
    assert charge_redundancy(0.8, 0.5) == (0.8, pytest.approx(0.3))
    assert charge_redundancy(0.8, 0.0) == (0.8, 0.8)


# --- a real run writes both numbers, and they differ by the penalty ----------


TWIN_RUBRIC = """# Rubric

## Criterion: Effort
+2 many content actions, -2 none.

## Criterion: Diligence
+2 many content actions, -2 none.
"""

TWIN_RULES = "def rule_noop(trajectory):\n    return None\n"

TWIN_POLICY = """RUBRIC_CRITERIA = ['Effort', 'Diligence']


def decide(trajectory, scores, triggered):
    mean = sum(scores.values()) / max(1, len(scores))
    return {'decision': 'REVIEW', 'score': 50.0 + mean * 10.0, 'confidence': 0.5}
"""


class _TwinScorer:
    """Two criteria that always agree, so |rho| is 1 and the penalty bites."""

    def score(self, trajectory, criteria, rubric_md=""):
        value = float(len(trajectory.turns) % 5) - 2.0
        return {c: value for c in criteria}, [f"{len(trajectory.turns)} turns"], ScorerCost()


class _PairClient:
    """A judge that prefers whichever trajectory id says it is good."""

    model = "stub"

    def generate(self, prompt, **_kw):
        first = prompt.split("=== TRAJECTORY A (", 1)[1].split(")", 1)[0]
        return LLMResponse(
            text=json.dumps(
                {"winner": "A" if "good" in first else "B", "rationale": "it verified the fix"}
            ),
            prompt_tokens=0,
            completion_tokens=0,
        )


class _Backend:
    client = _PairClient()

    def critic_verdict(self, trajectory):
        return None


class _TweakingProposer(Proposer):
    """One candidate: the same twin rubric with a different policy constant."""

    def __init__(self) -> None:
        self.client = _PairClient()

    def propose(self, *, candidates_root, iteration, **_kw):
        cand = Path(candidates_root) / f"iter{iteration:03d}_00_twin"
        from meta_filter.filter_loader import materialize_candidate

        materialize_candidate(
            cand,
            rubric_md=TWIN_RUBRIC,
            rules_py=TWIN_RULES,
            policy_py=TWIN_POLICY.replace("mean * 10.0", "mean * 12.0"),
        )
        return ProposerResult(
            prompt="",
            response=LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0),
            candidate_dirs=[cand],
        )


def _twin_run(tmp_path: Path) -> Path:
    initial = tmp_path / "initial"
    from meta_filter.filter_loader import materialize_candidate

    materialize_candidate(
        initial, rubric_md=TWIN_RUBRIC, rules_py=TWIN_RULES, policy_py=TWIN_POLICY
    )
    optimizer = Optimizer(
        initial_filter_dir=initial,
        train_data=ROOT / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=_TweakingProposer(),
        selector=Selector(),
        solver=Solver(rubric_scorer=_TwinScorer()),
        validator=Validator(loss_weights={}, critic_backend=_Backend()),
        config=OptimizerConfig(
            iterations=1,
            candidates_per_iteration=1,
            pair_budget=8,
            max_pairs=66,
            pair_judge_cache_dir=tmp_path / "pair_cache",
            selection_data=selection_rows(tmp_path),
        ),
        run_id="redundancy",
    )
    return optimizer.run()


@pytest.fixture(scope="module")
def twin_run(tmp_path_factory) -> Path:
    return _twin_run(tmp_path_factory.mktemp("twin"))


def _ranking(run_dir: Path) -> dict:
    return json.loads((run_dir / "iterations" / "000" / "ranking.json").read_text("utf-8"))


def _summary_row(run_dir: Path) -> dict[str, str]:
    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        return next(iter(csv.DictReader(handle)))


def test_the_penalty_is_measured(twin_run):
    """Otherwise every assertion below is vacuous."""

    for cid, diag in _ranking(twin_run)["candidates"].items():
        assert diag["criterion_stats"]["max_abs_corr"] == pytest.approx(1.0), cid
        assert diag["redundancy_penalty"] > 0, cid


def test_the_penalty_stays_out_of_the_selection_score(twin_run):
    for cid, diag in _ranking(twin_run)["candidates"].items():
        assert diag["score_with_redundancy"] == pytest.approx(
            diag["score"] - diag["redundancy_penalty"]
        ), cid


def test_summary_csv_carries_both_scores_beside_the_penalty(twin_run):
    row = _summary_row(twin_run)
    for column in ("score", "redundancy_penalty", "score_with_redundancy"):
        assert column in row, column
    winner = _ranking(twin_run)["candidates"][row["selected"]]
    assert float(row["redundancy_penalty"]) == pytest.approx(winner["redundancy_penalty"])
    assert float(row["score_with_redundancy"]) == pytest.approx(winner["score_with_redundancy"])
    assert float(row["score"]) == pytest.approx(winner["score"])


def test_the_iteration_record_carries_the_penalty(twin_run):
    record = json.loads((twin_run / "iterations" / "000" / "record.json").read_text("utf-8"))
    winner = _ranking(twin_run)["candidates"][record["selected_candidate"]]

    assert record["redundancy_penalty"] == pytest.approx(winner["redundancy_penalty"])
    assert record["score_with_redundancy"] == pytest.approx(winner["score_with_redundancy"])


# --- what the proposer is told ----------------------------------------------

REPORT_SENTENCE = (
    "Redundancy (|rho| > 0.7 between criteria) is measured and shown here but "
    "does not enter this iteration's selection; it is charged to the final "
    "filter's reported score, so avoid it where you can without giving up "
    "ordering power."
)


def _formula(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.startswith("  score =") or line.startswith("        ")
    ]


def test_the_formula_leaves_redundancy_out_and_says_why():
    text = build_proposer_system(redundancy_penalty=0.5, redundancy_threshold=0.7)

    assert _formula(text) == [
        "score = pair_concordance",
        "- 0.15 * min(1, complexity / 120)",
    ]
    # Dropped from the objective, not hidden: the term the proposer is shown
    # in four other places is given its status in one sentence.
    assert REPORT_SENTENCE in text


def test_the_tool_using_mission_states_the_consequence():
    text = _agentic_instructions("c", 1, 2, "cf", parent_counts={"rubric_criteria": 2})

    assert "does NOT enter this iteration's selection" in text
    assert "LOSES score for it" not in text
    assert "{redundancy_note}" not in text


def test_the_criterion_health_block_names_the_score_the_pair_costs():

    from meta_filter.criterion_stats import render_criterion_health

    stats = {
        "n_rows": 12,
        "n_criteria": 2,
        "n_live": 2,
        "per_criterion": {
            "Effort": {"modal_share": 0.4, "modal_value": 1.0, "dead": False, "distribution": {}},
            "Diligence": {
                "modal_share": 0.4,
                "modal_value": 1.0,
                "dead": False,
                "distribution": {},
            },
        },
        "effective_rank": 1.0,
        "max_abs_corr": 0.95,
        "max_pair": ["Effort", "Diligence"],
        "mean_abs_corr": 0.95,
    }

    report = render_criterion_health(stats, threshold=0.7)

    assert "|rho|=0.95" in report
    assert "reported score rather than to this iteration's selection" in report


def _precheck(cid: str, *, score: float, penalty: float) -> PrecheckResult:
    selection, reported = charge_redundancy(score, penalty)
    return PrecheckResult(
        candidate_id=cid,
        concordance=score,
        n_pairs=10,
        redundancy_penalty=penalty,
        score=selection,
        score_with_redundancy=reported,
    )


def test_the_precheck_block_prints_the_penalty_outside_the_score():
    block = render_precheck_block({"c0": _precheck("c0", score=0.8, penalty=0.5)})

    assert "score = concordance - complexity_term" in block
    assert "redundancy_penalty" not in block
    assert "does NOT enter this iteration's selection" in block
    # The number itself is still there, with the score it does reach.
    assert "redundancy -0.500" in block
    assert "reported score 0.3000" in block
