"""A row nobody could score is not a row anybody gets graded on.

The scorer refuses to invent a score for a response it cannot read, which
leaves the row without one — and the ranking side has to
decide what an absent score means. `unscored="loss"` is the right answer when a
candidate's own policy declined to rank a row, and the wrong one here: the row
is missing because the model's reply was unreadable, which is a property of the
call and not of the filter, so charging the candidate for it grades a filter on
the router's bad minute.

So the row leaves the exam for everyone: out of the score maps, and therefore
out of pair selection, out of concordance, and out of the artifacts that record
what the iteration was graded on. The count is written down, because an
iteration graded on a smaller pool than it was handed should say so rather than
leave it to be inferred from a pair count.

The same run checks the per-role cost columns, since both land in `record.json`
and `summary.csv` and one loop pass exercises both.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from meta_filter.core.solver import HeuristicRubricScorer, ScorerCost, Solver
from meta_filter.gate.selector import Selector
from meta_filter.gate.validator import Validator
from meta_filter.llm.base import LLMResponse
from meta_filter.llm.cost_meter import metered
from meta_filter.loop.optimizer import Optimizer, OptimizerConfig
from meta_filter.propose.proposer import Proposer, ProposerResult
from tests.demo_pool import selection_rows

ROOT = Path(__file__).resolve().parents[1]
INITIAL_RANKING = ROOT / "tests" / "fixtures" / "filters" / "six_criteria"
TRAIN = ROOT / "data" / "demo_train.jsonl"

# One real row of the demo pool. Which one does not matter; that it is one the
# other candidates DID score is the whole point.
FAILED_ROW = "T01_good_clean"

CALL_COST = 0.002


class _PairClient:
    """A judge with a consistent, order-independent preference."""

    model = "stub"

    def generate(self, prompt, **_kw):
        first = prompt.split("=== TRAJECTORY A (", 1)[1].split(")", 1)[0]
        letter = "A" if "good" in first else "B"
        return LLMResponse(
            text=f'{{"winner": "{letter}", "rationale": "it verified the fix"}}',
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=CALL_COST,
            cost_source="openrouter",
        )


class _Backend:
    """A pair-judging backend that rules on no individual trajectory."""

    client = metered(_PairClient(), "judge")

    def critic_verdict(self, trajectory):
        return None


class _InvertingProposer(Proposer):
    """One candidate that ranks the pool backwards, so pairs are disputed."""

    def __init__(self) -> None:
        self.client = _PairClient()

    def propose(self, *, candidates_root, iteration, **_kw):
        cand = Path(candidates_root) / f"iter{iteration:03d}_00_inverted"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(INITIAL_RANKING, cand, ignore=shutil.ignore_patterns("__pycache__"))
        policy = cand / "policy.py"
        source = policy.read_text(encoding="utf-8")
        policy.write_text(
            source.replace("score = 30.0 + mean * 15.0", "score = 80.0 - mean * 15.0"),
            encoding="utf-8",
        )
        return ProposerResult(
            prompt="",
            response=LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0),
            candidate_dirs=[cand],
        )


@dataclass
class _ScorerReportingFailures:
    """A working scorer that also reports rows as unscoreable.

    Deliberately still returns a score for the failed row. A scorer that
    simply omitted it would be excluded by the pre-existing `rank_score is
    None` path, and the test would pass without the fix under test; declaring
    the row while scoring it is what proves the exclusion reads
    `failed_row_ids`.
    """

    failed_row_ids: set[str] = field(default_factory=set)
    inner: HeuristicRubricScorer = field(default_factory=HeuristicRubricScorer)

    def score(self, trajectory, criteria, rubric_md: str = ""):
        return self.inner.score(trajectory, criteria, rubric_md)

    def reset_failure_stats(self) -> None:
        """The optimizer clears the count each iteration; these are this
        iteration's, so the set survives its own reset."""

    @property
    def parse_failures(self) -> int:
        return len(self.failed_row_ids)


def _run(tmp_path: Path, failed: set[str]) -> Path:
    scorer = _ScorerReportingFailures(failed_row_ids=set(failed))
    optimizer = Optimizer(
        initial_filter_dir=INITIAL_RANKING,
        train_data=TRAIN,
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=_InvertingProposer(),
        selector=Selector(),
        solver=Solver(rubric_scorer=scorer),
        validator=Validator(loss_weights={}, critic_backend=_Backend()),
        config=OptimizerConfig(
            iterations=1,
            candidates_per_iteration=1,
            pair_budget=8,
            max_pairs=66,
            pair_judge_cache_dir=tmp_path / "pair_cache",
            selection_data=selection_rows(tmp_path),
        ),
        run_id=f"failed-rows-{'with' if failed else 'without'}",
    )
    return optimizer.run()


def _ranking(run_dir: Path) -> dict:
    return json.loads((run_dir / "iterations" / "000" / "ranking.json").read_text(encoding="utf-8"))


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "iterations" / "000" / "record.json").read_text(encoding="utf-8"))


def _summary(run_dir: Path) -> list[dict]:
    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- the exclusion ----------------------------------------------------------


def test_a_failed_row_is_dropped_from_every_candidate_s_score_map(tmp_path):
    """Every candidate loses the row, not just the one that hit the failure —
    otherwise the candidates are sitting different exams."""

    control = _ranking(_run(tmp_path / "control", set()))
    treated = _ranking(_run(tmp_path / "treated", {FAILED_ROW}))

    assert control["candidates"], "the control run produced no ranking diagnostics"
    for cand_id, diag in treated["candidates"].items():
        before = control["candidates"][cand_id]["distribution"]["n"]
        assert diag["distribution"]["n"] == before - 1


def test_no_candidate_is_charged_for_the_row_nobody_could_score(tmp_path):
    """The row is gone from the pair universe, so it costs no candidate its
    pairs — which is what `unscored="loss"` would have done with it."""

    treated = _ranking(_run(tmp_path, {FAILED_ROW}))

    for diag in treated["candidates"].values():
        assert diag["unscored_rows"] == 0
        assert diag["unscored_pairs"] == 0


def test_the_count_and_the_ids_are_written_down(tmp_path):
    run_dir = _run(tmp_path, {FAILED_ROW})

    ranking = _ranking(run_dir)
    assert ranking["scorer_failed_rows"] == 1
    assert ranking["scorer_failed_row_ids"] == [FAILED_ROW]

    record = _record(run_dir)
    assert record["scorer_failed_rows"] == 1
    assert record["scorer_failed_row_ids"] == [FAILED_ROW]

    assert int(_summary(run_dir)[0]["scorer_failed_rows"]) == 1


def test_a_clean_iteration_reports_zero_rather_than_nothing(tmp_path):
    run_dir = _run(tmp_path, set())

    assert _ranking(run_dir)["scorer_failed_rows"] == 0
    assert _record(run_dir)["scorer_failed_rows"] == 0
    assert int(_summary(run_dir)[0]["scorer_failed_rows"]) == 0


# --- the cost columns -------------------------------------------------------


def test_summary_and_record_carry_a_cost_column_per_role(tmp_path):
    run_dir = _run(tmp_path, set())
    row = _summary(run_dir)[0]
    record = _record(run_dir)

    for role in ("solver", "judge", "proposer", "critic"):
        assert f"cost_{role}" in row
        assert f"cost_{role}" in record

    # The judge is the only metered client in this run, and it was called.
    assert float(row["cost_judge"]) > 0
    assert record["cost_judge"] == float(row["cost_judge"])
    # A heuristic scorer spends nothing, and that zero is a fact rather than
    # a gap.
    assert float(row["cost_solver"]) == 0.0


def test_the_judge_column_counts_the_calls_that_were_made(tmp_path):
    run_dir = _run(tmp_path, set())
    record = _record(run_dir)

    calls = record["cost_judge"] / CALL_COST
    assert calls >= 1
    assert abs(calls - round(calls)) < 1e-6, "the column is not a whole number of calls"


def test_costs_are_per_iteration_not_cumulative(tmp_path):
    """Two runs in one process share the ledger; each iteration reports its
    own spend, not the process's."""

    first = _record(_run(tmp_path / "a", set()))["cost_judge"]
    second = _record(_run(tmp_path / "b", set()))["cost_judge"]

    assert first > 0
    assert second == first


def test_scorer_cost_is_unused_by_the_heuristic_path(tmp_path):
    """Guard the seam the meter sits on: ScorerCost is still what the solver
    returns, and metering did not change it."""

    cost = ScorerCost()
    assert cost.llm_calls == 0


# --- the flag on the decision -----------------------------------------------


def test_the_solver_marks_a_row_it_could_not_score(tmp_path):
    """`scorer_failed` says the row has no score because nobody produced one,
    which a missing `rank_score` alone cannot: a filter may decline to rank a
    row by choice, and that is its own business."""

    from meta_filter.core.filter_loader import load_filter
    from meta_filter.core.llm_scorer import ScoreFailure, ScorerParseError
    from meta_filter.visibility.trajectory import load_trajectories

    class _RefusingScorer(HeuristicRubricScorer):
        """Unreadable on one row, ordinary on the rest — a whole pool of
        failures is a separate, fatal condition (`_report_unscoreable`)."""

        def score(self, trajectory, criteria, rubric_md=""):
            if trajectory.trajectory_id == FAILED_ROW:
                raise ScorerParseError(
                    ScoreFailure(
                        trajectory_id=trajectory.trajectory_id,
                        criterion="Task Grounding",
                        attempts=3,
                    )
                )
            return super().score(trajectory, criteria, rubric_md)

    rows = load_trajectories(TRAIN)
    decisions = Solver(rubric_scorer=_RefusingScorer(), lazy_scoring=False).run(
        load_filter(INITIAL_RANKING), rows
    )

    by_id = {d.trajectory_id: d for d in decisions}
    assert by_id[FAILED_ROW].scorer_failed is True
    assert by_id[FAILED_ROW].rank_score is None
    others = [d for d in decisions if d.trajectory_id != FAILED_ROW]
    assert others and not any(d.scorer_failed for d in others)


def test_a_scored_row_is_not_marked(tmp_path):
    from meta_filter.core.filter_loader import load_filter
    from meta_filter.visibility.trajectory import load_trajectories

    rows = load_trajectories(TRAIN)[:2]
    decisions = Solver(rubric_scorer=HeuristicRubricScorer(), lazy_scoring=False).run(
        load_filter(INITIAL_RANKING), rows
    )

    assert decisions
    assert not any(d.scorer_failed for d in decisions)


def test_a_flagged_decision_alone_removes_the_row_from_the_exam(tmp_path):
    """The scorer need not expose a counter: a decision carrying the flag is
    enough, which is what covers a refusal and a swapped-in scorer."""

    @dataclass
    class _FlaggingSolver(Solver):
        """Scores everything, then flags one row the way a failure would."""

        def run(self, loaded_filter, trajectories, **kwargs):
            decisions = super().run(loaded_filter, trajectories, **kwargs)
            for decision in decisions:
                if decision.trajectory_id == FAILED_ROW:
                    decision.scorer_failed = True
            return decisions

    optimizer = Optimizer(
        initial_filter_dir=INITIAL_RANKING,
        train_data=TRAIN,
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=_InvertingProposer(),
        selector=Selector(),
        solver=_FlaggingSolver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}, critic_backend=_Backend()),
        config=OptimizerConfig(
            iterations=1,
            candidates_per_iteration=1,
            pair_budget=8,
            max_pairs=66,
            pair_judge_cache_dir=tmp_path / "pair_cache",
            selection_data=selection_rows(tmp_path),
        ),
        run_id="flagged-decision",
    )
    ranking = _ranking(optimizer.run())

    assert ranking["scorer_failed_rows"] == 1
    assert ranking["scorer_failed_row_ids"] == [FAILED_ROW]
    control = _ranking(_run(tmp_path / "control", set()))
    for cand_id, diag in ranking["candidates"].items():
        assert diag["distribution"]["n"] == control["candidates"][cand_id]["distribution"]["n"] - 1
