"""The precheck inside a real ranking run: two legs, never three.

`test_precheck.py` checks the measurement. This checks the loop around it —
that leg 1's candidates get a `precheck.json`, that the proposer is handed
the readout and gets exactly one call to act on it, and that an iteration
with nothing adjudicated yet skips the whole thing instead of spending a
second call on an empty block.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from meta_filter.core.solver import HeuristicRubricScorer, Solver
from meta_filter.gate.selector import Selector
from meta_filter.gate.validator import Validator
from meta_filter.llm.base import LLMResponse
from meta_filter.loop.optimizer import Optimizer, OptimizerConfig
from meta_filter.propose.proposer import Proposer, ProposerResult
from tests.demo_pool import selection_rows

ROOT = Path(__file__).resolve().parents[1]
INITIAL_RANKING = ROOT / "tests" / "fixtures" / "filters" / "six_criteria"


class _PairClient:
    """A judge with a consistent, order-independent preference."""

    model = "stub"

    def generate(self, prompt, **_kw):
        first = prompt.split("=== TRAJECTORY A (", 1)[1].split(")", 1)[0]
        letter = "A" if "good" in first else "B"
        return LLMResponse(
            text=f'{{"winner": "{letter}", "rationale": "it verified the fix"}}',
        )


class _Backend:
    client = _PairClient()

    def critic_verdict(self, trajectory):
        return None


class _TwoLegProposer(Proposer):
    """Writes one inverted candidate, then nudges it back on the revision.

    Inverted on purpose: a candidate that disagrees with the baseline is what
    creates disputed pairs, and disputed pairs are what the judge is asked
    about — without one there is nothing adjudicated for iteration 1 to
    precheck against.
    """

    def __init__(self) -> None:
        self.client = _PairClient()
        self.blocks: list[str] = []
        self.revise_calls = 0

    def _write(self, candidates_root: Path, iteration: int, base: float) -> Path:
        cand = Path(candidates_root) / f"iter{iteration:03d}_00_inverted"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(INITIAL_RANKING, cand, ignore=shutil.ignore_patterns("__pycache__"))
        policy = cand / "policy.py"
        source = policy.read_text(encoding="utf-8")
        assert "score = 30.0 + mean * 15.0" in source
        policy.write_text(
            source.replace("score = 30.0 + mean * 15.0", f"score = {base} - mean * 15.0"),
            encoding="utf-8",
        )
        return cand

    def propose(self, *, candidates_root, iteration, **_kw):
        cand = self._write(candidates_root, iteration, 80.0)
        return ProposerResult(
            prompt="leg one",
            response=LLMResponse(text="{}"),
            candidate_dirs=[cand],
            legs=1,
        )

    def revise(self, *, result, precheck_block, candidates_root, iteration):
        self.revise_calls += 1
        self.blocks.append(precheck_block)
        # Edit in place, the way the agentic proposer does.
        cand = self._write(candidates_root, iteration, 78.0)
        return ProposerResult(
            prompt=result.prompt + "\n\nleg two",
            response=LLMResponse(text="{}"),
            candidate_dirs=[cand],
            legs=result.legs + 1,
        )


def _run(tmp_path: Path, *, iterations: int = 2):
    proposer = _TwoLegProposer()
    optimizer = Optimizer(
        initial_filter_dir=INITIAL_RANKING,
        train_data=ROOT / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=proposer,
        selector=Selector(),
        solver=Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}, critic_backend=_Backend()),
        config=OptimizerConfig(
            iterations=iterations,
            candidates_per_iteration=1,
            pair_budget=8,
            max_pairs=66,
            pair_judge_cache_dir=tmp_path / "pair_cache",
            selection_data=selection_rows(tmp_path),
        ),
        run_id="precheck-e2e",
    )
    return optimizer.run(), proposer


def _record(run_dir: Path, k: int) -> dict:
    return json.loads(
        (run_dir / "iterations" / f"{k:03d}" / "record.json").read_text(encoding="utf-8")
    )


def test_the_proposer_never_gets_a_third_leg(tmp_path):
    run_dir, proposer = _run(tmp_path)

    for k in range(2):
        assert _record(run_dir, k)["proposer_legs"] <= 2
    assert proposer.revise_calls <= 2, "one revision per iteration at most"


def test_every_candidate_carries_a_precheck_json(tmp_path):
    run_dir, _ = _run(tmp_path)

    it = run_dir / "iterations" / "001" / "candidates"
    written = sorted(p.parent.name for p in it.glob("*/precheck.json"))
    assert "iter001_00_inverted" in written
    payload = json.loads((it / "iter001_00_inverted" / "precheck.json").read_text())
    assert payload["n_pairs"] > 0
    assert payload["error"] == ""
    assert "score" in payload and "concordance" in payload


def test_the_record_keeps_the_readout_from_both_sides_of_the_revision(tmp_path):
    """Only the pair says whether the leg bought anything."""

    record = _record(_run(tmp_path)[0], 1)

    assert record["precheck"], "what the proposer was shown"
    assert record["precheck_after_revision"], "what it produced"
    for side in ("precheck", "precheck_after_revision"):
        for payload in record[side].values():
            assert payload["n_pairs"] > 0


def test_iteration_zero_skips_the_revision_because_nothing_is_judged_yet(tmp_path):
    """No adjudicated pair means no measurement, so the second call is waste."""

    run_dir, _ = _run(tmp_path)
    first = _record(run_dir, 0)

    assert first["proposer_legs"] == 1
    assert first["precheck"] == {}
    assert not list((run_dir / "iterations" / "000" / "candidates").glob("*/precheck.json"))


def test_the_proposer_is_shown_the_parent_to_compare_against(tmp_path):
    _, proposer = _run(tmp_path)

    assert proposer.blocks, "the revision leg ran at least once"
    block = proposer.blocks[0]
    assert "Parent filter" in block
    assert "no judge was called" in block
    assert "iter001_00_inverted" in block


def test_a_proposer_that_declines_the_leg_is_recorded_as_one_leg(tmp_path):
    """The static proposer has no workspace, so there is nothing to revise."""

    from meta_filter.propose.proposer import Proposer

    class _Declining(_TwoLegProposer):
        revise = Proposer.revise

    proposer = _Declining()
    optimizer = Optimizer(
        initial_filter_dir=INITIAL_RANKING,
        train_data=ROOT / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=proposer,
        selector=Selector(),
        solver=Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}, critic_backend=_Backend()),
        config=OptimizerConfig(
            iterations=2,
            candidates_per_iteration=1,
            pair_budget=8,
            max_pairs=66,
            pair_judge_cache_dir=tmp_path / "pair_cache",
            selection_data=selection_rows(tmp_path),
        ),
        run_id="precheck-declined",
    )
    run_dir = optimizer.run()
    record = _record(run_dir, 1)

    # The measurement still happens and still lands on disk — only the
    # revision does not.
    assert record["proposer_legs"] == 1
    assert record["precheck"]
    assert record["precheck_after_revision"] == {}
    assert (
        run_dir / "iterations" / "001" / "candidates" / "iter001_00_inverted" / "precheck.json"
    ).exists()
