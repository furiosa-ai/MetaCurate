"""A run that cannot propose has to stop, not iterate.

Run `claude` as root and the CLI refuses whenever
`--dangerously-skip-permissions` is set, so every proposer call dies the same
way. Catching each one, logging it and moving on would fail twenty iterations
in seconds and still print `run complete` and exit 0.

Skipping a failed proposal is right for a timeout or a 5xx. It is wrong for a
misconfiguration, and the two are told apart by whether the next one also
fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meta_filter.core.solver import HeuristicRubricScorer, Solver
from meta_filter.gate.selector import Selector
from meta_filter.gate.validator import Validator
from meta_filter.loop.optimizer import MAX_CONSECUTIVE_PROPOSER_FAILURES, Optimizer, OptimizerConfig
from meta_filter.propose.proposer import Proposer
from tests.demo_pool import IdJudgeBackend, selection_rows

ROOT = Path(__file__).resolve().parents[1]


class _SilentClient:
    """The optimizer builds its critics from `proposer.client`."""

    model = "stub"

    def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        from meta_filter.llm.base import LLMResponse

        return LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0)


class _AlwaysFailingProposer(Proposer):
    """Stands in for a CLI that refuses to run at all."""

    def __init__(self) -> None:
        self.attempts = 0
        self.client = _SilentClient()

    def propose(self, **kwargs):
        self.attempts += 1
        raise RuntimeError(
            "claude headless exit 1: --dangerously-skip-permissions cannot be "
            "used with root/sudo privileges for security reasons"
        )


def _optimizer(tmp_path, proposer, *, iterations: int) -> Optimizer:
    return Optimizer(
        initial_filter_dir=ROOT / "tests" / "fixtures" / "filters" / "five_criteria",
        train_data=ROOT / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=proposer,
        selector=Selector(),
        solver=Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}, critic_backend=IdJudgeBackend()),
        config=OptimizerConfig(
            iterations=iterations,
            candidates_per_iteration=1,
            selection_data=selection_rows(tmp_path),
            pair_judge_cache_dir=tmp_path / "pair_cache",
        ),
        run_id="giving-up",
    )


def test_a_run_that_never_proposes_fails_instead_of_completing(tmp_path):
    """Twenty wasted iterations and a zero exit status is the failure this
    guards against — the machine stays busy and nothing says anything is wrong."""

    proposer = _AlwaysFailingProposer()

    with pytest.raises(RuntimeError, match="configuration failure"):
        _optimizer(tmp_path, proposer, iterations=20).run()

    assert proposer.attempts == MAX_CONSECUTIVE_PROPOSER_FAILURES, (
        "it must give up after the third strike, not run out the iteration budget"
    )


def test_an_isolated_failure_is_still_skipped(tmp_path):
    """The original behaviour has to survive: one timeout must not discard the
    work of the iterations around it."""

    class _FailsOnceThenWorks(_AlwaysFailingProposer):
        def propose(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("provider 503")
            raise KeyboardInterrupt  # stop the run here; we only care it got past 1

    proposer = _FailsOnceThenWorks()
    with pytest.raises(KeyboardInterrupt):
        _optimizer(tmp_path, proposer, iterations=20).run()

    assert proposer.attempts == 2, "the second iteration must still be attempted"


def test_a_spend_cap_stops_on_the_first_failure(tmp_path):
    """A cap is conclusive immediately; counting to three only wastes calls.

    Treated as a streak, a spend cap burns further iterations proving what
    the first 400 already said, and is then reported as "a configuration
    failure, not a transient one", pointing whoever reads it at auth flags
    rather than at the billing page.
    """

    from meta_filter.llm.base import UsageLimitError

    class _CappedProposer(_AlwaysFailingProposer):
        def propose(self, **kwargs):
            self.attempts += 1
            raise UsageLimitError(
                "claude headless refused: You have reached your specified API "
                "usage limits. You will regain access on 2026-10-01 at 00:00 UTC."
            )

    proposer = _CappedProposer()
    with pytest.raises(RuntimeError, match="out of allowance") as caught:
        _optimizer(tmp_path, proposer, iterations=20).run()

    assert proposer.attempts == 1, (
        "a cap must stop the run on the first failure, not after "
        f"{MAX_CONSECUTIVE_PROPOSER_FAILURES} strikes"
    )
    # The reason must carry the regain date, and must NOT send the reader to
    # the configuration.
    assert "2026-10-01" in str(caught.value)
    assert "configuration failure" not in str(caught.value)


def test_a_failure_on_the_last_iteration_is_not_a_clean_run(tmp_path):
    """One summary row short of the iteration count, and exit 0.

    When the last iteration's proposer dies, the skip-handler's `continue`
    falls out of the loop and the run returns normally — so the run record
    would say "complete" for a run that lost its final iteration. A single
    failure is still skipped (the test above), but a run may not *end* on
    one.
    """

    class _FailsOnlyOnTheLastIteration(_AlwaysFailingProposer):
        def propose(self, **kwargs):
            self.attempts += 1
            if self.attempts >= 3:  # iterations are 0-indexed; this is the last
                raise RuntimeError("claude headless exit 1: usage limits")
            return Proposer.propose(self, **kwargs)

    proposer = _FailsOnlyOnTheLastIteration()
    opt = _optimizer(tmp_path, proposer, iterations=3)
    proposer.client = opt.proposer.client if hasattr(opt.proposer, "client") else proposer.client

    with pytest.raises(RuntimeError, match="ended on a failed iteration") as caught:
        opt.run()

    # Guard against passing for the wrong reason: if all three iterations had
    # failed this would be the streak handler's "configuration failure", and
    # if none had it would not raise at all.
    assert proposer.attempts == 3
    assert "1 iteration(s)" in str(caught.value)
    assert "2 of 3 iterations" in str(caught.value)
