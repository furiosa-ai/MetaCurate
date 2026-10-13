"""What the scorer does when the model's answer cannot be read.

Scoring an unparseable response 0.0 and caching it is wrong twice over. A 0.0
is a real point on the rubric's -2..+2 scale, so a row the model never judged
would enter ranking as a confident middling verdict, and because the value is
written to the per-criterion cache the wrong number would be served for the
rest of the run without another call. On a rubric where most
cells sit above zero, a handful of these is enough to move a candidate's
concordance.

The behaviour: retry the identical prompt twice more (temperature is 0, so
a retry buys a re-roll of the sampler and nothing else, which is exactly what
a truncated or fenced response needs), and if all three attempts are
unreadable, fail the row. Nothing is cached, no score is invented, and the
count is on the scorer for the loop to report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from meta_filter.core.llm_scorer import (
    PARSE_ATTEMPTS,
    LLMRubricScorer,
    ScorerParseError,
    _parse_scorer_response,
)
from meta_filter.core.schemas import Trajectory
from meta_filter.llm.base import LLMResponse

RUBRIC = """\
# Rubric

## Task Grounding
The agent stays anchored to the stated task.

## Error Recovery
The agent notices failures and adapts.
"""

CRITERIA = ["Task Grounding", "Error Recovery"]


def _requested_criteria(prompt: str) -> list[str]:
    for line in prompt.splitlines():
        if line.startswith("Criteria to score (in order): "):
            return [c for c in line.removeprefix("Criteria to score (in order): ").split(", ") if c]
    raise AssertionError("prompt has no 'Criteria to score' line")


@dataclass
class _Client:
    """Emits `bad_first` unreadable responses per criterion, then valid JSON."""

    bad_first: int = 0
    model: str = "stub-model"
    bad_text: str = "I am not going to answer in JSON."
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    _seen: dict[str, int] = field(default_factory=dict)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        self.prompts.append(prompt)
        criteria = _requested_criteria(prompt)
        key = ",".join(criteria)
        self._seen[key] = self._seen.get(key, 0) + 1
        if self._seen[key] <= self.bad_first:
            return LLMResponse(text=self.bad_text, prompt_tokens=10, completion_tokens=5)
        body = json.dumps({"scores": {c: {"score": 1.0, "evidence": f"saw {c}"} for c in criteria}})
        return LLMResponse(text=body, prompt_tokens=10, completion_tokens=5)


def _traj(id_: str = "traj-1") -> Trajectory:
    # The id goes in the task text too: the prompt carries the task, not the
    # id, and a stub that has to fail one specific row needs a tell.
    return Trajectory(trajectory_id=id_, task=f"fix the bug in {id_}", turns=[])


def _scorer(tmp_path: Path, client: _Client) -> LLMRubricScorer:
    return LLMRubricScorer(client=client, cache_dir=tmp_path / "cache")


# --- the parser's own contract ----------------------------------------------


def test_unreadable_text_parses_to_nothing() -> None:
    assert _parse_scorer_response("sorry, no", CRITERIA) is None


def test_a_missing_criterion_is_a_failure_not_a_zero() -> None:
    """The half-answer case: valid JSON that simply omits what was asked."""

    text = json.dumps({"scores": {"Task Grounding": {"score": 1.0}}})
    assert _parse_scorer_response(text, CRITERIA) is None


def test_a_complete_answer_parses() -> None:
    text = json.dumps({"scores": {c: {"score": -1.0, "evidence": "e"} for c in CRITERIA}})
    parsed = _parse_scorer_response(text, CRITERIA)
    assert parsed is not None
    scores, evidence = parsed
    assert scores == {c: -1.0 for c in CRITERIA}
    assert evidence


# --- retry and failure ------------------------------------------------------


def test_one_bad_response_is_retried_and_the_row_still_scores(tmp_path: Path) -> None:
    client = _Client(bad_first=1)
    scorer = _scorer(tmp_path, client)

    scores, _evidence, cost = scorer.score(_traj(), CRITERIA, RUBRIC)

    assert scores == {c: 1.0 for c in CRITERIA}
    # Two criteria, one wasted attempt each, one good attempt each.
    assert client.calls == 4
    assert cost.llm_calls == 4
    assert scorer.parse_failures == 0


def test_three_bad_responses_fail_the_row(tmp_path: Path) -> None:
    client = _Client(bad_first=PARSE_ATTEMPTS)
    scorer = _scorer(tmp_path, client)

    with pytest.raises(ScorerParseError) as excinfo:
        scorer.score(_traj(), CRITERIA, RUBRIC)

    failure = excinfo.value.failure
    assert failure.trajectory_id == "traj-1"
    assert failure.criterion == "Task Grounding"
    assert failure.attempts == PARSE_ATTEMPTS
    # The prompt is identical on every attempt: a retry is a re-roll, not a
    # different question.
    assert len(set(client.prompts)) == 1
    assert client.calls == PARSE_ATTEMPTS


def test_a_failed_row_is_counted_and_named(tmp_path: Path) -> None:
    scorer = _scorer(tmp_path, _Client(bad_first=PARSE_ATTEMPTS))

    for row in ("a", "b"):
        with pytest.raises(ScorerParseError):
            scorer.score(_traj(row), CRITERIA, RUBRIC)

    assert scorer.parse_failures == 2
    assert scorer.failed_row_ids == {"a", "b"}

    scorer.reset_failure_stats()
    assert scorer.parse_failures == 0
    assert scorer.failed_row_ids == set()


def test_a_failure_is_logged_at_warning(tmp_path: Path, caplog) -> None:
    scorer = _scorer(tmp_path, _Client(bad_first=PARSE_ATTEMPTS))

    with caplog.at_level(logging.WARNING, logger="meta_filter.llm_scorer"):
        with pytest.raises(ScorerParseError):
            scorer.score(_traj(), CRITERIA, RUBRIC)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("traj-1" in r.getMessage() for r in warnings)


def test_nothing_is_cached_for_a_failed_criterion(tmp_path: Path) -> None:
    """The point of the whole fix: a failure must not become a stored zero."""

    client = _Client(bad_first=PARSE_ATTEMPTS)
    scorer = _scorer(tmp_path, client)
    with pytest.raises(ScorerParseError):
        scorer.score(_traj(), CRITERIA, RUBRIC)

    written = list((tmp_path / "cache" / "per_criterion").glob("*.json"))
    assert written == []

    # And a later attempt with a working model gets the real score rather than
    # a cached artefact of the failure.
    fresh = LLMRubricScorer(client=_Client(bad_first=0), cache_dir=tmp_path / "cache")
    scores, _evidence, _cost = fresh.score(_traj(), CRITERIA, RUBRIC)
    assert scores == {c: 1.0 for c in CRITERIA}


def test_a_criterion_scored_before_the_failure_stays_cached(tmp_path: Path) -> None:
    """Criterion 1 succeeded; only criterion 2 was unreadable."""

    class _SecondFails(_Client):
        def generate(self, prompt: str, **kwargs) -> LLMResponse:
            if "Error Recovery" in _requested_criteria(prompt):
                self.calls += 1
                self.prompts.append(prompt)
                return LLMResponse(text="nope", prompt_tokens=1, completion_tokens=1)
            return super().generate(prompt, **kwargs)

    scorer = _scorer(tmp_path, _SecondFails())
    with pytest.raises(ScorerParseError):
        scorer.score(_traj(), CRITERIA, RUBRIC)

    written = list((tmp_path / "cache" / "per_criterion").glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["criterion"] == "Task Grounding"


def test_prefetch_survives_a_failed_row_and_reports_it(tmp_path: Path, caplog) -> None:
    """A row the model will not score must not end the pass for the others."""

    class _OneRowFails(_Client):
        def generate(self, prompt: str, **kwargs) -> LLMResponse:
            if "traj-bad" in prompt:
                self.calls += 1
                return LLMResponse(text="nope", prompt_tokens=1, completion_tokens=1)
            return super().generate(prompt, **kwargs)

    scorer = _scorer(tmp_path, _OneRowFails())
    rows = [_traj("traj-good"), _traj("traj-bad")]
    with caplog.at_level(logging.WARNING, logger="meta_filter.llm_scorer"):
        scorer.prefetch_parallel(rows, CRITERIA, RUBRIC)

    assert scorer.failed_row_ids == {"traj-bad"}
    assert any("1 row" in r.getMessage() for r in caplog.records)
    # The good row is cached and needs no further call.
    calls_before = scorer.client.calls
    scorer.score(_traj("traj-good"), CRITERIA, RUBRIC)
    assert scorer.client.calls == calls_before
