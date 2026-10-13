"""`fresh_draw` asks the model again and makes the answer THE cached answer.

The promotion re-score needs a fresh draw for cells the cache already answers
— the draw that won selection is biased upward and must not become the
baseline's number. Filing that draw under a salted key (`__salt_promo{k}`)
would leave the winning draw under the plain key, so the next iteration's
rubric-identical candidates would read the lucky cells while the baseline
carried the fresh ones, and `b − c` would measure scorer noise. The fresh draw
overwrites in place: one key, one value, read by everyone.

The unsalted key itself must not move — the literals below are what
`_criterion_cache_key` produces, and every cache on disk is keyed that way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from meta_filter.core.llm_scorer import LLMRubricScorer, _criterion_cache_key
from meta_filter.core.schemas import Trajectory
from meta_filter.llm.base import LLMResponse

# Computed on the pre-salt code (commit 4db77b0) and pasted, not derived.
KEY_FULL_BEFORE = (
    "row:1__crit_8c974bc6_9dbfbd4418__view_12289822__hdr_2c837072__meta_abc__model_mock-1"
)
KEY_BARE_BEFORE = "t1__crit_602c6094_920a25ef68__view_12289822"


def test_the_cache_key_is_unchanged_from_before_the_promotion_rescore_existed():
    full = _criterion_cache_key(
        "## Criterion: Effort\n+2 many",
        "Effort",
        Trajectory(trajectory_id="row:1"),
        "mock-1",
        "abc",
        "# Rubric",
    )
    bare = _criterion_cache_key("sec", "crit", Trajectory(trajectory_id="t1"))

    assert full == KEY_FULL_BEFORE
    assert bare == KEY_BARE_BEFORE


def test_there_is_no_salt_any_more():
    """A second key per cell is the defect; the signature must not offer one."""

    with pytest.raises(TypeError):
        _criterion_cache_key("sec", "crit", Trajectory(trajectory_id="t1"), salt="promo0")
    assert not hasattr(LLMRubricScorer, "cache_salt")


@dataclass
class _CountingClient:
    model: str = "stub"
    calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        self.calls += 1
        self.prompts.append(prompt)
        # `score` is per criterion; the prompt names the one it wants.
        criterion = next(
            line.removeprefix("Criteria to score (in order): ")
            for line in prompt.splitlines()
            if line.startswith("Criteria to score (in order): ")
        )
        body = json.dumps(
            {"scores": {criterion: {"score": float(self.calls), "evidence": f"call {self.calls}"}}}
        )
        return LLMResponse(text=body, prompt_tokens=1, completion_tokens=1)


RUBRIC = "# Rubric\n\n## Criterion: Effort\n+2 many actions, -2 none.\n"


def _scorer(tmp_path: Path) -> tuple[LLMRubricScorer, _CountingClient]:
    client = _CountingClient()
    return LLMRubricScorer(client=client, cache_dir=tmp_path / "cache"), client


def _cache_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "cache" / "per_criterion").glob("*.json"))


def test_fresh_draw_asks_again_and_the_answer_becomes_the_cached_one(tmp_path):
    scorer, client = _scorer(tmp_path)
    traj = Trajectory(trajectory_id="row-1", task="fix", turns=[])

    scores_first, _, _ = scorer.score(traj, ["Effort"], RUBRIC)
    assert client.calls == 1 and scores_first == {"Effort": 1.0}
    scorer.score(traj, ["Effort"], RUBRIC)
    assert client.calls == 1, "the plain path hits its cache"

    with scorer.fresh_draw():
        scores_fresh, _, _ = scorer.score(traj, ["Effort"], RUBRIC)
    assert client.calls == 2, "inside the block the cached cell is a miss"
    assert scores_fresh == {"Effort": 2.0}

    # After the block a plain read returns the FRESH value: the winning draw
    # is gone, not filed beside it.
    scores_again, evidence_again, _ = scorer.score(traj, ["Effort"], RUBRIC)
    assert client.calls == 2
    assert scores_again == {"Effort": 2.0}
    assert evidence_again and "call 2" in evidence_again[0]

    files = _cache_files(tmp_path)
    assert len(files) == 1, "one key per cell — no salted sibling"
    assert not any("__salt_" in f.name for f in files)
    assert json.loads(files[0].read_text(encoding="utf-8"))["score"] == 2.0

    # A second scorer on the same directory (a resumed run) reads the fresh one.
    resumed = LLMRubricScorer(client=_CountingClient(), cache_dir=tmp_path / "cache")
    scores_resumed, _, _ = resumed.score(traj, ["Effort"], RUBRIC)
    assert scores_resumed == {"Effort": 2.0}
    assert resumed.client.calls == 0


def test_one_block_pays_once_per_cell_even_when_it_reads_twice(tmp_path):
    """The optimizer prefetches and then runs the solver on the same rows."""

    scorer, client = _scorer(tmp_path)
    rows = [Trajectory(trajectory_id=f"row-{i}", task="fix", turns=[]) for i in range(3)]
    for row in rows:
        scorer.score(row, ["Effort"], RUBRIC)
    assert client.calls == 3

    with scorer.fresh_draw():
        scorer.prefetch_parallel(rows, ["Effort"], RUBRIC)
        assert client.calls == 6, "every cached cell re-drawn once"
        for row in rows:
            scorer.score(row, ["Effort"], RUBRIC)
        assert client.calls == 6, "the second read inside the block is served"
    for row in rows:
        scorer.score(row, ["Effort"], RUBRIC)
    assert client.calls == 6


def test_a_cell_first_scored_inside_the_block_is_simply_cached(tmp_path):
    scorer, client = _scorer(tmp_path)
    traj = Trajectory(trajectory_id="new-row", task="fix", turns=[])
    with scorer.fresh_draw():
        scores, _, _ = scorer.score(traj, ["Effort"], RUBRIC)
    assert client.calls == 1
    assert scorer.score(traj, ["Effort"], RUBRIC)[0] == scores
    assert client.calls == 1


def test_fresh_draw_is_restored_when_the_block_raises(tmp_path):
    scorer, client = _scorer(tmp_path)
    traj = Trajectory(trajectory_id="row-1", task="fix", turns=[])
    scorer.score(traj, ["Effort"], RUBRIC)
    try:
        with scorer.fresh_draw():
            raise KeyError("boom")
    except KeyError:
        pass
    assert scorer._fresh_keys is None
    scorer.score(traj, ["Effort"], RUBRIC)
    assert client.calls == 1, "outside the block the cache hits again"
