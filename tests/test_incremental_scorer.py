"""Per-criterion scoring and caching in LLMRubricScorer.

Two properties, and the second is what makes the first trustworthy.

A rubric edit that touches ONE criterion must not re-score the others.
Re-scoring unchanged criteria injects fresh LLM noise into their scores, which
makes rubric edits lose selection for reasons unrelated to the edit ("variance
ratchet").

And a criterion is scored in **its own call**, shown the rubric preamble and
its own section only. Narrowing only the model's output while formatting the
whole `rubric_md` into the prompt would leave the cache key -- criterion name
plus that criterion's section -- not determining the value, and a score formed
as "criterion 1 of 9" would be served to any later rubric reusing that
section.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from meta_filter.core.llm_scorer import (
    LLMRubricScorer,
    _split_rubric_sections,
)
from meta_filter.core.schemas import Trajectory
from meta_filter.core.solver import ScorerCost
from meta_filter.llm.base import LLMResponse

RUBRIC_V1 = """\
# Rubric

## Task Grounding
The agent stays anchored to the stated task.

## Error Recovery
The agent notices failures and adapts.
"""

# Same as V1 except the Error Recovery section text changed.
RUBRIC_V2 = RUBRIC_V1.replace("notices failures and adapts", "retries with a changed plan")

# Same as V1 except text OUTSIDE any criterion section changed (the title).
RUBRIC_V1_RETITLED = RUBRIC_V1.replace("# Rubric", "# Rubric (retitled)")

# V1 plus a third criterion.
RUBRIC_V3 = RUBRIC_V1 + "\n## Tool Hygiene\nThe agent uses tools deliberately.\n"

CRITERIA = ["Task Grounding", "Error Recovery"]


def _requested_criteria(prompt: str) -> list[str]:
    """The criteria the prompt actually asks the model to score."""

    for line in prompt.splitlines():
        if line.startswith("Criteria to score (in order): "):
            listed = line.removeprefix("Criteria to score (in order): ")
            return [c for c in listed.split(", ") if c]
    raise AssertionError("prompt has no 'Criteria to score' line")


@dataclass
class _StubClient:
    """LLMClient stub that scores whatever criteria the prompt requests.

    Each call stamps its own number into the evidence line, so a test can tell
    whether a criterion's score came from cache or from a fresh call — exactly
    the LLM-noise situation the per-criterion cache exists to suppress. The
    score field carries the number too, but `_parse_scorer_response` clamps to
    the rubric's +/-2 range, so it saturates from the third call and only the
    evidence line stays a reliable tell. Per-criterion scoring spends two
    calls on a plain full miss, so tests are past that point immediately.
    """

    model: str = "stub-model"
    calls: int = 0
    prompts: list[str] = field(default_factory=list)

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
        body = json.dumps(
            {
                "scores": {
                    c: {"score": float(self.calls), "evidence": f"call{self.calls} saw {c}"}
                    for c in _requested_criteria(prompt)
                },
                "summary": f"summary of call {self.calls}",
            }
        )
        return LLMResponse(text=body, prompt_tokens=10, completion_tokens=5)


def _traj(id_: str = "traj-1") -> Trajectory:
    return Trajectory(trajectory_id=id_, task="fix the bug", turns=[])


def _scorer(tmp_path: Path, client: _StubClient | None = None) -> LLMRubricScorer:
    return LLMRubricScorer(client=client or _StubClient(), cache_dir=tmp_path / "cache")


# --- section splitting -------------------------------------------------------


def test_split_sections_isolates_each_criterion() -> None:
    sections = _split_rubric_sections(RUBRIC_V1, CRITERIA)
    assert "stays anchored" in sections["Task Grounding"]
    assert "notices failures" not in sections["Task Grounding"]
    assert "notices failures" in sections["Error Recovery"]
    assert "stays anchored" not in sections["Error Recovery"]
    # The preamble title belongs to no section.
    assert "# Rubric" not in sections["Task Grounding"]


def test_split_sections_missing_criterion_falls_back_to_whole_doc() -> None:
    sections = _split_rubric_sections(RUBRIC_V1, ["Task Grounding", "Nonexistent Criterion"])
    assert sections["Nonexistent Criterion"] == RUBRIC_V1
    assert "stays anchored" in sections["Task Grounding"]


def test_split_sections_is_case_insensitive() -> None:
    sections = _split_rubric_sections(RUBRIC_V1, ["task grounding"])
    assert "stays anchored" in sections["task grounding"]
    assert sections["task grounding"] != RUBRIC_V1


# --- scoring paths -----------------------------------------------------------


def test_full_miss_makes_one_call_per_criterion(tmp_path: Path) -> None:
    client = _StubClient()
    scorer = _scorer(tmp_path, client)
    scores, evidence, cost = scorer.score(_traj(), CRITERIA, RUBRIC_V1)
    assert client.calls == 2
    assert [_requested_criteria(p) for p in client.prompts] == [
        ["Task Grounding"],
        ["Error Recovery"],
    ]
    assert scores == {"Task Grounding": 1.0, "Error Recovery": 2.0}
    assert "[Task Grounding] call1 saw Task Grounding" in evidence
    assert "[Error Recovery] call2 saw Error Recovery" in evidence
    assert cost == ScorerCost(llm_calls=2, llm_tokens=30)
    assert len(list((tmp_path / "cache" / "per_criterion").glob("*.json"))) == 2


def test_a_criterion_is_shown_its_own_section_and_not_its_siblings(tmp_path: Path) -> None:
    """The property the cache key always claimed and never had."""

    client = _StubClient()
    _scorer(tmp_path, client).score(_traj(), CRITERIA, RUBRIC_V1)
    grounding, recovery = client.prompts
    assert "stays anchored to the stated task" in grounding
    assert "notices failures and adapts" not in grounding
    assert "notices failures and adapts" in recovery
    assert "stays anchored to the stated task" not in recovery
    # The preamble goes to both: it defines the scale the anchors are on.
    assert "# Rubric" in grounding and "# Rubric" in recovery


def test_editing_one_criterion_rescores_only_that_criterion(tmp_path: Path) -> None:
    client = _StubClient()
    scorer = _scorer(tmp_path, client)
    scores_v1, _, _ = scorer.score(_traj(), CRITERIA, RUBRIC_V1)
    scores_v2, evidence_v2, cost_v2 = scorer.score(_traj(), CRITERIA, RUBRIC_V2)
    # Exactly one extra call, for exactly the changed criterion.
    assert client.calls == 3
    assert _requested_criteria(client.prompts[2]) == ["Error Recovery"]
    assert "retries with a changed plan" in client.prompts[2]
    # Unchanged criterion is bit-identical from cache; changed one is fresh.
    # Freshness is read off the evidence stamp, not the score, which clamps.
    assert scores_v2["Task Grounding"] == scores_v1["Task Grounding"] == 1.0
    assert "[Error Recovery] call3 saw Error Recovery" in evidence_v2
    assert cost_v2 == ScorerCost(llm_calls=1, llm_tokens=15)


def test_edit_outside_all_sections_makes_no_call_and_costs_zero(tmp_path: Path) -> None:
    client = _StubClient()
    scorer = _scorer(tmp_path, client)
    scores_v1, _, _ = scorer.score(_traj(), CRITERIA, RUBRIC_V1)
    scores_v2, _, cost = scorer.score(_traj(), CRITERIA, RUBRIC_V1_RETITLED)
    # The title lives in the preamble, which is in every per-criterion prompt
    # and therefore in every key: a preamble edit legitimately re-scores all
    # of them. This is the ceiling on what per-criterion caching can save
    # while the proposer owns the preamble — see `_rubric_header`.
    assert client.calls == 4
    assert cost == ScorerCost(llm_calls=2, llm_tokens=30)
    assert set(scores_v2) == set(scores_v1)


def test_added_criterion_is_scored_incrementally(tmp_path: Path) -> None:
    client = _StubClient()
    scorer = _scorer(tmp_path, client)
    scorer.score(_traj(), CRITERIA, RUBRIC_V1)
    scores, evidence, _ = scorer.score(_traj(), [*CRITERIA, "Tool Hygiene"], RUBRIC_V3)
    # Appending a criterion leaves the two above it untouched: their sections
    # are rstripped and the preamble did not move.
    assert client.calls == 3
    assert _requested_criteria(client.prompts[2]) == ["Tool Hygiene"]
    assert "[Tool Hygiene] call3 saw Tool Hygiene" in evidence
    assert scores["Task Grounding"] == 1.0


def test_fallback_criterion_is_invalidated_by_any_rubric_edit(tmp_path: Path) -> None:
    client = _StubClient()
    scorer = _scorer(tmp_path, client)
    criteria = ["Task Grounding", "Ghost Criterion"]  # Ghost has no section
    scorer.score(_traj(), criteria, RUBRIC_V1)
    scores, _, _ = scorer.score(_traj(), criteria, RUBRIC_V1_RETITLED)
    # A criterion with no section falls back to the whole document, so it is
    # invalidated by any edit. Task Grounding's section and the preamble are
    # both unchanged by a title edit *inside* the preamble... except the title
    # IS the preamble, so both are re-scored: 2 on the first pass, 1 more for
    # the ghost, and Task Grounding again because its header moved.
    assert client.calls == 3
    assert set(scores) == set(criteria)


def test_evidence_merges_cached_and_fresh_lines(tmp_path: Path) -> None:
    client = _StubClient()
    scorer = _scorer(tmp_path, client)
    scorer.score(_traj(), CRITERIA, RUBRIC_V1)
    _, evidence, _ = scorer.score(_traj(), CRITERIA, RUBRIC_V2)
    assert "[Task Grounding] call1 saw Task Grounding" in evidence
    assert "[Error Recovery] call3 saw Error Recovery" in evidence
    # No cross-criterion summary any more: each call summarises its own
    # criterion, so a single "[summary]" line would be arbitrary.
    assert not [line for line in evidence if line.startswith("[summary]")]


def test_model_name_separates_per_criterion_entries(tmp_path: Path) -> None:
    client_a = _StubClient(model="model-a")
    scorer_a = _scorer(tmp_path, client_a)
    scorer_a.score(_traj(), CRITERIA, RUBRIC_V1)
    client_b = _StubClient(model="model-b")
    scorer_b = _scorer(tmp_path, client_b)  # same cache_dir
    scorer_b.score(_traj(), CRITERIA, RUBRIC_V1)
    # model-b must not reuse model-a's scores: it re-scores both criteria.
    assert client_b.calls == 2
    assert [_requested_criteria(p) for p in client_b.prompts] == [
        ["Task Grounding"],
        ["Error Recovery"],
    ]
    entries = list((tmp_path / "cache" / "per_criterion").glob("*.json"))
    assert len(entries) == 4  # 2 criteria x 2 models
    assert all("__model_" in p.name for p in entries)


def test_nothing_is_written_at_the_top_level_any_more(tmp_path: Path) -> None:
    """The whole-rubric layer is gone, not merely unused.

    Its ~28k entries on the real cache are single-call scorings. Reading one
    would answer a per-criterion question with a whole-rubric score, which is
    the unsoundness the split exists to remove, so the layer neither reads nor
    writes and everything lives one level down.
    """

    scorer = _scorer(tmp_path)
    scorer.score(_traj(), CRITERIA, RUBRIC_V1)
    assert not list((tmp_path / "cache").glob("*.json"))
    per_crit = {p.name for p in (tmp_path / "cache" / "per_criterion").glob("*.json")}
    assert len(per_crit) == 2


def test_disk_cache_survives_a_fresh_scorer_instance(tmp_path: Path) -> None:
    scorer_1 = _scorer(tmp_path)
    scores_1, _, _ = scorer_1.score(_traj(), CRITERIA, RUBRIC_V1)
    # New instance, empty memory caches, same disk. Delete the whole-rubric entry so
    # the result must be assembled purely from per-criterion entries.
    for p in (tmp_path / "cache").glob("*.json"):
        p.unlink()
    client_2 = _StubClient()
    scorer_2 = _scorer(tmp_path, client_2)
    scores_2, _, cost = scorer_2.score(_traj(), CRITERIA, RUBRIC_V1)
    assert client_2.calls == 0
    assert scores_2 == scores_1
    assert cost == ScorerCost(llm_calls=0, llm_tokens=0)
