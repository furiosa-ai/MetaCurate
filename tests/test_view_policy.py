"""What each LLM role may read, and the cache key that records it.

Two halves, and the second is the load-bearing one:

* a rubric that declares a field gets it, subject to its role's ceiling, and
  the scorer's cache keys move so the widened score cannot be served from an
  entry taken without the field;
* a rubric that declares nothing renders byte for byte the baseline view and
  keys to the same strings, which is what keeps scorer entries and pair
  verdicts already on disk readable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from meta_filter.filter_loader import load_filter
from meta_filter.llm.base import LLMResponse
from meta_filter.llm_scorer import (
    LLMRubricScorer,
    _criterion_cache_key,
    criterion_field_sets,
)
from meta_filter.pair_judge import PAIR_METADATA_LINE, _meta_line
from meta_filter.schemas import Trajectory
from meta_filter.view_policy import (
    FIELD_CAPS,
    PAIR_JUDGE_FIELDS,
    SCORER_BASELINE_FIELDS,
    declared_fields,
    field_set_signature,
    fields_for,
    scorer_key_fields,
    scorer_metadata_block,
    scorer_reference_blocks,
    unrenderable,
)

ROOT = Path(__file__).resolve().parents[1]

# What the scorer renders undeclared, and the field set its cache-key
# signatures are computed over. Reproduced here rather than imported so the
# test fails if either moves without a deliberate decision: the second is part
# of every scorer cache key already on disk.
PINNED_SCORER_FIELDS = ("api_calls", "n_messages", "submission_bytes", "source")
PINNED_SCORER_KEY_FIELDS = (
    "api_calls",
    "n_messages",
    "submission_bytes",
    "exit_status",
    "instance_id",
    "source",
    "resolved",
    "f2p_pass",
    "p2p_pass",
)


def _traj(**metadata: object) -> Trajectory:
    return Trajectory(trajectory_id="t-1", task="fix the bug", turns=[], metadata=metadata)


# --- the baseline must not move ---------------------------------------------


def test_scorer_baseline_is_the_pinned_allow_list() -> None:
    assert SCORER_BASELINE_FIELDS == frozenset(PINNED_SCORER_FIELDS)
    assert scorer_key_fields(SCORER_BASELINE_FIELDS) == frozenset(PINNED_SCORER_KEY_FIELDS)


def test_the_metadata_block_renders_only_the_allow_listed_keys() -> None:
    metadata = {
        "api_calls": 23,
        "instance_id": "modin-project__modin-6404",
        "submission": "diff --git a/x b/x",  # present, must stay out
        "resolved": False,
        "host": "host-local",  # not in the allow-list
        "f2p_pass": None,
        "n_messages": 54,
        "source": "mini-swe-agent-1.1",
        "submission_bytes": 670,
        "p2p_pass": False,
        "exit_status": "Submitted",
    }
    pinned = json.dumps({k: v for k, v in metadata.items() if k in PINNED_SCORER_FIELDS}, indent=2)
    assert scorer_metadata_block(metadata, SCORER_BASELINE_FIELDS) == pinned


def test_baseline_signature_is_empty_and_adds_nothing_to_the_key() -> None:
    assert field_set_signature("scorer", SCORER_BASELINE_FIELDS) == ""
    traj = _traj()
    baseline = _criterion_cache_key("sec", "C", traj, "sonnet", "", "hdr")
    assert baseline == _criterion_cache_key("sec", "C", traj, "sonnet", header="hdr")
    assert "__meta_" not in baseline


def test_pair_judge_line_is_the_constant_whatever_the_row_carries() -> None:
    traj = _traj(exit_status="Submitted", resolved=True, problem_statement="text")
    assert _meta_line(traj) == PAIR_METADATA_LINE
    assert PAIR_METADATA_LINE == (
        '{"resolved": null, "f2p_pass": null, "p2p_pass": null, "exit_status": null}'
    )


# --- declaration widens, within the ceiling ---------------------------------


def test_declared_fields_reads_every_reads_comment() -> None:
    text = "## Criterion: X\n<!-- reads: submission, problem_statement -->\nanchors\n"
    assert declared_fields(text) == frozenset({"submission", "problem_statement"})
    assert declared_fields("no declaration here") == frozenset()


def test_declaration_widens_the_scorer_and_changes_its_signature() -> None:
    fields = fields_for("scorer", ["problem_statement"])
    assert fields == SCORER_BASELINE_FIELDS | {"problem_statement"}
    assert field_set_signature("scorer", fields) != ""


def test_ceiling_blocks_fields_the_role_may_not_render() -> None:
    # `test_patch` names the hidden tests; the rest are simply not declarable.
    fields = fields_for("scorer", ["problem_statement", "test_patch", "gold_patch", "bogus"])
    assert fields == SCORER_BASELINE_FIELDS | {"problem_statement"}
    assert unrenderable("scorer", ["test_patch", "gold_patch", "bogus"]) == frozenset(
        {"test_patch", "gold_patch", "bogus"}
    )


def test_pair_judge_cannot_be_widened_by_any_rubric() -> None:
    assert fields_for("pair_judge", ["submission", "problem_statement"]) == frozenset(
        PAIR_JUDGE_FIELDS
    )
    assert field_set_signature("pair_judge", fields_for("pair_judge", ["problem_statement"])) == ""


def test_unknown_role_is_an_error_not_a_silent_baseline() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        fields_for("proposer", ["problem_statement"])


def test_a_declared_diff_is_rendered_raw_not_json_escaped() -> None:
    """Long text inside `json.dumps` arrives on one line with `\\n` escapes,
    so it gets its own block instead."""

    patch = "The parser fails on:\n\n    x = 1\n\nExpected it to pass.\n"
    fields = fields_for("scorer", ["problem_statement"])
    metadata = {"problem_statement": patch}
    assert "problem_statement" not in json.loads(scorer_metadata_block(metadata, fields))
    block = scorer_reference_blocks(metadata, fields)
    assert patch in block  # raw, newlines intact
    assert "\\n" not in block
    assert block.startswith("\n\nFull problem statement")


def test_reference_block_is_empty_at_the_baseline() -> None:
    # Splicing an empty string in is what keeps the baseline prompt byte-identical.
    assert scorer_reference_blocks({"problem_statement": "diff"}, SCORER_BASELINE_FIELDS) == ""
    assert scorer_reference_blocks({}, fields_for("scorer", ["problem_statement"])) == ""


def test_declared_field_is_capped_and_baseline_fields_are_not() -> None:
    cap = FIELD_CAPS["problem_statement"]
    fields = fields_for("scorer", ["problem_statement"])
    traj = _traj(problem_statement="x" * (cap + 500), source="a" * 5000, api_calls=1)
    block = scorer_reference_blocks(traj.metadata, fields)
    assert "chars elided" in block
    assert "x" * cap in block
    assert "x" * (cap + 1) not in block
    # No cap entry means untouched, however long.
    assert json.loads(scorer_metadata_block(traj.metadata, fields))["source"] == "a" * 5000


# --- a filter without declarations -----------------------------------------


def test_a_filter_that_declares_nothing_is_byte_stable() -> None:
    loaded = load_filter(ROOT / "tests" / "fixtures" / "filters" / "six_criteria")
    criteria = loaded.policy_module.RUBRIC_CRITERIA
    sets = criterion_field_sets(loaded.rubric_md, criteria)
    assert set(sets) == set(criteria)
    for criterion, fields in sets.items():
        assert fields == SCORER_BASELINE_FIELDS, criterion
        assert field_set_signature("scorer", fields) == "", criterion


RUBRIC_TWO_SETS = """\
# Rubric

## Criterion: Plain Axis
Scored from the transcript alone.

## Criterion: Reference Axis
<!-- reads: problem_statement -->
Scored against the problem statement.
"""

CRITERIA_TWO_SETS = ["Plain Axis", "Reference Axis"]


def _requested(prompt: str) -> list[str]:
    for line in prompt.splitlines():
        if line.startswith("Criteria to score (in order): "):
            return [c for c in line.removeprefix("Criteria to score (in order): ").split(", ") if c]
    raise AssertionError("prompt has no 'Criteria to score' line")


@dataclass
class _StubClient:
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
                "scores": {c: {"score": 1.0, "evidence": "e"} for c in _requested(prompt)},
                "summary": "s",
            }
        )
        return LLMResponse(text=body, prompt_tokens=10, completion_tokens=5)


def test_one_llm_call_per_field_set_and_only_the_widened_one_sees_the_field(
    tmp_path: Path,
) -> None:
    client = _StubClient()
    scorer = LLMRubricScorer(client=client, cache_dir=tmp_path / "cache")
    traj = _traj(api_calls=3, problem_statement="THE-STATEMENT-VALUE")

    scorer.score(traj, CRITERIA_TWO_SETS, RUBRIC_TWO_SETS)

    # Two distinct field sets, so two calls — not one prompt carrying the
    # declared field while a baseline-keyed criterion is scored beside it.
    assert client.calls == 2
    by_criterion = {tuple(_requested(p)): p for p in client.prompts}
    plain = by_criterion[("Plain Axis",)]
    reference = by_criterion[("Reference Axis",)]
    # The *value* is what must not leak. Both prompts carry the whole rubric
    # as context (that is how `score_only` works), so the literal string
    # "problem_statement" appears in both via the declaration comment.
    assert "THE-STATEMENT-VALUE" in reference
    assert "THE-STATEMENT-VALUE" not in plain


def test_widened_criterion_does_not_read_a_baseline_cache_entry(tmp_path: Path) -> None:
    traj = _traj(api_calls=3, problem_statement="THE-STATEMENT-VALUE")
    plain_rubric = RUBRIC_TWO_SETS.replace("<!-- reads: problem_statement -->\n", "")

    first = _StubClient()
    LLMRubricScorer(client=first, cache_dir=tmp_path / "cache").score(
        traj, CRITERIA_TWO_SETS, plain_rubric
    )
    assert first.calls == 2  # one call per criterion

    # Same section text for Plain Axis, so it stays cached; Reference Axis now
    # declares a field, so its key moved and it must be re-asked.
    second = _StubClient()
    LLMRubricScorer(client=second, cache_dir=tmp_path / "cache").score(
        traj, CRITERIA_TWO_SETS, RUBRIC_TWO_SETS
    )
    assert second.calls == 1
    assert _requested(second.prompts[0]) == ["Reference Axis"]


def test_criterion_key_records_the_field_set() -> None:
    traj = _traj()
    baseline = _criterion_cache_key("section", "C", traj, "sonnet", "")
    widened = _criterion_cache_key("section", "C", traj, "sonnet", "deadbeef")
    assert baseline != widened
    assert "__meta_" not in baseline
    assert "__meta_deadbeef" in widened
