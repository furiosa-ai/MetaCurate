"""RubricCritic + RuleCritic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from meta_filter.core.schemas import Decision, Trajectory, Turn
from meta_filter.critics import (
    RubricCritic,
    RubricCriticReport,
    RuleCritic,
)
from meta_filter.critics._shared import CRITIC_PREAMBLE
from meta_filter.critics.rubric_critic import RUBRIC_CRITIC_SYSTEM
from meta_filter.critics.rule_critic import RULE_CRITIC_SYSTEM
from meta_filter.llm.base import LLMResponse


@dataclass
class _StubClient:
    """Canned-response stub. Records every prompt for inspection."""

    body: str
    model: str = "stub-model"
    calls: int = 0
    prompts: list[str] = field(default_factory=list)
    systems: list[str] = field(default_factory=list)

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
        self.systems.append(system or "")
        return LLMResponse(text=self.body, prompt_tokens=1, completion_tokens=1)


# --------------------------------------------------------------- shared preamble


def test_shared_preamble_appears_in_both_critic_systems():
    assert CRITIC_PREAMBLE in RUBRIC_CRITIC_SYSTEM
    assert CRITIC_PREAMBLE in RULE_CRITIC_SYSTEM
    assert "SWE" in CRITIC_PREAMBLE
    # Preamble must mention that gold labels don't exist — that's the whole
    # reason the critic exists.
    lower = CRITIC_PREAMBLE.lower()
    assert "gold labels" in lower and "not have" in lower


# ---------------------------------------------------------------- RubricCritic


def test_rubric_critic_parses_json_and_clamps_quality(tmp_path):
    body = json.dumps(
        {
            "rubric_quality": 1.7,  # deliberately out of range
            "issues": [
                {
                    "criterion": "Verification Completeness",
                    "problem": "anchor is vague",
                    "suggested_edit": "require post-edit test run",
                }
            ],
            "missing_criteria": ["regression risk", "root-cause vs symptom"],
        }
    )
    critic = RubricCritic(client=_StubClient(body), cache_dir=tmp_path)
    report = critic.review(rubric_md="# rubric v1", curated_evidence=[])

    assert isinstance(report, RubricCriticReport)
    assert report.rubric_quality == 1.0  # clamped
    assert report.issues[0]["criterion"] == "Verification Completeness"
    assert "regression risk" in report.missing_criteria
    assert report.from_cache is False


def test_rubric_critic_handles_malformed_response(tmp_path):
    critic = RubricCritic(client=_StubClient("not json at all"), cache_dir=tmp_path)
    report = critic.review(rubric_md="# rubric", curated_evidence=[])
    # Falls back to neutral quality + empty lists, does not raise.
    assert report.rubric_quality == 0.5
    assert report.issues == []
    assert report.missing_criteria == []


def test_rubric_critic_caches_on_content_hash(tmp_path):
    body = json.dumps({"rubric_quality": 0.7, "issues": [], "missing_criteria": []})
    client = _StubClient(body)
    critic = RubricCritic(client=client, cache_dir=tmp_path)

    critic.review(rubric_md="rubric text", curated_evidence=[{"traj": "a"}])
    critic.review(rubric_md="rubric text", curated_evidence=[{"traj": "a"}])
    assert client.calls == 1  # second call served from cache

    # Different content → new LLM call.
    critic.review(rubric_md="rubric text v2", curated_evidence=[{"traj": "a"}])
    assert client.calls == 2


def test_rubric_critic_evidence_lands_in_prompt(tmp_path):
    body = json.dumps({"rubric_quality": 0.5})
    client = _StubClient(body)
    critic = RubricCritic(client=client, cache_dir=tmp_path)
    critic.review(
        rubric_md="# rubric",
        curated_evidence=[{"trajectory_id": "traj42", "misfire": "rubric"}],
    )
    assert "traj42" in client.prompts[0]
    assert "misfire" in client.prompts[0]


# ------------------------------------------------------------------ RuleCritic


def _t(id_: str, turn_count: int = 1) -> Trajectory:
    return Trajectory(
        trajectory_id=id_,
        task="t",
        turns=[Turn(role="user", content="hi")] * turn_count,
    )


def _rule_no_turns(traj):
    if len(traj.turns) == 0:
        return {
            "name": "no_turns",
            "verdict": "REJECT",
            "evidence": ["0 turns"],
        }
    return None


def _rule_short_task(traj):
    if len(traj.task) <= 2:
        return {
            "name": "short_task",
            "verdict": "REJECT",
            "evidence": [f"task='{traj.task}'"],
        }
    return None


def _rule_never_fires(traj):
    return None


def test_rule_critic_single_rule_keeps_quality_neutral_but_still_explains(tmp_path):
    """A one-rule filter must not report a bare 0.5 with nothing behind it.

    Redundancy needs a second rule, so the aggregate stays neutral. Precision,
    recall and the LLM's read of them do not — suppressing those would make
    every single-rule iteration unreviewable.
    """

    critic = RuleCritic(
        client=_StubClient(
            json.dumps(
                {
                    "rules": [
                        {
                            "name": "_rule_no_turns",
                            "recommendation": "keep",
                            "note": "cheap and exact",
                        }
                    ],
                    "signals_missing_from_rules": ["empty patch"],
                }
            )
        ),
        cache_dir=tmp_path,
    )
    report = critic.review(
        rules_source="def rule_x(): pass",
        rule_functions=[_rule_no_turns],
        trajectories=[_t("a", turn_count=0), _t("b", turn_count=3)],
        truth_map={"a": Decision.REJECT, "b": Decision.ACCEPT},
    )
    assert report.rule_quality == 0.5
    assert [r.name for r in report.per_rule] == ["_rule_no_turns"]
    assert report.per_rule[0].precision == 1.0
    assert report.llm_used is True
    assert report.recommendations[0]["recommendation"] == "keep"


def test_rule_critic_with_no_rules_at_all_is_neutral_and_empty(tmp_path):
    report = RuleCritic(cache_dir=tmp_path).review(
        rules_source="",
        rule_functions=[],
        trajectories=[_t("a")],
        truth_map={"a": Decision.ACCEPT},
    )
    assert report.rule_quality == 0.5
    assert report.per_rule == []


def test_rule_critic_precision_recall_deterministic(tmp_path):
    trajectories = [
        _t("empty1", turn_count=0),
        _t("empty2", turn_count=0),
        _t("full1", turn_count=3),
        _t("full2", turn_count=3),
    ]
    truth_map = {
        "empty1": Decision.REJECT,  # rule fires, truth agrees → precision hit
        "empty2": Decision.REJECT,  # rule fires, truth agrees
        "full1": Decision.ACCEPT,  # rule silent, truth disagrees w/ rule verdict
        "full2": Decision.REJECT,  # rule silent BUT truth==REJECT → missed fire (recall miss)
    }
    critic = RuleCritic(cache_dir=tmp_path)
    report = critic.review(
        rules_source="src",
        rule_functions=[_rule_no_turns, _rule_never_fires],
        trajectories=trajectories,
        truth_map=truth_map,
    )
    stats = {r.name: r for r in report.per_rule}
    assert stats["_rule_no_turns"].fires == 2
    assert stats["_rule_no_turns"].truth_agreed == 2
    assert stats["_rule_no_turns"].precision == 1.0
    # 2 truth-agreed fires, 1 missed (full2) → recall = 2/3
    assert abs(stats["_rule_no_turns"].recall - (2 / 3)) < 1e-4
    # _rule_never_fires never fires → precision defaults to 0, recall 0.
    assert stats["_rule_never_fires"].fires == 0
    assert stats["_rule_never_fires"].precision == 0.0
    # Precision over the firing rules is perfect, but they only decide half
    # the labelled pool, and `rule_quality` is now scaled by that coverage:
    # 1.0 * (0.25 + 0.75 * 0.5) = 0.625. Crediting perfect precision on a
    # sliver of the pool as a perfect score is what let two runs collapse
    # their rule sets — see tests/test_objective_gradients.py.
    assert report.rule_quality == pytest.approx(0.625)
    assert report.llm_used is False


def test_rule_critic_redundancy_detected(tmp_path):
    # Two rules that fire on the exact same trajectory set → redundancy 1.0.
    trajectories = [_t("empty1", turn_count=0), _t("full", turn_count=3)]
    truth_map = {"empty1": Decision.REJECT, "full": Decision.ACCEPT}

    def _duplicate_no_turns(traj):
        if len(traj.turns) == 0:
            return {"name": "dup", "verdict": "REJECT", "evidence": []}
        return None

    critic = RuleCritic(cache_dir=tmp_path)
    report = critic.review(
        rules_source="src",
        rule_functions=[_rule_no_turns, _duplicate_no_turns],
        trajectories=trajectories,
        truth_map=truth_map,
    )
    stats = {r.name: r for r in report.per_rule}
    assert stats["_rule_no_turns"].redundancy == 1.0
    assert stats["_duplicate_no_turns"].redundancy == 1.0
    # Both contribute precision * (1 - 1.0) = 0 → aggregate quality = 0.
    assert report.rule_quality == 0.0


def test_rule_critic_llm_pass_is_opt_in(tmp_path):
    """No client → no LLM call, no recommendations, llm_used False."""

    critic = RuleCritic(cache_dir=tmp_path)
    report = critic.review(
        rules_source="src",
        rule_functions=[_rule_no_turns, _rule_short_task],
        trajectories=[_t("a", turn_count=0), _t("b", turn_count=3)],
        truth_map={"a": Decision.REJECT, "b": Decision.ACCEPT},
    )
    assert report.llm_used is False
    assert report.recommendations == []
    assert report.signals_missing_from_rules == []


def test_rule_critic_llm_pass_populates_recommendations(tmp_path):
    body = json.dumps(
        {
            "rules": [
                {
                    "name": "_rule_no_turns",
                    "recommendation": "keep",
                    "note": "cheap and precise",
                },
                {
                    "name": "_rule_short_task",
                    "recommendation": "drop",
                    "note": "trivially satisfied by empty tasks",
                },
            ],
            "signals_missing_from_rules": [
                "detect patch that touches unrelated files",
            ],
        }
    )
    client = _StubClient(body)
    critic = RuleCritic(client=client, cache_dir=tmp_path)
    report = critic.review(
        rules_source="src",
        rule_functions=[_rule_no_turns, _rule_short_task],
        trajectories=[_t("a", turn_count=0), _t("b", turn_count=3)],
        truth_map={"a": Decision.REJECT, "b": Decision.ACCEPT},
    )
    assert report.llm_used is True
    assert len(report.recommendations) == 2
    assert report.recommendations[0]["recommendation"] == "keep"
    assert "patch that touches" in report.signals_missing_from_rules[0]
    assert client.calls == 1

    # Same inputs → cache hit.
    report2 = critic.review(
        rules_source="src",
        rule_functions=[_rule_no_turns, _rule_short_task],
        trajectories=[_t("a", turn_count=0), _t("b", turn_count=3)],
        truth_map={"a": Decision.REJECT, "b": Decision.ACCEPT},
    )
    assert report2.from_cache is True
    assert client.calls == 1


def test_rule_critic_llm_recommendation_invalid_coerces_to_keep(tmp_path):
    body = json.dumps(
        {
            "rules": [
                {"name": "r1", "recommendation": "banana", "note": ""},
            ],
        }
    )
    critic = RuleCritic(client=_StubClient(body), cache_dir=tmp_path)
    report = critic.review(
        rules_source="src",
        rule_functions=[_rule_no_turns, _rule_short_task],
        trajectories=[_t("a", turn_count=0), _t("b", turn_count=3)],
        truth_map={"a": Decision.REJECT, "b": Decision.ACCEPT},
    )
    assert report.recommendations[0]["recommendation"] == "keep"


def test_rule_critic_report_serialisable(tmp_path):
    critic = RuleCritic(cache_dir=tmp_path)
    report = critic.review(
        rules_source="src",
        rule_functions=[_rule_no_turns, _rule_short_task],
        trajectories=[_t("a", turn_count=0)],
        truth_map={"a": Decision.REJECT},
    )
    payload = report.to_dict()
    # Round-trip through JSON — everything the optimizer will persist
    # must be JSON-friendly.
    Path(tmp_path / "out.json").write_text(json.dumps(payload))
    assert "rule_quality" in payload
    assert isinstance(payload["per_rule"], list)


def test_rubric_report_serialisable(tmp_path):
    body = json.dumps({"rubric_quality": 0.6, "issues": [], "missing_criteria": []})
    critic = RubricCritic(client=_StubClient(body), cache_dir=tmp_path)
    report = critic.review(rubric_md="# r", curated_evidence=[])
    payload = report.to_dict()
    Path(tmp_path / "out.json").write_text(json.dumps(payload))
    assert payload["rubric_quality"] == 0.6
