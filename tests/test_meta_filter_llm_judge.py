"""LLM judge critic — SWE prompt + misfire field + select_for_proposer."""

from __future__ import annotations

import json
from dataclasses import dataclass

from meta_filter.core.schemas import Decision, Trajectory
from meta_filter.critics.trajectory_critic import (
    JUDGE_PROMPT,
    JUDGE_SYSTEM,
    TrajectoryCritic,
    TrajectoryVerdict,
    _parse_judge_response,
)
from meta_filter.llm.base import LLMResponse


@dataclass
class _StubClient:
    """Minimal LLMClient stub that returns a canned JSON body."""

    body: str
    model: str = "stub-model"
    calls: int = 0

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.body, prompt_tokens=1, completion_tokens=1)


def _traj(id_: str) -> Trajectory:
    return Trajectory(trajectory_id=id_, task="t", turns=[])


def test_judge_system_names_swe_and_misfire():
    """The judge preamble must name the SWE domain and the misfire concept."""

    assert "SWE" in JUDGE_SYSTEM
    assert "gold" in JUDGE_SYSTEM.lower() and "not have" in JUDGE_SYSTEM.lower()
    for tier in ("rubric", "rules", "policy"):
        assert tier in JUDGE_PROMPT.lower()
    assert "misfire" in JUDGE_PROMPT.lower()


def test_parse_judge_response_extracts_misfire():
    body = json.dumps(
        {
            "decision": "REJECT",
            "rationale": "patch is a stub",
            "closest_candidate": "cand_2",
            "misfire": "rubric",
            "misfire_note": "criterion `verification` under-fired",
        }
    )
    parsed = _parse_judge_response(body)
    assert parsed["decision"] is Decision.REJECT
    assert parsed["closest_candidate"] == "cand_2"
    assert parsed["misfire"] == "rubric"
    assert "under-fired" in parsed["misfire_note"]


def test_parse_judge_response_coerces_unknown_misfire_to_none():
    body = json.dumps({"decision": "ACCEPT", "misfire": "gibberish"})
    parsed = _parse_judge_response(body)
    assert parsed["decision"] is Decision.ACCEPT
    assert parsed["misfire"] == "none"


def test_misfire_survives_cache_roundtrip(tmp_path):
    body = json.dumps(
        {
            "decision": "REJECT",
            "rationale": "obvious hallucination",
            "closest_candidate": "cand_1",
            "misfire": "rules",
            "misfire_note": "rule_no_turns should have fired",
        }
    )
    backend = TrajectoryCritic(client=_StubClient(body), cache_dir=tmp_path)
    traj = _traj("t42")

    # First call: writes cache + populates in-memory verdict.
    dec = backend.critic_verdict(traj)
    assert dec is Decision.REJECT
    v = backend.verdict_for("t42")
    assert isinstance(v, TrajectoryVerdict)
    assert v.misfire == "rules"
    assert v.closest_candidate == "cand_1"

    # New backend against the same cache_dir must recover the same fields
    # from disk, without any LLM call.
    fresh = TrajectoryCritic(
        client=_StubClient("(should not be called)"),
        cache_dir=tmp_path,
    )
    reloaded = fresh.verdict_for("t42")
    assert reloaded is not None
    assert reloaded.decision is Decision.REJECT
    assert reloaded.misfire == "rules"
    assert reloaded.misfire_note.startswith("rule_no_turns")
    assert fresh.client.calls == 0


def test_verdict_for_returns_none_when_uncached(tmp_path):
    backend = TrajectoryCritic(
        client=_StubClient("{}"),
        cache_dir=tmp_path,
    )
    assert backend.verdict_for("never_seen") is None


def _seed_verdict(backend, traj_id: str, misfire: str) -> None:
    """Bypass the LLM by writing the cache file directly."""

    path = backend.cache_path_for(traj_id)
    path.write_text(
        json.dumps(
            {
                "trajectory_id": traj_id,
                "decision": "REJECT",
                "rationale": "seeded",
                "closest_candidate": "cand_1",
                "misfire": misfire,
                "misfire_note": "seed",
            }
        ),
        encoding="utf-8",
    )


def test_select_for_proposer_respects_bucket_counts(tmp_path):
    backend = TrajectoryCritic(client=_StubClient("{}"), cache_dir=tmp_path)
    # 3 FN, 3 FP, 2 audit-mismatch, 3 candidates with misfire cached.
    fn = [f"fn{i}" for i in range(3)]
    fp = [f"fp{i}" for i in range(3)]
    audit = [f"au{i}" for i in range(2)]
    instructive = [f"ins{i}" for i in range(3)]
    for tid in instructive:
        _seed_verdict(backend, tid, misfire="rubric")

    picked = backend.select_for_proposer(
        false_negative_ids=fn,
        false_positive_ids=fp,
        audit_mismatch_ids=audit,
        candidate_trajectory_ids=instructive,
        seed=17,
        budget=6,
    )
    assert len(picked) == 6
    fn_taken = [x for x in picked if x.startswith("fn")]
    fp_taken = [x for x in picked if x.startswith("fp")]
    au_taken = [x for x in picked if x.startswith("au")]
    ins_taken = [x for x in picked if x.startswith("ins")]
    assert len(fn_taken) == 2  # bucket cap
    assert len(fp_taken) == 2
    assert len(au_taken) == 1
    assert len(ins_taken) == 1


def test_select_for_proposer_backfills_from_largest_when_bucket_empty(tmp_path):
    backend = TrajectoryCritic(client=_StubClient("{}"), cache_dir=tmp_path)
    fn = [f"fn{i}" for i in range(5)]
    fp: list[str] = []
    audit: list[str] = []
    instructive: list[str] = []

    picked = backend.select_for_proposer(
        false_negative_ids=fn,
        false_positive_ids=fp,
        audit_mismatch_ids=audit,
        candidate_trajectory_ids=instructive,
        seed=17,
        budget=6,
    )
    # With everything else empty, FN must absorb the leftover slots
    # up to its 5-item pool (budget is 6 so we stop at 5 items).
    assert set(picked) <= set(fn)
    assert len(picked) == 5


def test_select_for_proposer_prefers_rubric_tier(tmp_path):
    backend = TrajectoryCritic(client=_StubClient("{}"), cache_dir=tmp_path)
    _seed_verdict(backend, "policy1", misfire="policy")
    _seed_verdict(backend, "rules1", misfire="rules")
    _seed_verdict(backend, "rubric1", misfire="rubric")

    picked = backend.select_for_proposer(
        false_negative_ids=[],
        false_positive_ids=[],
        audit_mismatch_ids=[],
        candidate_trajectory_ids=["policy1", "rules1", "rubric1"],
        seed=17,
        budget=1,
    )
    # Only one slot allocated to instructive → must be the rubric-tier case.
    assert picked == ["rubric1"]


def test_select_for_proposer_is_deterministic_under_seed(tmp_path):
    backend = TrajectoryCritic(client=_StubClient("{}"), cache_dir=tmp_path)
    for i in range(4):
        _seed_verdict(backend, f"ins{i}", misfire="rubric")
    fn = [f"fn{i}" for i in range(4)]
    fp = [f"fp{i}" for i in range(4)]
    audit = [f"au{i}" for i in range(3)]
    instructive = [f"ins{i}" for i in range(4)]

    args = dict(
        false_negative_ids=fn,
        false_positive_ids=fp,
        audit_mismatch_ids=audit,
        candidate_trajectory_ids=instructive,
        seed=123,
        budget=6,
    )
    picked_a = backend.select_for_proposer(**args)
    picked_b = backend.select_for_proposer(**args)
    assert picked_a == picked_b


def test_a_verdict_from_a_different_view_is_not_reused(tmp_path, monkeypatch):
    """A verdict formed on a thumbnail is not an opinion about the full transcript.

    Keying the cache on trajectory_id alone would let verdicts formed on an
    excerpt leak into later runs that render the full transcript.
    """

    import meta_filter.critics.trajectory_critic as judge_mod

    backend = TrajectoryCritic(client=_StubClient("{}"), cache_dir=tmp_path)
    _seed_verdict(backend, "trajA", misfire="rubric")
    assert backend.verdict_for("trajA") is not None

    monkeypatch.setattr(judge_mod, "JUDGE_TOOL_BUDGET", 999)
    widened = TrajectoryCritic(client=_StubClient("{}"), cache_dir=tmp_path)
    assert widened.verdict_for("trajA") is None


def test_two_judges_do_not_share_a_cache_entry(tmp_path):
    """Verdicts from two backends are opinions from two different readers."""

    opus = TrajectoryCritic(client=_StubClient("{}"), cache_dir=tmp_path, model_tag="opus")
    sonnet = TrajectoryCritic(client=_StubClient("{}"), cache_dir=tmp_path, model_tag="sonnet")
    _seed_verdict(opus, "trajA", misfire="rubric")
    assert opus.verdict_for("trajA") is not None
    assert sonnet.verdict_for("trajA") is None
