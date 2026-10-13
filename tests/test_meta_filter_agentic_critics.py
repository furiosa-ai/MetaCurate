"""Agentic critics — verify tools, workspace contents, and parsing reuse."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from meta_filter.core.schemas import Decision, Trajectory, Turn
from meta_filter.critics.agentic import AgenticRubricCritic, AgenticTrajectoryCritic
from meta_filter.llm.base import LLMResponse
from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient


@dataclass
class _RecordingAgent(ClaudeCodeHeadlessClient):
    """Captures the cwd it was handed and what it could see there."""

    reply: str = "{}"
    calls: list[dict] = field(default_factory=list)

    def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        cwd = Path(self.scratch_dir)
        self.calls.append(
            {
                "prompt": prompt,
                "tools": self.allowed_tools,
                "cwd": cwd,
                "visible": sorted(p.name for p in cwd.iterdir()) if cwd.is_dir() else [],
            }
        )
        return LLMResponse(text=self.reply, prompt_tokens=1, completion_tokens=1, model="opus")


def _trajectory() -> Trajectory:
    return Trajectory(
        trajectory_id="acme__widget-1::att0",
        task="fix the widget",
        turns=[Turn(role="assistant", action="bash", content="pytest -q") for _ in range(40)],
        metadata={"exit_status": "submitted"},
    )


def test_judge_gets_full_trajectory_on_disk(tmp_path):
    reply = json.dumps(
        {
            "decision": "ACCEPT",
            "rationale": "verified the fix",
            "closest_candidate": "",
            "misfire": "none",
            "misfire_note": "",
        }
    )
    agent = _RecordingAgent(model="opus", scratch_dir=tmp_path, reply=reply)
    backend = AgenticTrajectoryCritic(client=agent, cache_dir=tmp_path / "cache")

    verdict, raw = backend._query(_trajectory(), None)

    assert verdict.decision == Decision.ACCEPT
    assert verdict.rationale == "verified the fix"
    call = agent.calls[0]
    assert call["tools"] == ("Read", "Grep", "Glob", "Bash")
    assert call["visible"] == ["trajectory.json"]
    assert "trajectory.json" in call["prompt"]


def test_judge_workspace_holds_every_turn(tmp_path):
    captured = {}

    @dataclass
    class _Peeker(_RecordingAgent):
        def generate(self, prompt, **kw):
            payload = json.loads((Path(self.scratch_dir) / "trajectory.json").read_text())
            captured["n_turns"] = len(payload["turns"])
            return super().generate(prompt, **kw)

    agent = _Peeker(model="opus", scratch_dir=tmp_path, reply="{}")
    backend = AgenticTrajectoryCritic(client=agent, cache_dir=tmp_path / "cache")
    backend._query(_trajectory(), None)

    # The static prompt shows 6; the agent must be able to reach all 40.
    assert captured["n_turns"] == 40


def test_rubric_critic_is_rooted_at_the_run_dir(tmp_path):
    run_dir = tmp_path / "run_x"
    (run_dir / "iterations" / "000").mkdir(parents=True)
    (run_dir / "summary.csv").write_text("iteration,score\n0,0.1\n")

    reply = json.dumps({"rubric_quality": 0.8, "issues": [], "missing_criteria": ["tests added"]})
    agent = _RecordingAgent(model="opus", scratch_dir=tmp_path, reply=reply)
    critic = AgenticRubricCritic(
        client=agent, cache_dir=tmp_path / "cache", run_dir=run_dir, model_tag="opus"
    )

    report = critic.review("# rubric\nCriterion: A\n", [])

    assert report.rubric_quality == 0.8
    assert report.missing_criteria == ["tests added"]
    call = agent.calls[0]
    assert call["cwd"] == run_dir
    assert "iterations" in call["visible"] and "summary.csv" in call["visible"]


def test_rubric_critic_without_run_dir_stays_static(tmp_path):
    agent = _RecordingAgent(
        model="opus", scratch_dir=tmp_path, reply=json.dumps({"rubric_quality": 0.4})
    )
    critic = AgenticRubricCritic(client=agent, cache_dir=tmp_path / "cache", model_tag="opus")

    report = critic.review("# rubric\n", [])

    assert report.rubric_quality == 0.4
    assert agent.calls[0]["tools"] == ()
