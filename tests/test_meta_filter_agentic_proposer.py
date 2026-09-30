"""AgenticProposer — discovers candidates the agent wrote to disk."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from meta_filter.agentic_proposer import AgenticProposer
from meta_filter.filter_loader import load_filter, materialize_candidate
from meta_filter.llm.base import LLMResponse
from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient
from meta_filter.schemas import FilterMetrics

GOOD_RULES = "def rule_one(t):\n    return None\n"
GOOD_POLICY = "def decide(*a, **k):\n    return {'decision': 'REVIEW'}\n"


def _empty_metrics() -> FilterMetrics:
    return FilterMetrics(
        total=0,
        critic_accept_total=0,
        critic_reject_total=0,
        accepted=0,
        rejected=0,
        reviewed=0,
        agree_accept=0,
        agree_reject=0,
        filter_accept_critic_reject=0,
        filter_reject_critic_accept=0,
        critic_reject_recall=0.0,
        critic_accept_retention=0.0,
        disputed_accept_rate=0.0,
        disputed_reject_rate=0.0,
        review_rate=0.0,
    )


def _run_dir(tmp_path: Path, iteration: int = 0) -> tuple[Path, Path]:
    run_dir = tmp_path / "run_x"
    candidates = run_dir / "iterations" / f"{iteration:03d}" / "candidates"
    candidates.mkdir(parents=True)
    # The optimizer copies the baseline in before calling the proposer.
    materialize_candidate(
        candidates / f"iter{iteration:03d}_baseline",
        rubric_md="# base",
        rules_py=GOOD_RULES,
        policy_py=GOOD_POLICY,
    )
    return run_dir, candidates


@dataclass
class _ScriptedAgent(ClaudeCodeHeadlessClient):
    """Stands in for the CLI: writes files, then returns a transcript.

    The proposer copies its client (workspace + tools), so both fields are
    shared references — the original still observes what the copy did.
    """

    writer: Callable[[], None] = lambda: None
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        self.prompts.append(prompt)
        self.writer()
        return LLMResponse(text="done", prompt_tokens=1, completion_tokens=1, model="opus")


def _propose(proposer, candidates, iteration=0):
    return proposer.propose(
        loaded_filter=load_filter(candidates / f"iter{iteration:03d}_baseline"),
        candidates_root=candidates,
        iteration=iteration,
        current_metrics=_empty_metrics(),
        recent_false_negatives=[],
        recent_false_positives=[],
    )


def test_discovers_written_candidates_and_ignores_baseline(tmp_path):
    run_dir, candidates = _run_dir(tmp_path)

    def write_two():
        for i, slug in enumerate(["tighten_review", "drop_dead_rule"]):
            materialize_candidate(
                candidates / f"iter000_{i:02d}_{slug}",
                rubric_md=f"# {slug}",
                rules_py=GOOD_RULES,
                policy_py=GOOD_POLICY,
            )
        (candidates / "diagnosis.json").write_text(
            json.dumps(
                {
                    "diagnosis": "review band is too wide",
                    "candidates": [
                        {"name": "tighten_review", "notes": "narrow the band"},
                        {"name": "drop_dead_rule", "notes": "rule never fires"},
                    ],
                }
            )
        )

    proposer = AgenticProposer(
        client=_ScriptedAgent(writer=write_two, model="opus", scratch_dir=tmp_path / "scratch"),
        run_dir=run_dir,
        n_candidates=2,
    )
    result = _propose(proposer, candidates)

    names = sorted(p.name for p in result.candidate_dirs)
    assert names == ["iter000_00_tighten_review", "iter000_01_drop_dead_rule"]
    assert result.parse_errors == []
    rationale = (candidates / "iter000_00_tighten_review" / "rationale.md").read_text()
    assert "narrow the band" in rationale
    assert "review band is too wide" in rationale


def test_unloadable_candidate_is_rejected_not_raised(tmp_path):
    run_dir, candidates = _run_dir(tmp_path)

    def write_one_broken():
        materialize_candidate(
            candidates / "iter000_00_broken",
            rubric_md="# b",
            rules_py="def rule_x(",  # syntax error
            policy_py=GOOD_POLICY,
        )
        materialize_candidate(
            candidates / "iter000_01_fine",
            rubric_md="# f",
            rules_py=GOOD_RULES,
            policy_py=GOOD_POLICY,
        )

    proposer = AgenticProposer(
        client=_ScriptedAgent(writer=write_one_broken, model="opus", scratch_dir=tmp_path / "s"),
        run_dir=run_dir,
        n_candidates=2,
    )
    result = _propose(proposer, candidates)

    assert [p.name for p in result.candidate_dirs] == ["iter000_01_fine"]
    assert any("iter000_00_broken" in e for e in result.parse_errors)


def test_prompt_tells_agent_where_to_write(tmp_path):
    run_dir, candidates = _run_dir(tmp_path, iteration=4)
    agent = _ScriptedAgent(model="opus", scratch_dir=tmp_path / "s")
    proposer = AgenticProposer(client=agent, run_dir=run_dir, n_candidates=3)
    result = _propose(proposer, candidates, iteration=4)

    assert "iterations/004/candidates" in agent.prompts[0]
    assert "iter004_00_" in agent.prompts[0]
    # No candidates written at all must surface as an error, not an exception.
    assert result.candidate_dirs == []
    assert any("no loadable candidate" in e for e in result.parse_errors)


# --- leg 2: the one revision round ------------------------------------------


def _write_candidate(candidates: Path, name: str, rubric: str = "# c") -> Path:
    materialize_candidate(
        candidates / name, rubric_md=rubric, rules_py=GOOD_RULES, policy_py=GOOD_POLICY
    )
    return candidates / name


def test_the_revision_leg_rescans_disk_and_counts_as_the_second_leg(tmp_path):
    run_dir, candidates = _run_dir(tmp_path)
    agent = _ScriptedAgent(
        writer=lambda: _write_candidate(candidates, "iter000_00_first"),
        model="opus",
        scratch_dir=tmp_path / "s",
    )
    proposer = AgenticProposer(client=agent, run_dir=run_dir, n_candidates=1)
    first = _propose(proposer, candidates)
    assert first.legs == 1

    # The revision adds a second candidate and edits the first.
    agent.writer = lambda: _write_candidate(candidates, "iter000_01_second")
    revised = proposer.revise(
        result=first,
        precheck_block="## PRECHECK\n\niter000_00_first — score 0.4000\n",
        candidates_root=candidates,
        iteration=0,
    )

    assert revised.legs == 2
    assert sorted(p.name for p in revised.candidate_dirs) == [
        "iter000_00_first",
        "iter000_01_second",
    ]
    assert len(agent.prompts) == 2


def test_the_revision_prompt_carries_the_precheck_and_says_it_is_the_last(tmp_path):
    run_dir, candidates = _run_dir(tmp_path)
    agent = _ScriptedAgent(
        writer=lambda: _write_candidate(candidates, "iter000_00_first"),
        model="opus",
        scratch_dir=tmp_path / "s",
    )
    proposer = AgenticProposer(client=agent, run_dir=run_dir, n_candidates=1)
    first = _propose(proposer, candidates)
    agent.writer = lambda: None
    proposer.revise(
        result=first,
        precheck_block="## PRECHECK\n\nconcordance 0.61 over 40 pairs\n",
        candidates_root=candidates,
        iteration=0,
    )

    leg2 = agent.prompts[1]
    assert "concordance 0.61 over 40 pairs" in leg2
    assert "ONLY revision round" in leg2
    assert "no further precheck" in leg2
    assert "precheck.json" in leg2


def test_leg_one_is_told_a_precheck_and_a_revision_are_coming(tmp_path):
    run_dir, candidates = _run_dir(tmp_path)
    agent = _ScriptedAgent(model="opus", scratch_dir=tmp_path / "s")
    proposer = AgenticProposer(client=agent, run_dir=run_dir, n_candidates=1)
    _propose(proposer, candidates)

    assert "one precheck, then one revision" in agent.prompts[0]
    assert "No judge is called for it" in agent.prompts[0]
    assert "exactly ONE more call" in agent.prompts[0]


def test_leg_two_is_a_retry_when_leg_one_wrote_nothing_loadable(tmp_path):
    """No candidates means nothing to precheck, so the last call buys one."""

    run_dir, candidates = _run_dir(tmp_path)
    agent = _ScriptedAgent(
        writer=lambda: materialize_candidate(
            candidates / "iter000_00_broken",
            rubric_md="# b",
            rules_py="def rule_x(",
            policy_py=GOOD_POLICY,
        ),
        model="opus",
        scratch_dir=tmp_path / "s",
    )
    proposer = AgenticProposer(client=agent, run_dir=run_dir, n_candidates=1)
    first = _propose(proposer, candidates)
    assert first.candidate_dirs == []

    agent.writer = lambda: _write_candidate(candidates, "iter000_01_recovered")
    retried = proposer.revise(
        result=first, precheck_block="", candidates_root=candidates, iteration=0
    )

    assert "did not load" in agent.prompts[1]
    assert "iter000_00_broken" in agent.prompts[1]
    assert "ONLY revision round" not in agent.prompts[1]
    assert [p.name for p in retried.candidate_dirs] == ["iter000_01_recovered"]
    assert retried.legs == 2


def test_the_revision_never_adopts_the_baseline_as_a_candidate(tmp_path):
    run_dir, candidates = _run_dir(tmp_path)
    agent = _ScriptedAgent(
        writer=lambda: _write_candidate(candidates, "iter000_00_first"),
        model="opus",
        scratch_dir=tmp_path / "s",
    )
    proposer = AgenticProposer(client=agent, run_dir=run_dir, n_candidates=1)
    first = _propose(proposer, candidates)
    agent.writer = lambda: None
    revised = proposer.revise(
        result=first, precheck_block="x", candidates_root=candidates, iteration=0
    )

    assert all("baseline" not in p.name for p in revised.candidate_dirs)


def test_the_static_proposer_declines_the_revision_leg(tmp_path):
    """It answers from its prompt alone; there is no workspace to revise in."""

    from meta_filter.llm.base import LLMResponse as _R
    from meta_filter.proposer import Proposer, ProposerResult

    result = ProposerResult(prompt="p", response=_R(text="{}"), candidate_dirs=[])
    same = Proposer().revise(
        result=result, precheck_block="x", candidates_root=tmp_path, iteration=0
    )

    assert same is result
    assert same.legs == 1


def test_a_ranking_prompt_is_pairwise_end_to_end(tmp_path):
    """The whole agent-facing prompt, body and mission together.

    The body and the mission are built by different modules from different
    flags, so each one being right on its own is not the property that
    matters: what the agent reads is the concatenation.
    """

    run_dir, candidates = _run_dir(tmp_path)
    agent = _ScriptedAgent()
    proposer = AgenticProposer(client=agent, run_dir=run_dir, n_candidates=2)
    _propose(proposer, candidates)

    prompt = agent.prompts[0]

    assert "### Misordered pairs" in prompt
    assert "no judged pairs yet" in prompt
    assert "Top-of-ordering composition" in prompt
    for per_row in ("false negative", "false positive", "review_rate", "must-accept"):
        assert per_row not in prompt.lower(), per_row
