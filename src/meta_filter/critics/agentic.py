"""Agentic variants of the two LLM critics.

Both static critics answer from a summary we pasted into their prompt, and
both summaries throw away most of what they are judging:

* the judge sees 6 of N turns truncated to 220 characters each, plus the
  first 800 bytes of the submission — then rules on whether the trajectory
  is good training data;
* the rubric critic reads `rubric.md` next to a handful of curated cases,
  never the scores the rubric actually produced across the whole pool.

These variants keep the prompts and the parsing, and change only where the
evidence comes from: the model gets read tools and a directory, and goes
looking. Meta-Harness does the same for its proposer — the working
directory is the sandbox, and there is no path allow-list inside it.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from meta_filter.critics.rubric_critic import RUBRIC_CRITIC_SYSTEM, RubricCritic
from meta_filter.critics.trajectory_critic import JUDGE_SYSTEM, TrajectoryCritic
from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient
from meta_filter.trajectory import Trajectory

logger = logging.getLogger(__name__)

# Read-only tool set. Neither critic has any business writing: the judge
# rules on a trajectory, the rubric critic reports on a rubric. Bash is
# included because grep/wc over a long transcript is how you actually count
# things, but nothing here should mutate the run.
CRITIC_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob", "Bash")

_JUDGE_TOOL_NOTE = """\

## Before you answer

The summary above is lossy — it shows a few turns, truncated, and the head
of the submission. The complete trajectory is on disk at `trajectory.json`
in your working directory: every turn, every action, the full submission.

Read it before ruling. Things the summary cannot tell you but the full
record can: whether an edit was ever verified, whether the same failing
command repeats late in the run, whether the submission actually addresses
the stated task, whether "done" was declared without a test ever passing.

Then answer with exactly the JSON object described above and nothing else.
"""

_RUBRIC_TOOL_NOTE = """\

## Before you answer

Your working directory is the run's own output tree. The curated cases
above are a sample; the actual per-trajectory scoring is all on disk:

  iterations/<NNN>/solver_outputs/   every trajectory's rubric_scores and
                                     the decision they produced
  iterations/<NNN>/critic_outputs/   the judge's disagreements with the
                                    filter, and its rationales
  iterations/<NNN>/current_filter/   the rubric as it stood that iteration
  iterations/<NNN>/record.json       that iteration's metrics

Go look at how this rubric actually behaved across the whole pool before
judging it. A criterion that scores nearly every trajectory identically is
carrying no information no matter how well written it reads; one that
disagrees with the judge in a consistent direction is miscalibrated. You
can only see either by reading the scores.

Then answer with exactly the JSON object described above and nothing else.
"""


def _agentic(client: ClaudeCodeHeadlessClient, workspace: Path) -> ClaudeCodeHeadlessClient:
    from meta_filter.llm.cost_meter import replace_client

    return replace_client(client, allowed_tools=CRITIC_TOOLS, scratch_dir=Path(workspace))


@dataclass
class AgenticTrajectoryCritic(TrajectoryCritic):
    """Judge that reads the whole trajectory rather than an excerpt."""

    def _ask(self, prompt: str, trajectory: Trajectory) -> str:
        workspace = Path(tempfile.mkdtemp(prefix="judge-"))
        try:
            (workspace / "trajectory.json").write_text(
                json.dumps(_trajectory_payload(trajectory), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            agent = _agentic(self.client, workspace)
            return agent.generate(prompt + _JUDGE_TOOL_NOTE, system=JUDGE_SYSTEM).text
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


@dataclass
class AgenticRubricCritic(RubricCritic):
    """Rubric critic that reads the run's actual scoring output."""

    run_dir: Path | None = None

    def _ask(self, prompt: str) -> str:
        if self.run_dir is None:
            return super()._ask(prompt)
        agent = _agentic(self.client, Path(self.run_dir))
        return agent.generate(prompt + _RUBRIC_TOOL_NOTE, system=RUBRIC_CRITIC_SYSTEM).text


def _trajectory_payload(trajectory: Trajectory) -> dict:
    return {
        "trajectory_id": trajectory.trajectory_id,
        "task": trajectory.task,
        "metadata": dict(trajectory.metadata),
        "turns": [
            {"role": t.role, "action": t.action, "content": t.content} for t in trajectory.turns
        ],
    }
