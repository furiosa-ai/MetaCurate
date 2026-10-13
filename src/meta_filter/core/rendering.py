"""Role-aware trajectory rendering for LLM-facing prompts.

A fixed "show 6 turns, 200 characters each" in each of the scorer and the
judge would be a thumbnail on a pool averaging ~30K characters per
trajectory: the model would be asked whether a trajectory is good SFT data
while seeing under 2% of it, chosen by position rather than by information
content.

This module renders the whole trajectory instead, and spends its budget where
the signal is:

* **assistant** turns — the agent's reasoning and the commands it ran — are
  never cut. They are what the rubric actually grades, and they are short.
* **tool** turns share one budget for the whole trajectory, because a single
  `pytest -q` can dump 200K characters of passing-test noise and trajectories
  differ by 20x in how many such turns they contain.
* **exit** turns carry the submitted patch, capped separately and more
  generously.

Ingest stays lossless; truncation lives here, at the read boundary, so the
pseudo gold judge can be handed the full text from disk while the cheap
scorer gets the budgeted view.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from meta_filter.core.schemas import Trajectory, Turn

# Bumped when the rendering *logic* changes in a way the constants below do
# not capture — dropping a role, reordering turns, changing where a cut lands.
# See `view_signature`.
RENDERER_VERSION = 2

# Tool output is budgeted per *trajectory*, not per turn. On this pool a
# trajectory carries ~55 tool turns, so a per-turn cap prices a 5-turn
# trajectory the same as a 100-turn one: the short trajectory is starved of
# detail it could afford, and the long one still costs 20x more.
DEFAULT_TRAJECTORY_TOOL_BUDGET = 30_000
DEFAULT_SUBMISSION_BUDGET = 4000
# No tool turn is squeezed below this, so an early `ls` still shows what it
# found rather than an ellipsis.
MIN_TOOL_OUTPUT = 300
# The last tool turn gets this many times the share of the first. The fix and
# the test run that confirms it happen at the end of a trajectory; the start
# is orientation.
TAIL_WEIGHT = 4.0
# The system prompt is identical across every trajectory in a pool, so it
# carries no discriminating signal and is dropped from the rendered view.
_SKIPPED_ROLES = frozenset({"system"})


def truncate_tool_output(text: str, budget: int) -> str:
    """Cut `text` to `budget` characters, keeping the most diagnostic parts.

    Tool output in these trajectories is dominated by test runs and
    tracebacks. Both put the *conclusion* at the end — the assertion that
    failed, the final return code — while the *identity* of what ran sits at
    the top. The middle is usually a list of passing tests.
    """

    if len(text) <= budget:
        return text
    # Keep a small head so the reader can tell what command produced this,
    # then the tail, which is where the verdict lives.
    head = budget // 4
    tail = budget - head
    dropped = len(text) - budget
    return f"{text[:head]}\n... [{dropped} chars elided] ...\n{text[-tail:]}"


def view_signature(tool_output_budget: int) -> str:
    """Fingerprint of everything that decides what a reader actually sees.

    A cached LLM verdict is only reusable if the view behind it is the view we
    would render today. Folding this into a cache key turns a renderer change
    into a cache miss instead of a silent mix of two generations of verdicts.
    """

    payload = json.dumps(
        {
            "renderer": RENDERER_VERSION,
            "tool_budget": tool_output_budget,
            "submission_budget": DEFAULT_SUBMISSION_BUDGET,
            "min_tool_output": MIN_TOOL_OUTPUT,
            "tail_weight": TAIL_WEIGHT,
            "skipped_roles": sorted(_SKIPPED_ROLES),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _tail_weights(n: int) -> list[float]:
    if n == 1:
        return [1.0]
    return [1.0 + (TAIL_WEIGHT - 1.0) * (i / (n - 1)) for i in range(n)]


def allocate_tool_budgets(
    lengths: Sequence[int],
    total: int = DEFAULT_TRAJECTORY_TOOL_BUDGET,
) -> list[int]:
    """Split `total` characters across a trajectory's tool turns.

    Two things shape the split. Later turns weigh more, because that is where
    the fix and the test run confirming it live. And a turn shorter than its
    share hands the surplus back to the turns still asking for more, so a
    trajectory full of one-line `ls` output spends its budget on the pytest
    run instead of leaving it unclaimed.
    """

    n = len(lengths)
    if n == 0:
        return []
    if sum(lengths) <= total:
        return list(lengths)

    weights = _tail_weights(n)
    budgets = [0] * n
    pending = list(range(n))
    remaining = total
    while pending:
        weight_sum = sum(weights[i] for i in pending)
        floor = min(MIN_TOOL_OUTPUT, remaining // len(pending))
        share = {i: max(floor, int(remaining * weights[i] / weight_sum)) for i in pending}
        fits = {i for i in pending if lengths[i] <= share[i]}
        if not fits:
            for i in pending:
                budgets[i] = share[i]
            break
        for i in fits:
            budgets[i] = lengths[i]
            remaining -= lengths[i]
        pending = [i for i in pending if i not in fits]
    return budgets


def _render_turn(index: int, turn: Turn, budget: int | None) -> str:
    role = (turn.role or "").strip()
    content = turn.content or ""
    action = turn.action or "-"
    if budget is not None:
        content = truncate_tool_output(content, budget)
    return f"[turn {index:03d}] role={role} action={action}\n{content}"


def render_turns(
    trajectory: Trajectory,
    *,
    tool_output_budget: int = DEFAULT_TRAJECTORY_TOOL_BUDGET,
) -> str:
    """Every turn in order, with the trajectory's tool budget spread over it."""

    kept = [
        (i, turn)
        for i, turn in enumerate(trajectory.turns)
        if (turn.role or "").strip() not in _SKIPPED_ROLES
    ]
    tool_positions = [pos for pos, (_, turn) in enumerate(kept) if turn.role == "tool"]
    allocated = allocate_tool_budgets(
        [len(kept[pos][1].content or "") for pos in tool_positions],
        tool_output_budget,
    )
    budgets: dict[int, int] = dict(zip(tool_positions, allocated))

    rendered = []
    for pos, (index, turn) in enumerate(kept):
        budget = budgets.get(pos)
        if budget is None and turn.role == "exit":
            budget = DEFAULT_SUBMISSION_BUDGET
        rendered.append(_render_turn(index, turn, budget))
    return "\n\n".join(rendered) if rendered else "(no turns)"


def render_submission(
    trajectory: Trajectory,
    budget: int = DEFAULT_SUBMISSION_BUDGET,
) -> str:
    """The submitted patch, preferring the lossless copy kept at ingest."""

    submission = trajectory.metadata.get("submission")
    if isinstance(submission, str) and submission.strip():
        return truncate_tool_output(submission, budget)
    for turn in reversed(trajectory.turns):
        if turn.role == "exit" and turn.content:
            return truncate_tool_output(turn.content, budget)
    return "(no submission)"
