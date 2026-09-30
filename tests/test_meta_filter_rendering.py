"""Role-aware trajectory rendering.

The thing being defended here is that the scorer and the judge see the parts
of a trajectory that carry signal. The previous renderer showed 6 turns of 200
characters, which on this pool is under 2% of the text and chosen by position
rather than content.
"""

from __future__ import annotations

from meta_filter.rendering import (
    DEFAULT_TRAJECTORY_TOOL_BUDGET,
    MIN_TOOL_OUTPUT,
    allocate_tool_budgets,
    render_submission,
    render_turns,
    truncate_tool_output,
)
from meta_filter.schemas import Trajectory, Turn


def _traj(turns: list[Turn], **metadata) -> Trajectory:
    return Trajectory(
        trajectory_id="org__repo-1",
        task="fix it",
        turns=turns,
        metadata=metadata,
    )


def test_every_turn_is_rendered_not_just_six():
    turns = [Turn(role="assistant", content=f"step {i}", action="bash") for i in range(40)]
    out = render_turns(_traj(turns))
    assert "step 0" in out
    assert "step 20" in out
    assert "step 39" in out


def test_assistant_reasoning_is_never_truncated():
    """The agent's own words are what the rubric grades, and they are cheap."""

    long_reasoning = "because " * 2000
    out = render_turns(_traj([Turn(role="assistant", content=long_reasoning, action="bash")]))
    assert long_reasoning in out


def test_tool_output_is_capped():
    noise = "PASSED test_thing\n" * 5000
    out = render_turns(_traj([Turn(role="tool", content=noise, action="tool_result")]))
    assert len(out) < len(noise)
    assert "chars elided" in out


def test_tool_budget_is_shared_across_the_whole_trajectory():
    """Cost should track the trajectory, not the number of tool turns in it."""

    noise = "PASSED test_thing\n" * 5000
    few = render_turns(_traj([Turn(role="tool", content=noise, action="t") for _ in range(5)]))
    many = render_turns(_traj([Turn(role="tool", content=noise, action="t") for _ in range(100)]))
    assert len(many) < 2 * len(few)
    assert len(many) < 2 * DEFAULT_TRAJECTORY_TOOL_BUDGET


def test_later_tool_turns_get_more_room():
    """The fix and the test run that confirms it happen at the end."""

    noise = "x" * 50_000
    budgets = allocate_tool_budgets([len(noise)] * 10, total=30_000)
    assert budgets[-1] > budgets[0]
    assert sum(budgets) <= 30_000


def test_a_short_tool_turn_hands_its_surplus_back():
    budgets = allocate_tool_budgets([10, 100_000], total=30_000)
    assert budgets[0] == 10
    assert budgets[1] == 30_000 - 10


def test_no_tool_turn_is_squeezed_below_the_floor():
    budgets = allocate_tool_budgets([100_000] * 40, total=30_000)
    assert min(budgets) >= min(MIN_TOOL_OUTPUT, 30_000 // 40)


def test_a_trajectory_under_budget_is_untouched():
    assert allocate_tool_budgets([100, 200, 300], total=30_000) == [100, 200, 300]


def test_truncation_keeps_the_tail_where_the_verdict_lives():
    text = "RUNNING pytest\n" + ("noise\n" * 5000) + "AssertionError: expected 3 got 4"
    out = truncate_tool_output(text, budget=400)
    assert out.startswith("RUNNING pytest")
    assert out.endswith("AssertionError: expected 3 got 4")
    assert len(out) < len(text)


def test_short_output_is_left_alone():
    assert truncate_tool_output("ok", DEFAULT_TRAJECTORY_TOOL_BUDGET) == "ok"


def test_system_turn_is_dropped():
    """Identical across the whole pool, so it discriminates nothing."""

    turns = [
        Turn(role="system", content="You are a helpful agent", action=None),
        Turn(role="assistant", content="looking around", action="bash"),
    ]
    out = render_turns(_traj(turns))
    assert "You are a helpful agent" not in out
    assert "looking around" in out


def test_submission_prefers_the_lossless_metadata_copy():
    patch = "diff --git a/x.py b/x.py\n+fixed"
    traj = _traj([Turn(role="exit", content="stale copy", action=None)], submission=patch)
    assert render_submission(traj) == patch


def test_submission_falls_back_to_the_exit_turn():
    traj = _traj([Turn(role="exit", content="diff --git a/x.py b/x.py", action=None)])
    assert render_submission(traj) == "diff --git a/x.py b/x.py"


def test_no_turns_renders_a_placeholder_rather_than_empty_string():
    assert render_turns(_traj([])) == "(no turns)"
