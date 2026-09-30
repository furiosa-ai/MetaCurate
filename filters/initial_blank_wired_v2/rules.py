"""Deterministic rules for the wired blank-slate filter.

One rule, and it is here to show the shape rather than to filter anything:
it encodes no opinion about software repair, only that a transcript with no
turns cannot be evidence of anything.

Contract:
    def rule_<name>(trajectory) -> dict | None:
        return None if the rule does not fire.
        return {"name": str, "verdict": "ACCEPT|REJECT|REVIEW", "evidence": [str]}

WHAT A TRAJECTORY ACTUALLY CARRIES — read this before writing a rule, because
a rule that reads a field the row does not have returns None on every row and
nothing in the run will tell you. Silence is the failure mode: a rule that
never fires and a rule that does not exist look identical from the outside.

    trajectory.trajectory_id   str
    trajectory.task            str
    trajectory.turns           list of turns, each with .role, .action, .content
    trajectory.metadata        dict — THE KEYS VARY BY POOL

There is no `messages` and no `metadata.model_patch`. The transcript is
`turns`.

The metadata keys are deliberately not listed here. This seed is shared by
pools whose rows carry different keys, so any list written into this file
would be a cheat sheet that is wrong somewhere: a rule reading
`metadata["submission_bytes"]` because a docstring promised it has already
been written, and it fired on 0 of 220 rows. Find the keys instead — open an
actual row of THIS run's train pool and read them off it. Then check that the
key varies across rows; a key with the same value on every row orders nothing,
which is the same dead end as a key that is absent.

Verify every draft rule before proposing it with
`uv run python -m meta_filter.rule_tool --run-dir <this run> --rules
<candidate>/rules.py --sample 20`, which reports per-rule firing counts and
costs nothing, because rules involve no LLM call.
"""

from __future__ import annotations


def rule_no_turns(trajectory) -> dict | None:
    """A trajectory with zero turns cannot be evidence of anything."""

    if len(getattr(trajectory, "turns", []) or []) == 0:
        return {
            "name": "no_turns",
            "verdict": "REJECT",
            "evidence": ["trajectory has 0 turns"],
        }
    return None
