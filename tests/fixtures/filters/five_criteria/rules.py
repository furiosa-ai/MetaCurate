"""Deterministic rules for the initial filter.

The initial rule set is intentionally *minimal*: the research question is
whether optimization discovers new deterministic rules under quality/cost
pressure. Rules must be pure functions of a Trajectory-like object.

Contract:
    def rule_<name>(trajectory) -> dict | None:
        return None if the rule does not fire.
        return {"name": str, "verdict": "ACCEPT|REJECT|REVIEW", "evidence": [str]}

`trajectory` is duck-typed: it has `.turns` (an iterable of turn objects each
with `.role`, `.action`, `.content`).
"""

from __future__ import annotations


def rule_no_turns(trajectory) -> dict | None:
    """A trajectory with zero turns cannot be evidence of anything."""

    if len(trajectory.turns) == 0:
        return {
            "name": "no_turns",
            "verdict": "REJECT",
            "evidence": ["trajectory has 0 turns"],
        }
    return None
