"""NEGATIVE FIXTURE: a rule that reads an outcome field.

Used only by the tests that check such a filter is refused — as an initial
filter at boot, and as a candidate at precheck. Never a filter to run.
"""

from __future__ import annotations


def rule_reads_outcome(trajectory) -> dict | None:
    """Fires on rows whose harness outcome is recorded as passing."""

    if (getattr(trajectory, "metadata", None) or {}).get("resolved"):
        return {"name": "reads_outcome", "verdict": "ACCEPT", "evidence": ["resolved row"]}
    return None
