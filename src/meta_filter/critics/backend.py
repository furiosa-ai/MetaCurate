"""Critic backends supply a second opinion on a trajectory.

A backend answers "would you keep this trajectory as SFT training data?".
That answer is an *opinion*, not ground truth — this experiment has no gold
labels. The critics call a backend rather than reading anything off the
trajectory so that the opinion's provenance is always explicit and swappable
(LLM judge, ensemble, human audit).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from meta_filter.schemas import Decision, Trajectory


@runtime_checkable
class CriticBackend(Protocol):
    """A source of adjudicating verdicts used by the critics."""

    def critic_verdict(self, trajectory: Trajectory) -> Decision | None: ...


@dataclass
class StaticVerdictBackend:
    """Backend over a fixed `trajectory_id -> Decision` table.

    For tests and the synthetic demo pool, where the verdicts are stipulated
    by the fixture rather than judged. Returns `None` for unknown ids so an
    incomplete table degrades to "not adjudicated" instead of a wrong verdict.
    """

    verdicts: Mapping[str, Decision] = field(default_factory=dict)

    def critic_verdict(self, trajectory: Trajectory) -> Decision | None:
        return self.verdicts.get(trajectory.trajectory_id)
