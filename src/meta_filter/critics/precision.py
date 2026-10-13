"""PrecisionCritic — trajectories the filter dropped but the critic would keep.

The mirror of `RecallCritic`. Often called a "false positive", which
overstates what we know: this is a disagreement, not a proven mistake.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from meta_filter.core.schemas import CounterExample, Decision, FilterDecision, Trajectory
from meta_filter.critics.backend import CriticBackend


@dataclass
class PrecisionCritic:
    backend: CriticBackend

    def find_false_positives(
        self,
        trajectories: Iterable[Trajectory],
        decisions: Iterable[FilterDecision],
        *,
        iteration: int,
        filter_version: str,
    ) -> list[CounterExample]:
        traj_by_id = {t.trajectory_id: t for t in trajectories}
        wrong_rejections: list[CounterExample] = []
        for d in decisions:
            traj = traj_by_id.get(d.trajectory_id)
            if traj is None:
                continue
            if self.backend.critic_verdict(traj) != Decision.ACCEPT:
                continue
            if d.decision != Decision.REJECT:
                continue
            wrong_rejections.append(
                CounterExample(
                    trajectory_id=traj.trajectory_id,
                    source_iteration=iteration,
                    filter_version=filter_version,
                    critic_decision=Decision.ACCEPT,
                    predicted_decision=d.decision,
                    failure_type="false_positive",
                    failure_subtype=_infer_good_subtype(traj),
                    severity=2,
                    confidence=d.confidence,
                    evidence_turns=list(range(min(len(traj.turns), 8))),
                    description="Filter rejected a trajectory that should be accepted.",
                    why_filter_failed=_why_rejected(d),
                    raw_trajectory=traj,
                    solver_decision=d,
                )
            )
        return wrong_rejections


def _infer_good_subtype(traj: Trajectory) -> str:
    hint = traj.metadata.get("good_subtype")
    if isinstance(hint, str) and hint:
        return hint
    if len(traj.turns) < 3:
        return "brief_but_valid"
    return "long_but_valid"


def _why_rejected(d: FilterDecision) -> str:
    if d.triggered_rules:
        return f"rules fired incorrectly: {d.triggered_rules}"
    negatives = [k for k, v in d.rubric_scores.items() if v <= -1]
    return f"rubric scored too negatively on {negatives}"
