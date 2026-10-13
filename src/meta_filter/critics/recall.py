"""RecallCritic — trajectories the filter kept but the critic would drop.

The filter said this trajectory is worth training on; the critic, looking at
it again, says it is not. Often called a "false negative", which is a
misnomer here — nothing establishes which side is right. It is a disagreement
in one direction, and `PrecisionCritic` covers the mirror direction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from meta_filter.core.schemas import CounterExample, Decision, FilterDecision, Trajectory
from meta_filter.critics.backend import CriticBackend


@dataclass
class RecallCritic:
    backend: CriticBackend

    def find_false_negatives(
        self,
        trajectories: Iterable[Trajectory],
        decisions: Iterable[FilterDecision],
        *,
        iteration: int,
        filter_version: str,
    ) -> list[CounterExample]:
        traj_by_id = {t.trajectory_id: t for t in trajectories}
        false_negatives: list[CounterExample] = []
        for d in decisions:
            traj = traj_by_id.get(d.trajectory_id)
            if traj is None:
                continue
            if self.backend.critic_verdict(traj) != Decision.REJECT:
                continue
            # REVIEW is not a disagreement — it defers rather than accepts.
            if d.decision != Decision.ACCEPT:
                continue
            false_negatives.append(
                CounterExample(
                    trajectory_id=traj.trajectory_id,
                    source_iteration=iteration,
                    filter_version=filter_version,
                    critic_decision=Decision.REJECT,
                    predicted_decision=d.decision,
                    failure_type="false_negative",
                    failure_subtype=_infer_bad_subtype(traj),
                    severity=2,
                    confidence=d.confidence,
                    evidence_turns=_evidence_indices(traj),
                    description="Filter accepted a trajectory that should be rejected.",
                    why_filter_failed=_why_missed(d),
                    raw_trajectory=traj,
                    solver_decision=d,
                )
            )
        return false_negatives


def _infer_bad_subtype(traj: Trajectory) -> str:
    """Heuristic label for the bad pattern; used for reporting only."""

    hint = traj.metadata.get("bad_subtype")
    if isinstance(hint, str) and hint:
        return hint
    actions = [(getattr(t, "action", None) or t.role).lower() for t in traj.turns]
    if not actions:
        return "empty_trajectory"
    repeat_max = max((actions.count(a) for a in set(actions)), default=0)
    if repeat_max >= 5:
        return "action_loop"
    if actions.count("edit") + actions.count("test") >= 8:
        return "edit_test_thrash"
    return "unspecified_bad"


def _evidence_indices(traj: Trajectory) -> list[int]:
    return list(range(min(len(traj.turns), 8)))


def _why_missed(d: FilterDecision) -> str:
    if not d.triggered_rules and not d.rubric_scores:
        return "no rule fired and rubric was empty"
    if not d.triggered_rules:
        low = [k for k, v in d.rubric_scores.items() if v <= -1]
        return (
            "no deterministic rule fired; rubric did not push far enough negative "
            f"(sub-anchors: {low})"
        )
    return "deterministic rules fired but policy still accepted"
