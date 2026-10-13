"""BoundaryCritic — flags fragile / near-threshold decisions.

Fragility signals:
    - low confidence (below `confidence_threshold`)
    - competing rubric signals (both large positives and large negatives)
    - rubric aggregate within `score_margin` of a threshold
    - disagreement between deterministic rules and rubric direction
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from meta_filter.core.schemas import CounterExample, Decision, FilterDecision, Trajectory


@dataclass
class BoundaryCritic:
    confidence_threshold: float = 0.55
    score_margin: float = 1.0

    def find_fragile_cases(
        self,
        trajectories: Iterable[Trajectory],
        decisions: Iterable[FilterDecision],
        *,
        iteration: int,
        filter_version: str,
    ) -> list[CounterExample]:
        traj_by_id = {t.trajectory_id: t for t in trajectories}
        fragile: list[CounterExample] = []
        for d in decisions:
            traj = traj_by_id.get(d.trajectory_id)
            if traj is None:
                continue
            reasons: list[str] = []

            if d.confidence < self.confidence_threshold and d.decision != Decision.REVIEW:
                reasons.append(f"low_confidence={d.confidence:.2f}")

            if d.rubric_scores:
                agg = sum(d.rubric_scores.values())
                if abs(agg) <= self.score_margin and d.decision != Decision.REVIEW:
                    reasons.append(f"rubric_near_threshold agg={agg:.1f}")
                pos = [v for v in d.rubric_scores.values() if v > 0]
                neg = [v for v in d.rubric_scores.values() if v < 0]
                if pos and neg and max(pos) >= 2 and min(neg) <= -2:
                    reasons.append("competing_rubric_signals")

            if d.triggered_rules and d.decision == Decision.ACCEPT:
                reasons.append("rules_fired_but_accept")

            if not reasons:
                continue

            fragile.append(
                CounterExample(
                    trajectory_id=traj.trajectory_id,
                    source_iteration=iteration,
                    filter_version=filter_version,
                    critic_decision=d.decision,
                    predicted_decision=d.decision,
                    failure_type="boundary",
                    failure_subtype=";".join(reasons),
                    severity=1,
                    confidence=d.confidence,
                    evidence_turns=list(range(min(len(traj.turns), 6))),
                    description="Decision correct or unresolved but structurally fragile.",
                    why_filter_failed="; ".join(reasons),
                    raw_trajectory=traj,
                    solver_decision=d,
                )
            )
        return fragile
