# Test fixture policy: a continuous score over six rubric criteria.
#
#   base       30 + rubric_mean * 15        rubric mean in [-2, +2] -> [0, 60]
#   vetoes     -15 per rejecting-verdict rule, up to two counted
#   worst mark -10 if any single criterion sits at the -2 floor
#   clamped to [0, 100]; the decision (score >= 65 accepts) is bookkeeping.

from __future__ import annotations

RUBRIC_CRITERIA = [
    "Task Grounding",
    "Query/PR Quality",
    "Gold-Patch Relation",
    "Edit Justification",
    "Verification Completeness",
    "Error Recovery",
]

VETO_PENALTY = 15.0
VETO_CAP = 2
WORST_FLOOR = -2.0
WORST_PENALTY = 10.0
ACCEPT_SCORE = 65.0


def _values(rubric_scores) -> list[float]:
    values = []
    for value in (rubric_scores or {}).values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def decide(trajectory, rubric_scores, triggered_rules):
    evidence: list[str] = []
    rejecting: list[str] = []
    for rule in triggered_rules or []:
        name = rule.get("name", "?")
        evidence.extend(f"[rule:{name}] {e}" for e in rule.get("evidence") or [])
        if (rule.get("verdict") or "").upper() == "REJECT":
            rejecting.append(name)

    values = _values(rubric_scores)
    mean = sum(values) / len(values) if values else 0.0
    worst = min(values) if values else 0.0

    score = 30.0 + mean * 15.0
    evidence.append(
        f"[score] base {30.0 + mean * 15.0:+.1f} from rubric mean {mean:+.2f} "
        f"over {len(values)} criteria"
    )
    if rejecting:
        counted = min(len(rejecting), VETO_CAP)
        score -= VETO_PENALTY * counted
        evidence.append(
            f"[score] -{VETO_PENALTY * counted:.0f} from {counted} rejecting "
            f"rule(s): {', '.join(rejecting[:VETO_CAP])}"
        )
    if values and worst <= WORST_FLOOR:
        score -= WORST_PENALTY
        evidence.append(
            f"[score] -{WORST_PENALTY:.0f} worst criterion at the {WORST_FLOOR:+.0f} floor"
        )
    score = max(0.0, min(100.0, score))

    decision = "ACCEPT" if score >= ACCEPT_SCORE else "REJECT"
    confidence = round(min(0.95, 0.5 + abs(score - ACCEPT_SCORE) / 70.0), 3)
    return {
        "decision": decision,
        "score": round(score, 2),
        "confidence": confidence,
        "reasoning_summary": (f"score {score:.1f} (mean={mean:+.2f}, vetoes={len(rejecting)})"),
        "evidence": evidence,
    }
