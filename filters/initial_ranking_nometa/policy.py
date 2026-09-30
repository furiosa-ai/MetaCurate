# Decision policy for the metadata-free ranking-mode initial filter.
#
# The non-blank seed of the metadata-free cells. It composes the four rubric
# criteria and the seven transcript rules and reads nothing else:
#
#     base       50 + rubric_mean * 25    mean in [-2, +2] -> [ 0, 100]
#     vetoes    -15 per rejecting rule, up to two
#     worst     -10 if any criterion sits at the -2 floor
#     ACCEPT at score >= 65  ->  m >= 0.60 with no veto
#
# Ranking mode reads the score, not the decision; the threshold only keeps
# the decision channel alive (a base that could never reach it would make
# every row REJECT). Four criteria give the mean 17 distinct values before
# any rule term, well clear of the loop's `min_distinct_levels` floor.

from __future__ import annotations

RUBRIC_CRITERIA = [
    "Task Grounding",
    "Edit Justification",
    "Verification Completeness",
    "Error Recovery",
]

VETO_PENALTY = 15.0
VETO_CAP = 2
WORST_FLOOR = -2.0
WORST_PENALTY = 10.0
ACCEPT_SCORE = 65.0
BASE_OFFSET = 50.0
BASE_SPAN = 25.0


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

    score = BASE_OFFSET + mean * BASE_SPAN
    evidence.append(
        f"[score] base {BASE_OFFSET + mean * BASE_SPAN:+.1f} from rubric mean "
        f"{mean:+.2f} over {len(values)} criteria"
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
