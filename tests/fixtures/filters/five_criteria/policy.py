"""Decision policy for the five-criterion test fixture filter.

Contract:
    def decide(
        trajectory,
        rubric_scores: dict[str, float],
        triggered_rules: list[dict],
    ) -> dict:
        return {
            "decision": "ACCEPT|REJECT|REVIEW",
            "confidence": float in [0, 1],
            "reasoning_summary": str,
            "evidence": list[str],
        }

The policy is *threshold-driven* rather than agent-driven. Any deterministic
rule that fires with verdict REJECT wins immediately; REVIEW verdicts move the
decision toward REVIEW; otherwise the rubric aggregate decides.
"""

from __future__ import annotations

# Threshold constants — the proposer is allowed to rewrite these.
REVIEW_LOWER = 0.25
REVIEW_UPPER = 0.7
REJECT_MARGIN = 2  # aggregate rubric score below -REJECT_MARGIN => REJECT

RUBRIC_CRITERIA = [
    "Task Grounding",
    "Localization Efficiency",
    "Edit Justification",
    "Verification Completeness",
    "Error Recovery",
]


def _aggregate_rubric(rubric_scores: dict[str, float]) -> float:
    if not rubric_scores:
        return 0.0
    return float(sum(rubric_scores.values()))


def decide(trajectory, rubric_scores, triggered_rules):
    """Aggregate deterministic rules + rubric scores into a three-way decision."""

    evidence: list[str] = []
    rule_reject = None
    rule_review = None
    rule_accept = None
    for rule in triggered_rules or []:
        verdict = rule.get("verdict", "").upper()
        rule_ev = rule.get("evidence", []) or []
        evidence.extend(f"[rule:{rule.get('name', '?')}] {e}" for e in rule_ev)
        if verdict == "REJECT" and rule_reject is None:
            rule_reject = rule
        elif verdict == "REVIEW" and rule_review is None:
            rule_review = rule
        elif verdict == "ACCEPT" and rule_accept is None:
            rule_accept = rule

    if rule_reject is not None:
        return {
            "decision": "REJECT",
            "confidence": 0.95,
            "reasoning_summary": f"deterministic rule {rule_reject.get('name', '?')} fired",
            "evidence": evidence,
        }

    agg = _aggregate_rubric(rubric_scores)
    n = max(1, len(rubric_scores))
    normalized = agg / (2.0 * n)  # roughly in [-1, 1]
    confidence = abs(normalized)

    if agg <= -REJECT_MARGIN:
        decision = "REJECT"
    elif agg >= REJECT_MARGIN:
        decision = "ACCEPT"
    else:
        decision = "REVIEW"

    if rule_review is not None and decision == "ACCEPT":
        decision = "REVIEW"

    # Confidence-band overlay: if the rubric-driven decision is confident but
    # sits in the review band, downgrade to REVIEW for safety.
    if decision == "ACCEPT" and confidence < REVIEW_LOWER:
        decision = "REVIEW"
    if decision == "REJECT" and confidence < REVIEW_LOWER:
        decision = "REVIEW"

    if decision == "REVIEW" and confidence > REVIEW_UPPER and agg > 0:
        decision = "ACCEPT"

    summary_bits = [f"rubric_sum={agg:.1f}", f"n={n}", f"conf={confidence:.2f}"]
    if triggered_rules:
        summary_bits.append(f"rules_fired={len(triggered_rules)}")

    return {
        "decision": decision,
        "confidence": round(confidence, 3),
        "reasoning_summary": " ".join(summary_bits),
        "evidence": evidence,
    }
