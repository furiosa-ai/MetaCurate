# Decision policy for the wired blank-slate ranking filter, v2.
#
# This file is `filters/initial_blank_wired`'s policy with no change to any
# constant or any function — v2 changed only `rules.py`, whose docstring used
# to hand the proposer a hardcoded list of metadata keys. That list was read
# off one pool and is false for the other, so it was teaching candidates to
# write rules that fire on nothing; it now tells the proposer to read the keys
# off a real row and to check a draft rule's firing count before proposing it.
#
# `filters/initial_blank` with one thing added: a worked example of rules
# reaching the SCORE. The blank seed read `triggered_rules` for evidence only,
# so a candidate that wanted a deterministic signal had to invent the wiring
# as well as the rule — and every rule either blank arm wrote was dead on
# arrival for an unrelated reason (it read fields the rows do not carry), so
# the loop never saw a rule move anything and stopped writing them.
#
# Still no domain knowledge: the one rule is "a transcript with zero turns is
# not evidence", and the table below prices it. What the example demonstrates
# is the PATTERN, not the content —
#
#   * rule verdicts are read generically, so a NEW rule is priced by adding
#     one line to RULE_WEIGHTS rather than by rewriting decide();
#   * an unpriced rule falls back to its verdict's default, so a rule is
#     never silently worth nothing;
#   * ACCEPT and REJECT both move the score, so a rule can promote as well as
#     demote — it is a term in a ranking, not a gate;
#   * every term is disclosed in `evidence`, which is how the rest of this
#     repo's filters report their arithmetic.
#
# Score composition:
#   base        50 + rubric_mean * 12.5      mean in [-2, +2] -> [25, 75]
#   rules       sum of each fired rule's weight, capped at +/- RULE_CAP
#
# The cap keeps rules a correction on the semantic ordering rather than a
# replacement for it: with one criterion the rubric spans 50 points and the
# rules can move a row 15, so they reorder within a band, not across it.

from __future__ import annotations

RUBRIC_CRITERIA = [
    "Overall Training Value",
]

BASE = 50.0
SPAN = 12.5
ACCEPT_SCORE = 50.0

# Per-rule score adjustment, keyed by the name a fired rule ARRIVES WITH.
#
# That is the Python function name, `rule_no_turns` — NOT the label the rule
# passes to `_fire`. The runner substitutes it (`Solver`, "prefer the function
# name so the trigger label matches the rules.py source-of-truth"), so a table
# keyed by the declared label misses every time and the rule silently takes
# DEFAULT_WEIGHT instead. A policy keyed that way can leave a rule that fires
# on most rows worth nothing at all, if it has also dropped the fallback.
RULE_WEIGHTS: dict[str, float] = {
    "rule_no_turns": -15.0,
}
DEFAULT_WEIGHT = {"REJECT": -8.0, "ACCEPT": 4.0, "REVIEW": 0.0}
RULE_CAP = 15.0


def _values(rubric_scores) -> list[float]:
    values = []
    for value in (rubric_scores or {}).values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _rule_adjustment(triggered_rules, evidence: list[str]) -> float:
    """What the fired rules are worth, disclosed one line per rule."""

    total = 0.0
    for rule in triggered_rules or []:
        name = rule.get("name", "?")
        verdict = (rule.get("verdict") or "").upper()
        weight = RULE_WEIGHTS.get(name, DEFAULT_WEIGHT.get(verdict, 0.0))
        total += weight
        for line in rule.get("evidence") or []:
            evidence.append(f"[rule:{name}] {line}")
        evidence.append(f"[score] {weight:+.1f} from rule {name} ({verdict or 'no verdict'})")
    clamped = max(-RULE_CAP, min(RULE_CAP, total))
    if clamped != total:
        evidence.append(f"[score] rule total {total:+.1f} clamped to {clamped:+.1f}")
    return clamped


def decide(trajectory, rubric_scores, triggered_rules):
    evidence: list[str] = []

    values = _values(rubric_scores)
    mean = sum(values) / len(values) if values else 0.0
    score = BASE + mean * SPAN
    evidence.append(
        f"[score] {score:.1f} from rubric mean {mean:+.2f} over {len(values)} criteria"
    )

    score += _rule_adjustment(triggered_rules, evidence)
    score = max(0.0, min(100.0, score))

    decision = "ACCEPT" if score >= ACCEPT_SCORE else "REJECT"
    confidence = round(min(0.95, 0.5 + abs(score - ACCEPT_SCORE) / 70.0), 3)
    return {
        "decision": decision,
        "score": round(score, 2),
        "confidence": confidence,
        "reasoning_summary": (
            f"score {score:.1f} (mean={mean:+.2f}, "
            f"rules={len(triggered_rules or [])})"
        ),
        "evidence": evidence,
    }
