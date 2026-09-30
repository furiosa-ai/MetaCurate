"""An LLM's read of whether a rule set is *designed* well, not just precise.

The deterministic `rule_quality` is a mean of `precision x (1 - redundancy)`
over the rules that fired. A mean has no term for coverage, so its optimum is
to delete every imperfect rule, and an optimizer finds that:

    before  rule_quality 0.6358   5 rules firing, precisions 1.00 / 0.95 /
                                  0.81 / 0.67 / 0.00
    after   rule_quality 1.0000   1 rule firing

It reaches a perfect score by deleting a 95%-precision rule and an
81%-precision rule, because both drag the mean down. Four of the five
surviving rules never fire at all; non-firing rules are excluded from the
mean, so they are invisible to it. The metric then saturates at 1.0 and has
no gradient left to give.

What a mean of precisions cannot express is what the rules *miss*. That is
what this pass adds: the model is shown the measured stats — never asked to
guess them — plus the trajectories no rule fired on, and it scores the design
across six dimensions. The result multiplies the deterministic score rather
than replacing it, so a rule set that is precise but blind is penalised for
being blind while an imprecise one still cannot score well.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from meta_filter.critics._shared import CRITIC_PREAMBLE

logger = logging.getLogger(__name__)

# Every dimension is scored on the same -2..+2 anchors the rubric uses. A free
# 0..1 score clusters around 0.7 and stops discriminating.
DIMENSIONS = (
    "coverage",
    "precision_discipline",
    "factoring",
    "division_of_labor",
    "robustness",
    "policy_coherence",
)

# How many evidence rows to show per block. Enough for a recurring pattern to
# be visible, small enough to leave room for the sources.
EVIDENCE_LIMIT = 12

RULE_SCORE_SYSTEM = (
    CRITIC_PREAMBLE + " Your specific role: score the DESIGN of a deterministic rule set. "
    "Precision, recall and redundancy have already been measured for you and "
    "are given below — do not recompute or guess them, interpret them. Your "
    "value is in what those numbers cannot express: which recurring failure "
    "modes no rule covers, which rules are dead weight, and which judgments "
    "sit in the wrong file. Return exactly one JSON object; no prose outside "
    "it."
)

RULE_SCORE_PROMPT = """\
Score the design of this deterministic rule set.

=== rubric.md (the semantic half of the same filter) ===
{rubric_md}

=== rules.py (source) ===
{rules_source}

=== policy.py (source) ===
{policy_source}
# The aggregation layer. It reads the criteria the rubric defines and the
# verdicts the rules emit, and turns them into one decision. Nothing else
# reviews this file.

=== measured per-rule stats (authoritative — do not contradict) ===
{stats_block}
# fires / truth_agreed / precision / recall / redundancy over N={n_trajectories}
# trajectories. A rule with fires=0 never triggered on this pool.

=== uncovered decisions ({n_uncovered} of {n_labelled} labelled) ===
# Trajectories the critic reached a verdict on where NO rule fired. These are
# what the rule set is blind to and leaves entirely to the rubric.
{uncovered_block}

=== misfires ===
# Trajectories where a rule fired and disagreed with the critic. These are
# what the rule set actively gets wrong.
{misfire_block}

Score each dimension on this scale. Cite concrete evidence — a rule name, a
trajectory id, or a stat — for every non-zero score.

  coverage           Do the rules catch the structural failure modes that
                     recur in the uncovered set?
    +2 every recurring structural mode is covered; what remains uncovered is
       genuinely semantic and belongs to the rubric.
    +1 the dominant mode is covered; a minor one is not.
     0 partial coverage, or the uncovered set has no recurring mode.
    -1 a recurring, cheaply-detectable structural mode has no rule.
    -2 two or more such modes are uncovered, or the rules fire on under 10%
       of trajectories while the uncovered set is large.

  precision_discipline  Are the measured precisions defensible?
    +2 every firing rule is above 0.9, and any lower one emits REVIEW rather
       than a hard verdict.
    +1 one firing rule is 0.7-0.9 and its misfires share a nameable cause.
     0 mixed, no clear pattern.
    -1 a firing rule is below 0.7 and emits a hard ACCEPT/REJECT.
    -2 a firing rule is below 0.5, OR a rule's precision is high only because
       it reads the same metadata field the critic reads. Name the field.

  factoring          Is the set well-factored?
    +2 every rule fires, no two overlap materially, each tests one thing.
    +1 one dead or near-duplicate rule.
     0 some dead rules but no harmful overlap.
    -1 several rules never fire, or two rules have redundancy above 0.5.
    -2 the majority of rules never fire — the set is decorative.

  division_of_labor  Is each judgment in the right file?
    +2 rules are cheap structural predicates, the rubric holds the semantic
       judgment, neither duplicates the other.
    +1 one borderline case.
     0 unclear but not harmful.
    -1 a rule encodes a judgment needing reading comprehension, OR the rubric
       spends a criterion on what a five-line rule decides.
    -2 both, or a rule and a criterion contradict each other.

  policy_coherence   Does policy.py aggregate what the other two produce?
    +2 every criterion and rule the policy reads exists and is scored, and
       each threshold has a stated basis in a comment.
    +1 one threshold is unexplained but defensible.
     0 no evidence either way.
    -1 the policy reads a criterion the rubric no longer defines, ignores a
       criterion the rubric spends a section on, or a magic number decides a
       verdict with no stated basis.
    -2 the policy can return a verdict that the rubric and rules jointly
       contradict. Name the trajectory shape where that happens.

  robustness         Will these rules behave on unseen trajectories?
    +2 field access is defensive, missing metadata yields no fire rather than
       a wrong fire, thresholds are justified in a comment.
    +1 minor unguarded access on a field that is always present.
     0 no evidence either way.
    -1 a rule silently never fires because of a field-name mismatch, or a
       magic threshold appears with no stated basis.
    -2 a rule can raise, or it inverts its verdict when metadata is absent.

Then propose concrete changes. Each must name the file it applies to
(rules.py or policy.py), the rule or threshold it touches, what to change,
and which trajectories would flip.

DO NOT reward size. A three-rule set that covers the recurring modes scores
higher than a twelve-rule set where nine are dead. Deleting a rule is an
improvement only if nothing in the uncovered set needed it.

Return exactly this JSON:
{{
  "scores": {{"coverage": 0, "precision_discipline": 0, "factoring": 0,
              "division_of_labor": 0, "robustness": 0, "policy_coherence": 0}},
  "evidence": {{"<dimension>": "<one sentence citing a rule, trajectory or stat>"}},
  "proposals": [
    {{"action": "add" | "tighten" | "split" | "merge" | "drop" | "move_to_rubric",
      "file": "rules.py" | "policy.py",
      "target": "<rule name, threshold, or a name for a new rule>",
      "change": "<one or two sentences>",
      "expected_effect": "<which trajectories flip, and why>"}}
  ],
  "circularity_warnings": ["<a rule precision presented as more independent
      evidence than it is — misreporting what a precision measures is what
      to flag>"]
}}
"""


@dataclass
class DesignReview:
    """The model's scored read of a rule set's design."""

    scores: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    proposals: list[dict[str, str]] = field(default_factory=list)
    circularity_warnings: list[str] = field(default_factory=list)

    @property
    def quality(self) -> float:
        """Map the five -2..+2 dimensions onto [0, 1].

        An all-zero review lands at 0.5 — "nothing to praise, nothing damning"
        — which is the same neutral the deterministic side uses when it cannot
        speak.
        """

        if not self.scores:
            return 0.5
        total = sum(int(self.scores.get(d, 0)) for d in DIMENSIONS)
        return (total + 2 * len(DIMENSIONS)) / (4 * len(DIMENSIONS))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "evidence": dict(self.evidence),
            "proposals": list(self.proposals),
            "circularity_warnings": list(self.circularity_warnings),
            "design_quality": self.quality,
        }


# The multiplier is `DESIGN_FLOOR + (1 - DESIGN_FLOOR) * design`, so the floor
# decides how much authority the review has. It has to be low enough to reverse
# the delete-until-perfect outcome above, and that fixes it within a narrow
# band.
#
# "after" keeps one perfect rule and covers nothing: det 1.0, and a review that
# scores coverage and factoring at -2 lands at design 0.30. "before" keeps five
# rules that cover the pool: det 0.6358, design about 0.75. Requiring the
# second to win:
#
#     1.0 * (f + (1-f)*0.30)  <  0.6358 * (f + (1-f)*0.75)
#     0.70f + 0.30            <  0.15895f + 0.47685
#                          f  <  0.327
#
# A floor of 0.5 leaves the delete-until-perfect strategy winning 0.650 to
# 0.556 — the loophole stays open. 0.25 closes it with margin while still
# refusing to call a working filter worthless because a critic disliked it.
DESIGN_FLOOR = 0.25


def combine(deterministic: float, design: float | None) -> float:
    """Fold the design score into the measured one.

    Multiplying rather than averaging is the point: averaging lets a perfect
    deterministic score carry a damning review, which is exactly the strategy
    that has to stop paying.

    Note that a *neutral* review is not free — it still multiplies by 0.625.
    Making 0.5 the identity point would mean `m(d) = 0.5 + d`, and that puts
    the blind-but-perfect set back on top (0.800 against 0.795), so the
    shrink is what closes the loophole rather than a side effect of it. The
    consequence is that `rule_quality` is on a different scale once this pass
    is on, which is why runs that disagree on the flag are not comparable.
    """

    if design is None:
        return deterministic
    return deterministic * (DESIGN_FLOOR + (1.0 - DESIGN_FLOOR) * design)


def build_evidence(
    *,
    trajectory_ids: list[str],
    fired_ids: set[str],
    truth_map: dict[str, Any],
    misfires: list[tuple[str, str, Any, Any]],
    limit: int = EVIDENCE_LIMIT,
) -> tuple[str, str, int, int]:
    """Render the two evidence blocks the deterministic stats cannot carry.

    Returns (uncovered_block, misfire_block, n_uncovered, n_labelled).
    """

    labelled = [tid for tid in trajectory_ids if truth_map.get(tid) is not None]
    uncovered = [tid for tid in labelled if tid not in fired_ids]

    if uncovered:
        uncovered_block = "\n".join(
            f"{tid}  critic={_name(truth_map.get(tid))}" for tid in uncovered[:limit]
        )
        if len(uncovered) > limit:
            uncovered_block += f"\n... and {len(uncovered) - limit} more"
    else:
        uncovered_block = "(none — every labelled trajectory triggered a rule)"

    if misfires:
        misfire_block = "\n".join(
            f"{tid}  rule={rule}  rule_said={_name(said)}  critic={_name(truth)}"
            for tid, rule, said, truth in misfires[:limit]
        )
        if len(misfires) > limit:
            misfire_block += f"\n... and {len(misfires) - limit} more"
    else:
        misfire_block = "(none — no firing rule disagreed with the critic)"

    return uncovered_block, misfire_block, len(uncovered), len(labelled)


def _name(decision: Any) -> str:
    return getattr(decision, "value", None) or getattr(decision, "name", None) or str(decision)


def parse(text: str) -> DesignReview:
    """Read the model's JSON, clamping anything out of range.

    A single malformed field must not cost the iteration its rule score, so
    every piece degrades to a neutral rather than raising.
    """

    payload = _extract_json(text)
    if payload is None:
        logger.warning("rule design review was not JSON; treating as neutral")
        return DesignReview()

    raw_scores = payload.get("scores") or {}
    scores: dict[str, int] = {}
    for dim in DIMENSIONS:
        try:
            scores[dim] = max(-2, min(2, int(raw_scores.get(dim, 0))))
        except (TypeError, ValueError):
            scores[dim] = 0

    evidence = {str(k): str(v) for k, v in (payload.get("evidence") or {}).items() if v}
    proposals = [p for p in (payload.get("proposals") or []) if isinstance(p, dict)]
    warnings = [str(w) for w in (payload.get("circularity_warnings") or []) if w]

    return DesignReview(
        scores=scores,
        evidence=evidence,
        proposals=proposals,
        circularity_warnings=warnings,
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
