"""RuleCritic — precision / recall / redundancy audit of `rules.py`.

Deterministic metrics come first (no LLM required):
  precision(rule)  = truth-agreed fires / total fires
  recall(rule)     = truth-agreed fires / (truth-agreed fires + missed fires)
                     where "missed fires" = trajectories where truth agrees
                     with the rule's typical verdict but the rule did not fire
  redundancy(rule) = max Jaccard overlap between this rule's fire set and
                     every other rule's fire set

An overall `rule_quality ∈ [0, 1]` aggregates precision × (1 - redundancy).

An optional LLM pass ranks each rule as keep / move_to_rubric / split / drop
and lists signals currently hiding in the rubric that would be cheaper as
deterministic rules. The LLM pass is opt-in: pass `client=...` to enable.

Redundancy needs a second rule to be defined at all, so with
fewer than two rules `rule_quality` stays neutral at 0.5 rather than being
compared against multi-rule filters. Per-rule stats and the LLM pass still
run — they are what explains the number.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meta_filter.core.schemas import Decision, Trajectory
from meta_filter.critics import rule_design
from meta_filter.critics._shared import CRITIC_PREAMBLE, load_cache
from meta_filter.critics.rule_design import (
    RULE_SCORE_PROMPT,
    RULE_SCORE_SYSTEM,
    DesignReview,
)
from meta_filter.llm.base import LLMClient

logger = logging.getLogger(__name__)


RULE_CRITIC_SYSTEM = (
    CRITIC_PREAMBLE + " Your specific role: given deterministic precision, recall and "
    "redundancy numbers for each rule in rules.py, together with the rubric "
    "prompt, decide for each rule whether it is earning its keep, whether "
    "its work would be cleaner as a rubric criterion instead, or whether "
    "it should be dropped. Return exactly one JSON object; no prose outside "
    "it."
)

RULE_CRITIC_PROMPT = """\
Review the deterministic rules below and decide what to do with each.

=== rubric.md ===
{rubric_md}

=== rules.py (source) ===
{rules_source}

=== per-rule stats ===
{stats_block}

For each rule return a recommendation:
  - "keep": the rule earns its keep as-is.
  - "move_to_rubric": the rule's judgment would be cleaner as a rubric
    criterion (natural-language signal, not a hard threshold).
  - "split": the rule fires on several loosely-related conditions and
    should be broken into more targeted rules.
  - "drop": the rule is redundant with another rule or with the rubric, or
    its precision is too low to be worth keeping.

Also list any signals that the rubric currently *tries* to capture
semantically but that would be cheaper (and more consistent) as a
deterministic rule.

Return exactly this JSON:
{{
  "rules": [
    {{"name": "<rule_name>",
      "recommendation": "keep" | "move_to_rubric" | "split" | "drop",
      "note": "<one sentence>"}}
  ],
  "signals_missing_from_rules": ["<short description>"]
}}"""


@dataclass
class RuleStats:
    """Deterministic stats for a single rule."""

    name: str
    fires: int
    truth_agreed: int
    precision: float
    recall: float
    redundancy: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fires": self.fires,
            "truth_agreed": self.truth_agreed,
            "precision": self.precision,
            "recall": self.recall,
            "redundancy": self.redundancy,
        }


@dataclass
class RuleCriticReport:
    """RuleCritic's output for one iteration."""

    rule_quality: float
    per_rule: list[RuleStats] = field(default_factory=list)
    recommendations: list[dict[str, str]] = field(default_factory=list)
    signals_missing_from_rules: list[str] = field(default_factory=list)
    llm_used: bool = False
    from_cache: bool = False
    # The deterministic score before the design review folded into it, kept so
    # a run can be compared against one where the review was off.
    deterministic_quality: float | None = None
    design: DesignReview | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "rule_quality": self.rule_quality,
            "per_rule": [r.as_dict() for r in self.per_rule],
            "recommendations": list(self.recommendations),
            "signals_missing_from_rules": list(self.signals_missing_from_rules),
            "llm_used": self.llm_used,
        }
        if self.design is not None:
            payload["deterministic_quality"] = self.deterministic_quality
            payload["design"] = self.design.to_dict()
        return payload


@dataclass
class RuleCritic:
    """Deterministic-first rule auditor, LLM recommendations optional."""

    client: LLMClient | None = None
    cache_dir: Path = Path("data/rule_critic_cache")
    model_tag: str = ""
    # When on, an LLM scores the rule set's *design* and that score folds into
    # `rule_quality`. Off by default because it changes the objective: never
    # compare scores across runs that disagree on this flag.
    score_design: bool = False
    # The keep/move/split/drop pass. Separate from `score_design` so a run can
    # buy the score without paying for a second call it will not read.
    recommend: bool = True

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.model_tag and self.client is not None:
            self.model_tag = getattr(self.client, "model", "unknown")

    def review(
        self,
        *,
        rules_source: str,
        rule_functions: Iterable[Callable[[Any], dict | None]],
        trajectories: list[Trajectory],
        truth_map: dict[str, Decision | None],
        rubric_md: str = "",
        policy_source: str = "",
    ) -> RuleCriticReport:
        rule_functions = list(rule_functions)
        if not rule_functions:
            return RuleCriticReport(rule_quality=0.5)

        fire_map, verdict_map = _run_rules(rule_functions, trajectories)
        per_rule = _compute_stats(
            rule_functions=rule_functions,
            fire_map=fire_map,
            verdict_map=verdict_map,
            truth_map=truth_map,
            trajectory_ids=[t.trajectory_id for t in trajectories],
        )
        # Redundancy is a between-rule quantity, so with one rule the
        # aggregate is not comparable to a multi-rule filter's and stays
        # neutral. The per-rule stats and the LLM's read of them are still
        # worth having — suppressing those too would leave every single-rule
        # iteration reporting a bare 0.5 with nothing to explain it.
        # Coverage is a property of the rule set as a whole, so it is counted
        # over the union of fire sets rather than per rule.
        labelled_ids = {
            t.trajectory_id for t in trajectories if truth_map.get(t.trajectory_id) is not None
        }
        covered_ids = {tid for ids in fire_map.values() for tid in ids} & labelled_ids

        report = RuleCriticReport(
            rule_quality=(
                0.5
                if len(rule_functions) < 2
                else _aggregate_quality(
                    per_rule, covered=len(covered_ids), labelled=len(labelled_ids)
                )
            ),
            per_rule=per_rule,
        )

        if self.client is not None:
            if self.recommend:
                self._augment_with_llm(report, rules_source, rubric_md)
            if self.score_design:
                self._score_design(
                    report,
                    rules_source=rules_source,
                    rubric_md=rubric_md,
                    trajectories=trajectories,
                    fire_map=fire_map,
                    verdict_map=verdict_map,
                    truth_map=truth_map,
                    policy_source=policy_source,
                )
        return report

    def _score_design(
        self,
        report: RuleCriticReport,
        *,
        rules_source: str,
        rubric_md: str,
        policy_source: str,
        trajectories: list[Trajectory],
        fire_map: dict[str, set[str]],
        verdict_map: dict[str, dict[str, Decision]],
        truth_map: dict[str, Decision | None],
    ) -> None:
        trajectory_ids = [t.trajectory_id for t in trajectories]
        fired_ids = {tid for ids in fire_map.values() for tid in ids}
        misfires = [
            (tid, rule, verdict_map[rule][tid], truth_map[tid])
            for rule, ids in fire_map.items()
            for tid in sorted(ids)
            if truth_map.get(tid) is not None
            and verdict_map[rule].get(tid) is not None
            and verdict_map[rule][tid] != truth_map[tid]
        ]
        uncovered_block, misfire_block, n_uncovered, n_labelled = rule_design.build_evidence(
            trajectory_ids=trajectory_ids,
            fired_ids=fired_ids,
            truth_map=truth_map,
            misfires=misfires,
        )

        stats_block = "\n".join(json.dumps(r.as_dict()) for r in report.per_rule)
        # The evidence is part of the question, so it belongs in the cache key:
        # the same rules over a different pool is a different review.
        cache_key = _hash_inputs(
            rules_source,
            rubric_md,
            report.per_rule,
            # The trailing tag is part of every design-review key this code
            # writes; it stays so existing cache entries remain addressable.
            f"{self.model_tag}|design|{policy_source}|{uncovered_block}|{misfire_block}"
            + "|metadata_free",
        )
        cache_path = self.cache_dir / f"design_{cache_key}.json"
        cached = load_cache(cache_path)
        if cached is not None:
            review = rule_design.parse(cached.get("raw", ""))
        else:
            prompt = RULE_SCORE_PROMPT.format(
                rubric_md=(rubric_md or "(rubric not provided)").strip(),
                rules_source=rules_source.strip(),
                policy_source=(policy_source or "(policy not provided)").strip(),
                stats_block=stats_block,
                n_trajectories=len(trajectory_ids),
                n_uncovered=n_uncovered,
                n_labelled=n_labelled,
                uncovered_block=uncovered_block,
                misfire_block=misfire_block,
            )
            try:
                raw = self.client.generate(prompt, system=RULE_SCORE_SYSTEM, temperature=0.0).text
            except Exception:  # noqa: BLE001 — a critic outage must not end the run
                logger.warning("rule design review failed; leaving rule_quality as measured")
                return
            review = rule_design.parse(raw)
            cache_path.write_text(
                json.dumps(
                    {"model": self.model_tag, "raw": raw, **review.to_dict()},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        report.deterministic_quality = report.rule_quality
        report.design = review
        # The viewer reads this to decide whether to show a model-written
        # review; without it a run that paid for one displays as if it had
        # only the deterministic stats.
        report.llm_used = True
        report.rule_quality = rule_design.combine(report.rule_quality, review.quality)
        logger.info(
            "rule design review: %s -> design_quality=%.3f, rule_quality %.3f -> %.3f",
            review.scores,
            review.quality,
            report.deterministic_quality,
            report.rule_quality,
        )

    # ------------------------------------------------------------ LLM branch

    def _augment_with_llm(
        self,
        report: RuleCriticReport,
        rules_source: str,
        rubric_md: str,
    ) -> None:
        cache_key = _hash_inputs(
            rules_source,
            rubric_md,
            report.per_rule,
            self.model_tag,
        )
        cache_path = self.cache_dir / f"{cache_key}.json"
        cached = load_cache(cache_path)
        if cached is not None:
            report.recommendations = cached.get("recommendations", [])
            report.signals_missing_from_rules = cached.get("signals_missing_from_rules", [])
            report.llm_used = True
            report.from_cache = True
            return

        stats_block = "\n".join(json.dumps(r.as_dict()) for r in report.per_rule)
        prompt = RULE_CRITIC_PROMPT.format(
            rubric_md=(rubric_md or "(rubric not provided)").strip(),
            rules_source=rules_source.strip(),
            stats_block=stats_block,
        )
        response = self.client.generate(
            prompt,
            system=RULE_CRITIC_SYSTEM,
            temperature=0.0,
        )
        raw = response.text
        parsed = _parse_llm_report(raw)
        cache_path.write_text(
            json.dumps(
                {
                    **parsed,
                    "model": self.model_tag,
                    "raw": raw,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        report.recommendations = parsed["recommendations"]
        report.signals_missing_from_rules = parsed["signals_missing_from_rules"]
        report.llm_used = True


# --------------------------------------------------------- deterministic core


def _run_rules(
    rule_functions: list[Callable],
    trajectories: list[Trajectory],
) -> tuple[dict[str, set[str]], dict[str, dict[str, Decision]]]:
    """Return (fire_map, verdict_map).

    fire_map[rule_name]         = set of trajectory ids where this rule fired
    verdict_map[rule_name][tid] = verdict that rule emitted for that trajectory
    """

    fire_map: dict[str, set[str]] = {r.__name__: set() for r in rule_functions}
    verdict_map: dict[str, dict[str, Decision]] = {r.__name__: {} for r in rule_functions}
    for traj in trajectories:
        for rule in rule_functions:
            try:
                fired = rule(traj)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("rule %s raised %s", rule.__name__, exc)
                continue
            if not fired:
                continue
            verdict_str = str(fired.get("verdict", "REVIEW")).upper()
            try:
                verdict = Decision(verdict_str)
            except ValueError:
                verdict = Decision.REVIEW
            fire_map[rule.__name__].add(traj.trajectory_id)
            verdict_map[rule.__name__][traj.trajectory_id] = verdict
    return fire_map, verdict_map


def _typical_verdict(verdicts: dict[str, Decision]) -> Decision | None:
    """The verdict a rule 'usually' emits — used as the recall target."""

    if not verdicts:
        return None
    from collections import Counter

    c: Counter[Decision] = Counter(verdicts.values())
    return c.most_common(1)[0][0]


def _compute_stats(
    *,
    rule_functions: list[Callable],
    fire_map: dict[str, set[str]],
    verdict_map: dict[str, dict[str, Decision]],
    truth_map: dict[str, Decision | None],
    trajectory_ids: list[str],
) -> list[RuleStats]:
    stats: list[RuleStats] = []
    labeled_ids = [tid for tid in trajectory_ids if truth_map.get(tid) is not None]
    for rule in rule_functions:
        name = rule.__name__
        fires = fire_map[name]
        rule_verdicts = verdict_map[name]

        agreed = sum(
            1 for tid in fires if tid in labeled_ids and truth_map[tid] == rule_verdicts[tid]
        )
        precision = agreed / len(fires) if fires else 0.0

        typical = _typical_verdict(rule_verdicts)
        if typical is None:
            recall = 0.0
        else:
            should_fire = [tid for tid in labeled_ids if truth_map[tid] == typical]
            missed = [tid for tid in should_fire if tid not in fires]
            denom = agreed + len(missed)
            recall = agreed / denom if denom else 0.0

        # Jaccard against the other rules; max = worst offender.
        max_overlap = 0.0
        for other in rule_functions:
            if other.__name__ == name:
                continue
            other_fires = fire_map[other.__name__]
            if not fires and not other_fires:
                continue
            union = fires | other_fires
            if not union:
                continue
            overlap = len(fires & other_fires) / len(union)
            if overlap > max_overlap:
                max_overlap = overlap

        stats.append(
            RuleStats(
                name=name,
                fires=len(fires),
                truth_agreed=agreed,
                precision=round(precision, 4),
                recall=round(recall, 4),
                redundancy=round(max_overlap, 4),
            )
        )
    return stats


# How much of `rule_quality` survives when the rules decide none of the pool.
# Any aggregate built only from precision rewards deletion, because dropping a
# rule removes it from the numerator *and* the denominator. A coverage factor
# is the only thing that makes deleting a working rule cost something.
#
# The floor is set so the collapse below is a loss. A filter holding three
# working rules (precisions 1.00 / 0.95 / 0.81, firing on ~140 of 200) can
# delete two of them to keep one perfect rule firing on 82:
#
#     floor 1.00 (no coverage term)  keep 0.884  drop 1.000   -> deleting pays
#     floor 0.50                     keep 0.751  drop 0.705   -> margin 0.046
#     floor 0.25                     keep 0.685  drop 0.558   -> margin 0.128
#
# At 0.50 the margin survives only in the third decimal once the 0.20 score
# weight is applied, which is not enough to steer a proposer. 0.25 leaves a
# margin the optimizer can actually see, while still giving a narrow, precise
# rule set most of its credit.
COVERAGE_FLOOR = 0.25


def _aggregate_quality(per_rule: list[RuleStats], *, covered: int = 0, labelled: int = 0) -> float:
    """Overall rule_quality: how well the rules decide, times how much they decide.

    A rule contributes `precision * (1 - redundancy)`, weighted by its fire
    count — redundancy discounts it because the same signal could be caught by
    the twin rule alone, and the weighting stops one lucky single-fire rule
    from dominating. The whole is then scaled by the share of labelled
    trajectories any rule fired on.

    Both halves are needed. Under a plain mean, dropping any rule below the
    current average raises the score, so the optimum is one perfect rule and
    nothing else: five firing rules at 0.6358 lose to one rule at 1.0000
    after a 95%- and an 81%-precision rule are deleted to raise the mean.

    `covered` / `labelled` default to 0, which disables the coverage factor;
    callers that know the fire sets should always pass them.
    """

    firing = [r for r in per_rule if r.fires > 0]
    if not firing:
        return 0.5

    total_fires = sum(r.fires for r in firing)
    weighted = sum(r.precision * (1.0 - r.redundancy) * r.fires for r in firing)
    quality = weighted / total_fires

    if labelled > 0:
        coverage = min(1.0, covered / labelled)
        quality *= COVERAGE_FLOOR + (1.0 - COVERAGE_FLOOR) * coverage

    return max(0.0, min(1.0, quality))


# ------------------------------------------------------------- LLM plumbing


def _hash_inputs(
    rules_source: str,
    rubric_md: str,
    per_rule: list[RuleStats],
    model_tag: str,
) -> str:
    h = hashlib.sha256()
    h.update(model_tag.encode())
    h.update(b"\0")
    h.update(rules_source.encode())
    h.update(b"\0")
    h.update(rubric_md.encode())
    h.update(b"\0")
    h.update(
        json.dumps(
            [r.as_dict() for r in per_rule],
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    )
    return h.hexdigest()[:16]


def _parse_llm_report(text: str) -> dict[str, Any]:
    default = {"recommendations": [], "signals_missing_from_rules": []}
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    obj: Any = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return default

    valid_recs = {"keep", "move_to_rubric", "split", "drop"}
    recs: list[dict[str, str]] = []
    for item in obj.get("rules", []) or []:
        if not isinstance(item, dict):
            continue
        recommendation = str(item.get("recommendation", "")).lower().strip()
        if recommendation not in valid_recs:
            recommendation = "keep"
        recs.append(
            {
                "name": str(item.get("name", ""))[:120],
                "recommendation": recommendation,
                "note": str(item.get("note", ""))[:400],
            }
        )
    missing: list[str] = []
    for item in obj.get("signals_missing_from_rules", []) or []:
        s = str(item).strip()
        if s:
            missing.append(s[:200])

    return {
        "recommendations": recs,
        "signals_missing_from_rules": missing,
    }
