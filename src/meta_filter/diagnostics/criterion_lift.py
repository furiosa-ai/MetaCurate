"""Per-criterion and per-rule counterfactual contribution.

"Is the rubric used?" is the wrong resolution: a criterion can be scored on
every row and still decide nothing, and a whole-rubric number cannot tell the
proposer WHICH question earns its place. This measures, for the selected
filter each iteration, how many decisions flip when

  * the whole rubric is ablated (``decide`` re-run with an empty score dict),
  * each criterion is ablated alone (its key dropped, leave-one-out),
  * each RULE is ablated alone (its fired entries removed from
    ``triggered_rules``, leave-one-out).

Rule ablation measures the rule's verdict channel only: a policy that also
imports a rule-module helper directly (``from rules import post_edit_suite``)
keeps that signal — such usage is the policy's own logic, not the rule's
firing, and the fire-stats the rule critic already publishes cannot see this
distinction either.

Everything here is pure python — ``decide`` re-runs on already-obtained
scores, so the marginal LLM cost is zero. Rows the lazy-scoring pass decided
without the rubric contribute zero flips by construction and are skipped.

Reading the two levels together also exposes redundancy: when the
whole-rubric flip count is far above the sum of leave-one-out counts, two or
more criteria are carrying the same signal (ablating one lets the other
cover), which is exactly the double-counting the unified critic polices.

Ablation drops the KEY rather than writing 0.0: absent is the honest
counterfactual for "this criterion does not exist", and policies that treat
missing and zero differently should see the difference.
"""

from __future__ import annotations

import copy
import logging

from meta_filter.core.filter_loader import LoadedFilter
from meta_filter.core.schemas import FilterDecision, Trajectory

logger = logging.getLogger(__name__)


def _rules(loaded: LoadedFilter, traj: Trajectory) -> list[dict]:
    """The solver's rule pass, with its failure containment."""

    triggered = []
    for rule in loaded.rule_functions:
        try:
            fired = rule(traj)
        except Exception as exc:  # noqa: BLE001 — mirror the solver
            fired = {
                "name": rule.__name__,
                "verdict": "REVIEW",
                "evidence": [f"rule raised {type(exc).__name__}: {exc}"],
            }
        if fired:
            triggered.append({**fired, "name": rule.__name__})
    return triggered


def _score_of(loaded: LoadedFilter, traj: Trajectory, scores: dict, triggered) -> float | None:
    try:
        payload = loaded.policy_module.decide(traj, scores, copy.deepcopy(triggered))
    except Exception:  # noqa: BLE001 — an ablation probe must never raise
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("score")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def measure_ranking(
    loaded: LoadedFilter,
    trajectories: list[Trajectory],
    decisions: list[FilterDecision],
) -> dict:
    """Rank-lift: how much the ORDERING moves when an item is ablated.

    Counting decision flips would starve the rubric in a regime where flips
    are rare. Here every criterion score moves rows
    continuously, so contribution is measured as the Kendall distance between
    the original ranking and the ranking produced with the item ablated —
    always defined, never trivially zero for an item that actually shifts
    scores. Pure python on already-paid scores; zero marginal LLM cost.
    """

    from meta_filter.gate.ranking import kendall_distance

    by_id = {t.trajectory_id: t for t in trajectories}
    original: dict[str, float] = {}
    prepared: list[tuple[Trajectory, FilterDecision, list[dict]]] = []
    criteria: set[str] = set()
    rule_names: set[str] = set()
    for record in decisions:
        traj = by_id.get(record.trajectory_id)
        if traj is None or record.rank_score is None:
            continue
        triggered = _rules(loaded, traj)
        prepared.append((traj, record, triggered))
        original[record.trajectory_id] = float(record.rank_score)
        criteria.update((record.rubric_scores or {}).keys())
        rule_names.update(t.get("name") for t in triggered)

    def ablated_ranking(drop_criterion: str | None, drop_rule: str | None) -> dict[str, float]:
        out: dict[str, float] = {}
        for traj, record, triggered in prepared:
            scores = dict(record.rubric_scores or {})
            if drop_criterion is not None:
                scores.pop(drop_criterion, None)
            rules = triggered
            if drop_rule is not None:
                rules = [t for t in triggered if t.get("name") != drop_rule]
            s = _score_of(loaded, traj, scores, rules)
            if s is not None:
                out[traj.trajectory_id] = s
        return out

    criterion_shift = {
        c: round(kendall_distance(original, ablated_ranking(c, None)), 4) for c in sorted(criteria)
    }
    rule_shift = {
        r: round(kendall_distance(original, ablated_ranking(None, r)), 4)
        for r in sorted(rule_names)
    }
    rubric_shift = round(
        kendall_distance(
            original,
            {
                traj.trajectory_id: s
                for traj, record, triggered in prepared
                if (s := _score_of(loaded, traj, {}, triggered)) is not None
            },
        ),
        4,
    )
    return {
        "rows_ranked": len(original),
        "rubric_rank_shift": rubric_shift,
        "criterion_rank_shift": criterion_shift,
        "rule_rank_shift": rule_shift,
    }


ZERO_LIFT_ACTION = """\
  - Remove them in EVERY candidate you write, on top of that candidate's own
    hypothesis. This is bookkeeping, not an idea: it does NOT count as a second
    hypothesis, it does NOT use the iteration's one deletion slot, and the
    deletion floor does not count it either. A criterion at 0 costs a scorer
    call on every row and a rule at 0 costs an evaluation; both buy a
    measurement nothing reads.
  - Remove it properly — a criterion's `rubric.md` section AND its entries in
    `RUBRIC_CRITERIA` and `RUBRIC_WEIGHTS`; a rule's `rule_` function AND any
    policy branch that reads its verdict. Dropping only the rubric section
    leaves the scorer paying for it.
  - Do NOT give the removal a candidate of its own. Alone it reproduces the
    parent's ordering exactly, so it ties, and a tie is promoted only when
    every sibling also fails the gate. That is how a zero-lift criterion
    survived a whole run once already.
  - KEEP it in one candidate only if that candidate's hypothesis BUILDS ON it:
    weight in decide() AND anchors that separate these rows, a merge into
    another axis, or a policy branch that reads it as a precondition. Those are
    real ideas and earn a slot. Leaving it declared at zero is not.
  - The zero was measured by ablating this item from the CURRENT filter and
    re-ranking. It is strong evidence the removal is free, not a proof: if your
    candidate changes how decide() aggregates the rubric — a normalisation, a
    min/max over the score set, a floor on the worst criterion — re-check that
    the removal is still inert under YOUR policy."""


def zero_lift_items(report: dict) -> tuple[list[str], list[str]]:
    """The criteria and rules whose ablation moves the ordering by exactly 0.

    Distinct from the two health findings that already exist, and not caught
    by either. A DEAD axis is one whose SCORES sit at one level — its anchors
    are broken. An UNREAD axis is one `decide()` never references. This is the
    third case: a criterion that scores +0/+1/+2 across the pool (not dead)
    and is read by decide() (not unread), with its weight set to 0.0 — so it
    varies, it is read, and it cannot change a single pair's ordering, while
    still costing a scorer call on every exam row and every pool row.

    Measured, not declared: the number comes from ablating the item and
    re-ranking, so a candidate cannot talk its way out of it, and a weight of
    zero is only one of the ways an item lands here (a term the clamp always
    swallows, or one read in a branch that never fires, arrive at the same
    place).

    Returns:
        ``(criteria, rules)``, each sorted, each holding only exact zeros.
        Near-zero is deliberately not included: 0.0001 is a weak axis, which
        is a judgement call, while 0 is a fact about the ordering.
    """

    crit = sorted(k for k, v in (report.get("criterion_rank_shift") or {}).items() if v == 0)
    rules = sorted(k for k, v in (report.get("rule_rank_shift") or {}).items() if v == 0)
    return crit, rules


def render_ranking(report: dict) -> str:
    """Prompt-ready rank-lift lines."""

    if not report or not report.get("rows_ranked"):
        return (
            "rank lift: no rows carried a rank score this iteration — the "
            "policy is not emitting the ranking contract's `score` field"
        )
    crit = _ranked_float(report.get("criterion_rank_shift") or {})
    rules = _ranked_float(report.get("rule_rank_shift") or {})
    return (
        f"rank lift (Kendall distance of the ordering when one item is "
        f"ablated, over {report['rows_ranked']} rows): whole rubric "
        f"{report.get('rubric_rank_shift', 0)}; per criterion — "
        f"{crit or 'none shifts the ordering'}"
        f"\n- per rule — {rules or 'no rule shifts the ordering'}"
        "\n- these are DIAGNOSTICS, not targets: near-zero says the item "
        "does not move the ordering (prune, merge, or sharpen it); "
        "maximizing shift is wrong too — one item owning the whole ordering "
        "is a knife edge" + _zero_lift_line(report)
    )


def zero_lift_note(report: dict) -> str:
    """The ZERO LIFT finding alone, for prompts that carry no lift table.

    The revision leg is the caller: it is a fresh process that never saw the
    diagnosis prompt, and it is the leg that ships.
    """

    return _zero_lift_line(report).lstrip("\n")


def _zero_lift_line(report: dict) -> str:
    """The named finding for items at exactly zero, with its own action.

    The boundary is marked against the data, not only asserted in prose: the
    lift table printed directly above this sorts descending, so the exact zero
    always sits next to the smallest non-zero item, and the framing line one
    bullet up tells the proposer to prune NEAR-zero items too. Naming the
    nearest survivor by value is what stops "in EVERY candidate" from being
    read onto an axis that is merely small.
    """

    crit, rules = zero_lift_items(report)
    if not crit and not rules:
        return ""
    parts = []
    if crit:
        parts.append(f"criteria {', '.join(crit)}")
    if rules:
        parts.append(f"rules {', '.join(rules)}")
    shifts = {
        **(report.get("criterion_rank_shift") or {}),
        **(report.get("rule_rank_shift") or {}),
    }
    survivors = sorted((v, k) for k, v in shifts.items() if v > 0)
    boundary = (
        f" Exactly these, and nothing else: the next-smallest item is "
        f"{survivors[0][1]} at {survivors[0][0]}, which is NOT on this list and "
        f"must not be removed this way."
        if survivors
        else ""
    )
    rows = f" (over {report['rows_ranked']} rows)" if report.get("rows_ranked") else ""
    return (
        f"\n- ZERO LIFT — {'; '.join(parts)} move the ordering by exactly 0"
        f"{rows}.{boundary}\n{ZERO_LIFT_ACTION}"
    )


def _ranked_float(shifts: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in sorted(shifts.items(), key=lambda x: -x[1]))


# The criterion-signal table. The rank-lift line above
# says how much the ORDERING moves when a criterion is removed, which is a
# marginal quantity — an axis that duplicates another has near-zero lift and
# is still informative on its own. Alone-concordance is the complementary
# reading: how well each criterion, by itself, orders the judged pairs. Read
# together they separate "redundant" (high alone, low lift) from "noise"
# (low alone, low lift) from "load-bearing" (high alone, high lift), which
# neither number does on its own and which the proposer had to guess at.
CRITERION_SIGNAL_BUDGET = 600


def single_criterion_concordance(
    decisions: list[FilterDecision],
    pairs: list[tuple[str, str]],
    *,
    ties: list[tuple[str, str]] | None = None,
) -> dict[str, dict]:
    """Each criterion's scores alone, against the judged pairs.

    `pairs` are judge-ordered `(hi, lo)`; a pair is offered to a criterion
    only when both rows carry a score for it (`unscored="skip"` — a row the
    lazy pass never scored says nothing about the criterion). Returns
    ``{criterion: {"concordance", "n_pairs"}}``; empty without pairs.
    """

    from meta_filter.gate.ranking import concordance_detail

    if not pairs:
        return {}
    by_criterion: dict[str, dict[str, float]] = {}
    for record in decisions:
        for criterion, value in (record.rubric_scores or {}).items():
            try:
                by_criterion.setdefault(criterion, {})[record.trajectory_id] = float(value)
            except (TypeError, ValueError):
                continue
    out: dict[str, dict] = {}
    for criterion, scores in sorted(by_criterion.items()):
        detail = concordance_detail(scores, list(pairs), unscored="skip", ties=list(ties or []))
        out[criterion] = {
            "concordance": round(detail.concordance, 4) if detail.n_scored else None,
            "n_pairs": detail.n_scored,
        }
    return out


def render_criterion_signal(
    alone: dict[str, dict],
    lift: dict[str, float] | None,
    stats: dict | None,
    *,
    budget: int = CRITERION_SIGNAL_BUDGET,
) -> str:
    """The prompt table: criterion | alone concordance | lift when ablated | dead?

    Sorted by alone-concordance, highest first, so the axis that carries
    the ordering is the first line and a dead one the last. Rows past
    `budget` characters are folded into one "+N more" line; the header and
    the reading guide are never dropped.
    """

    if not alone:
        return ""
    lift = lift or {}
    per = (stats or {}).get("per_criterion") or {}
    rows = sorted(
        alone.items(),
        key=lambda kv: -(kv[1].get("concordance") if kv[1].get("concordance") is not None else -1),
    )
    n_pairs = max((v.get("n_pairs", 0) for v in alone.values()), default=0)
    lines = [
        f"criterion signal (last winner, over up to {n_pairs} judged train pairs): "
        "criterion | alone concordance | rank lift when ablated | dead?",
    ]
    body: list[str] = []
    for name, entry in rows:
        conc = entry.get("concordance")
        conc_text = f"{conc:.3f}" if conc is not None else "n/a"
        lift_text = f"{lift[name]:.4f}" if name in lift else "n/a"
        dead = "DEAD" if (per.get(name) or {}).get("dead") else "live"
        body.append(f"  {name} | {conc_text} | {lift_text} | {dead}")
    guide = (
        "  (alone ~0.5 = orders pairs no better than chance; high alone + low lift = "
        "redundant with another axis; low alone + low lift = noise)"
    )
    shown: list[str] = []
    used = len(lines[0]) + len(guide) + 2
    for i, line in enumerate(body):
        remaining = len(body) - i
        tail = len(f"  ... +{remaining} more") + 1 if remaining > 1 else 0
        if used + len(line) + 1 + tail > budget:
            shown.append(f"  ... +{remaining} more")
            break
        shown.append(line)
        used += len(line) + 1
    return "\n".join(lines + shown + [guide])


def rule_pricing(loaded_filter, decisions, trajectories=None) -> dict:
    """What each rule's firing is actually WORTH, as opposed to what the
    policy's source says it is worth.

    `Solver` replaces every fired rule's `name` with the Python function name
    (`solver.py`, "prefer the function name so the trigger label matches the
    rules.py source-of-truth"). A policy that keys its price table by the
    label the rule DECLARES — which is what the proposer reads when it writes
    rules.py — therefore looks up a key that is never present, and
    `dict.get(name, fallback)` hands back the fallback without raising.

    Two outcomes, both silent:

    * the fallback is a number, so the rule works but not as written: a rule
      priced -14.0 with an exception zeroing it under some condition misses
      the lookup, the `name ==` exception can never match either, and every
      firing row takes the fallback — including rows the exception was
      written to spare.
    * the fallback is zero, so the rule is inert: it fires on most rows and
      moves no score at all.

    `rule_rank_shift` sees the second case but reports it as "does not move
    the ordering", whose advice is to prune — so a rule whose WIRING is broken
    reads as a rule that is not worth having, and gets deleted. This
    separates the two: it re-runs `decide()` with one rule's firing withheld
    and reports the delta that rule is actually responsible for.

    Returns a mapping of rule name to
    ``{"fires": int, "deltas": {delta: count}, "inert": bool}``.
    """

    policy = getattr(loaded_filter, "policy_module", None)
    decide = getattr(policy, "decide", None)
    if decide is None:
        return {}

    # `FilterDecision.triggered_rules` is a list of NAMES, not the dicts
    # `decide()` was handed, and the counterfactual needs the dicts. Rules are
    # pure functions of the row, so re-firing them reproduces exactly what the
    # policy saw — including the runner's own substitution of `rule.__name__`
    # for the declared label, which is the whole point of this readout.
    by_id = {t.trajectory_id: t for t in (trajectories or [])}

    def refire(trajectory_id):
        traj = by_id.get(trajectory_id)
        if traj is None:
            return []
        out = []
        for rule in getattr(loaded_filter, "rule_functions", None) or []:
            try:
                got = rule(traj)
            except Exception:  # pragma: no cover - a broken rule is not ours to fix
                continue
            if got:
                out.append({**got, "name": rule.__name__})
        return out

    per_rule: dict[str, dict] = {}
    for decision in decisions or []:
        fired = refire(getattr(decision, "trajectory_id", None))
        if not fired:
            for name in getattr(decision, "triggered_rules", None) or []:
                label = name if isinstance(name, str) else (name or {}).get("name", "?")
                per_rule.setdefault(label, {"fires": 0, "deltas": {}})["fires"] += 1
            continue
        scores = getattr(decision, "rubric_scores", None) or {}
        actual = getattr(decision, "rank_score", None)
        if actual is None:
            continue
        for index, rule in enumerate(fired):
            name = rule.get("name", "?")
            entry = per_rule.setdefault(name, {"fires": 0, "deltas": {}})
            entry["fires"] += 1
            without = [r for i, r in enumerate(fired) if i != index]
            try:
                other = decide(None, scores, without)
            except Exception:  # pragma: no cover - a policy may need the row
                continue
            try:
                delta = round(float(actual) - float(other.get("score", 0.0)), 3)
            except (TypeError, ValueError):
                continue
            entry["deltas"][delta] = entry["deltas"].get(delta, 0) + 1

    for entry in per_rule.values():
        priced = entry["deltas"]
        entry["inert"] = bool(priced) and set(priced) == {0.0}
    return per_rule


def render_rule_pricing(report: dict) -> str:
    """Prompt-ready wiring line. Empty string when every rule is priced."""

    if not report:
        return ""
    broken = [n for n, e in report.items() if e.get("inert")]
    unpriced = [n for n, e in report.items() if e.get("fires") and not e.get("deltas")]
    if not broken and not unpriced:
        lines = ", ".join(
            f"{n}: {e['fires']} fires, {_fmt_deltas(e['deltas'])}"
            for n, e in sorted(report.items())
        )
        return f"rule pricing — {lines}"
    out = []
    if broken:
        out.append("RULE WIRING BROKEN — a rule fires and changes no score.")
    for name in sorted(broken):
        out.append(
            f"  `{name}` fired {report[name]['fires']} time(s) and moved the score by 0.0 "
            f"every time. Its firing reaches decide() with name={name!r} — the Python "
            f"function name, which the runner substitutes for whatever label rules.py "
            f"declares. If policy.py keys its price table or an `if name ==` branch by a "
            f"different spelling, the lookup misses and the fallback applies. This is a "
            f"WIRING fault: fix the key, do not delete the rule on the strength of its "
            f"zero rank lift."
        )
    if unpriced:
        out.append("rule pricing — not measurable for:")
        for name in sorted(unpriced):
            out.append(f"  `{name}` fired {report[name]['fires']} time(s), price not determined.")
    return "\n".join(out)


def _fmt_deltas(deltas: dict) -> str:
    if not deltas:
        return "no measurable delta"
    return ", ".join(f"{d:+g}x{c}" for d, c in sorted(deltas.items()))
