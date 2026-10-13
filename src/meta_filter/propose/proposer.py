"""Proposer — asks the LLM to emit N candidate filters.

Design notes
------------
The proposer prompt intentionally does **not** tell the LLM to "convert
semantic rubric criteria into deterministic rules" or use the phrase
"rule promotion". The research question is whether such structural changes
emerge under quality + cost + complexity pressure. So we describe:

    * the optimization objective (the v2 score the selector maximizes)
    * the three files the LLM may freely rewrite
    * the current values of those files
    * FN / FP counterexamples with brief context
    * the metrics of the current filter

and stop there.

The prompt also embeds a compact ``<STATE_JSON>...</STATE_JSON>`` block so
the deterministic mock client can dispatch on structured state without doing
any natural-language parsing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from meta_filter.core.filter_loader import (
    LoadedFilter,
    load_filter,
    materialize_candidate,
)
from meta_filter.core.schemas import CandidateEvaluation, CounterExample, FilterMetrics
from meta_filter.diagnostics.criterion_stats import UNREAD_TAG_MARK
from meta_filter.gate.complexity import measure_filter_complexity
from meta_filter.gate.constraints import FROZEN_FILES, ConstraintLimits, hypothesis_lines
from meta_filter.gate.metrics import DEFAULT_SCORE_WEIGHTS
from meta_filter.gate.ranking import DEGENERACY_PENALTY
from meta_filter.llm.base import LLMClient, LLMResponse
from meta_filter.llm.mock import MockLLMClient
from meta_filter.score_tool import ScoreToolAccess
from meta_filter.visibility.pseudonym import Pseudonymizer

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


_SYSTEM_PREAMBLE = (
    "You are an optimization engineer improving a filter that decides whether "
    "an agent trajectory is useful training data. The trajectories are rollouts "
    "of SWE-style coding agents — they are given a GitHub-issue-style task and "
    "must localize the bug, edit source, and verify with tests. The three "
    "canonical failure modes we care about are: (a) empty/trivial patches "
    "(<400 bytes) despite claiming completion, (b) run-away exploration or "
    "action loops that never produce a meaningful edit, (c) declared completion "
    "with no test invocation. Successful trajectories usually show "
    "explore-then-edit-then-verify structure using tools like grep/cat/sed/"
    "apply_patch/pytest."
)

_RANKING_EPILOGUE = (
    "`pair_concordance` dominates: the loop compares TRAJECTORY PAIRS. Pairs "
    "every candidate orders the same way are pseudo-truth for free; a "
    "decile-stratified sample of disputed pairs is adjudicated by a judge "
    "who reads both transcripts. Your concordance is the fraction of those "
    "truth pairs your continuous score orders correctly — a tie counts half. "
    'Your `decide()` must therefore return `"score": <float 0-100>` on '
    "every row; the decision field is bookkeeping."
    "\n\n"
    "Every judged pair is asked TWICE, once in each A/B presentation order, "
    "and only a pair both orders name the same winner for enters the truth "
    "set. Pairs the judge contradicts itself on leave the exam entirely — "
    "they are not half credit, and there is nothing to be gained by aiming "
    "at them. The `ranking.json` in each iteration records how many came "
    "back that way."
    "\n\n"
    "You may freely rewrite the three files shown below. Prefer changes that "
    "provably reorder concrete disputed pairs without scrambling the pairs "
    "you already order correctly."
    "\n\n"
    "Rubric edits are cheap: scoring is cached PER CRITERION, so editing "
    "one criterion re-scores only that criterion and unchanged criteria keep "
    "bit-identical scores. Write each criterion self-contained. Adding, "
    "splitting or reweighting criteria is fine. Do not restrict yourself to "
    "policy-threshold tweaks."
)


def _redundancy_report_note(threshold: float) -> str:
    """The redundancy term as the loop actually applies it: reported, not charged.

    Said in one sentence and in one place, because the proposer is shown the
    number in four others (STATE_JSON, the criterion-health block, the
    precheck block, `precheck.json`) and a number with no stated status is
    read as a scored term. It is not one here: |rho| between two criteria is
    only measurable after the pool has been scored, so there is nothing for
    the proposer to optimize at generation time, and charging it in selection
    made a newly added criterion pay for its first contact with the
    incumbents.
    """

    return (
        f"Redundancy (|rho| > {threshold:g} between criteria) is measured and "
        f"shown here but does not enter this iteration's selection; it is "
        f"charged to the final filter's reported score, so avoid it where you "
        f"can without giving up ordering power."
    )


_PRECHECK_NOTE = (
    "After you write your candidates you get ONE precheck, and then one "
    "chance to revise them. What the precheck computes, exactly:"
    "\n\n"
    "  * Each candidate is run against the truth pairs a judge has ALREADY "
    "adjudicated in earlier iterations of this run — the accumulated "
    "evaluation set, over the trajectories those pairs name. Its concordance "
    "on them and every other live term are combined with the formula above. "
    "The number you are shown is the formula the precheck reports — evidence "
    "for your revision, not the acceptance decision."
    "\n"
    "  * The rubric cells come from the scorer cache where a criterion's text "
    "is unchanged and are scored fresh where it is not. The loop pays for "
    "that scoring, not you: do not shape an edit around cache hits. A rubric "
    "edit is evaluated exactly like a rules edit, by this precheck and then "
    "by the full pass. Your rules and policy are deterministic code and are "
    "re-run in full either way."
    "\n\n"
    "What the precheck is NOT, so you do not read more into it than it says:"
    "\n"
    "  * NO judge is called. No new verdict is bought, no pair changes side, "
    "and nothing you write can add supervision to this set."
    "  * It is measured BEFORE this iteration's own judging round, on the "
    "pairs bought previously, not on the whole training pool. The final "
    "score is measured after that round and will differ."
    "\n\n"
    "The precheck lands as `candidates/<id>/precheck.json` beside each "
    "candidate and as a summary in the revision prompt. The revision leg is "
    "the LAST thing you do: after it the loop scores your candidates for "
    "real, and there is no third round."
)

# How acceptance works, as the proposer must hear it: the formula is
# evidence, and a paired test on hidden pairs picks the winner.
_SELECTION_GATE_NOTE = (
    "Selection is decided on a fixed set of trajectory pairs you cannot see: "
    "rows outside this pool, judged once at the start of the run. A candidate "
    "replaces the current filter only when it orders significantly more of "
    "those pairs correctly than the current filter does (paired test; b = "
    "pairs it gets right that the current filter gets wrong, c = the reverse). "
    "The selector's line in `iterations/*/selection.json` reports b, c, z and "
    "p for every candidate."
    "\n\n"
    "The train-pair concordance you are shown here — `pair_concordance`, and "
    "the misordered pairs — is the evidence you optimise against. Winning it "
    "is necessary but not sufficient: a change that only fixes the specific "
    "pairs listed and nothing general will not move the hidden set."
)

# What the stated formula is: evidence, not the acceptance decision.
_PREAMBLE_OBJECTIVE = (
    "Your job is to MAXIMIZE this score — it is the formula the precheck "
    "reports and the evidence you are shown; selection itself is decided on "
    "hidden pairs, as described below. "
)


def _degeneracy_note(min_distinct_levels: int) -> str:
    """The degeneracy guard, stated as the code applies it.

    Rendered rather than written into `_RANKING_EPILOGUE` because both halves
    of it are run-dependent. "A scorer with too few distinct values is zeroed
    outright" would understate the charge and name no number:

    * "Zeroed" is half of it. A degenerate candidate has its concordance
      forced to 0 AND `ranking.DEGENERACY_PENALTY` taken off its score, so
      the score goes negative and no secondary term can recover it. Told only
      "zeroed", a proposer can reasonably read the guard as a cap worth
      trading a little spread against.
    * The floor is not a constant. It scales with the pool
      (`ranking.default_min_distinct_levels`), so on a 50-row pool it is 3
      where a 200-row run sees 5. A prompt naming neither leaves the proposer
      guessing at the one threshold that can void its whole agreement term.
    """

    return (
        f"Degeneracy guard, stated precisely because it is not a soft "
        f"penalty: a candidate whose scores take fewer than "
        f"{min_distinct_levels} DISTINCT values across the pool cannot "
        f"express an ordering. Its concordance is forced to 0.000 and a "
        f"further {DEGENERACY_PENALTY:.3f} is subtracted from its score, so "
        f"the score goes negative and no other term can recover it. This is "
        f"not a threshold to trade against — compose enough independent "
        f"signals that your scores genuinely spread. Do not manufacture the "
        f"spread with noise: noise is ordering error and loses pairs "
        f"directly."
    )


def build_proposer_system(
    weights: dict[str, float] | None = None,
    *,
    redundancy_penalty: float = 0.0,
    redundancy_threshold: float = 0.7,
    min_distinct_levels: int | None = None,
) -> str:
    """System prompt stating the objective the precheck reports.

    The formula is rendered from `weights` rather than written out, so the
    stated objective and the applied one cannot drift apart.

    Zero-weight terms are DROPPED rather than printed with a 0.00
    coefficient: a proposer shown `+ 0.00 * rubric_quality` spends part of
    its reasoning on a term that cannot move its score. The redundancy
    penalty is not a term of the formula — selection never charges it — so
    it is stated as a separate sentence when it is armed, since the proposer
    is shown the number in several other places.

    `min_distinct_levels` is the run's effective degeneracy floor, which
    scales with the pool. Passing it renders `_degeneracy_note`; omitting it
    drops the paragraph, on the same principle as the zero-weight terms — a
    guard described with the wrong number is worse than one left unstated.

    The prompt also describes the precheck round the loop runs between the
    proposer's two legs, and says that acceptance is decided by a paired test
    on hidden pairs (`_SELECTION_GATE_NOTE`), so the formula is evidence.
    """

    w = {**DEFAULT_SCORE_WEIGHTS, **(weights or {})}
    lines = ["  score = pair_concordance"]
    optional = [
        (w["lambda_rubric_quality"], "+", "rubric_quality"),
        (w["lambda_rule_quality"], "+", "rule_quality"),
        (w["lambda_review"], "-", "review_rate"),
        (w["lambda_cost"], "-", f"min(1, cost / {w['cost_budget']:g})"),
        (
            w["lambda_complexity"],
            "-",
            f"min(1, complexity / {w['complexity_budget']:g})",
        ),
    ]
    for weight, sign, term in optional:
        if weight:
            lines.append(f"        {sign} {weight:.2f} * {term}")
    formula = "\n".join(lines)
    epilogue = _RANKING_EPILOGUE
    if redundancy_penalty:
        epilogue += f"\n\n{_redundancy_report_note(redundancy_threshold)}"
    if min_distinct_levels:
        epilogue += f"\n\n{_degeneracy_note(min_distinct_levels)}"
    epilogue += f"\n\n{_SELECTION_GATE_NOTE}"
    epilogue += f"\n\n{_PRECHECK_NOTE}"
    if w["lambda_rubric_quality"] or w["lambda_rule_quality"]:
        # Said only where it is true. The two critics still run and still
        # publish these numbers as diagnostics; describing them as scored
        # while their weight is 0 sends the proposer after prose quality it
        # cannot be paid for.
        epilogue += (
            "\n\n`rubric_quality` and `rule_quality` are the rubric critic's "
            "and rule critic's own assessments of your criteria and rules, so "
            "vague or non-discriminating items lose points even when the "
            "decisions come out right."
        )
    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"{_PREAMBLE_OBJECTIVE}"
        "Every term of it is written below; "
        "terms that are not written down have weight zero and buy you "
        "nothing:\n\n"
        f"{formula}\n\n"
        f"{epilogue}"
    )


PROPOSER_SYSTEM = build_proposer_system()


_CAP_COMPONENTS = ("rubric_criteria", "rule_count", "rules_loc", "policy_loc", "rubric_chars")


def render_edit_discipline(
    *,
    n_candidates: int,
    parent_counts: dict[str, int],
    initial_counts: dict[str, int] | None = None,
    candidates_scored: int | None = None,
    hypothesis_source: str,
    limits: ConstraintLimits | None = None,
    frozen: tuple[str, ...] = (),
    score_tool: bool = False,
) -> str:
    """The edit discipline, as the proposer reads it: one mechanism per
    candidate, one deletion per iteration, and the growth caps as numbers.

    The caps are printed with the current filter's actual counts filled in
    rather than as rules ("+15 %"), because the check that enforces them
    (`constraints.check_candidate`) is arithmetic on those counts and a
    proposer told the rule but not the numbers has to redo the arithmetic
    itself — or not do it. `initial_counts` is the
    run's starting filter, which the run-level hard cap and deletion floor
    are measured against; when the caller does not know it (a proposer built
    outside the optimizer) the parent stands in, which is exact at iteration
    0 and conservative after.

    `hypothesis_source` names where this proposer states each candidate's
    hypothesis — the tool-using proposer writes `diagnosis.json`, the static
    one answers in JSON — because the loop copies it into `rationale.md` as
    the one `Hypothesis:` line the precheck counts.

    The gating sentence appears only when `candidates_scored` is set below
    `n_candidates`: a proposer told that half its candidates will be dropped
    when none will be hedges for no reason. No sentence names an outcome
    field: the rows carry none, and a sentence listing them would be the
    prompt telling the proposer they exist.
    """

    limits = limits or ConstraintLimits.DEFAULT
    initial = initial_counts or parent_counts
    p = {k: int(parent_counts.get(k, 0)) for k in _CAP_COMPONENTS}
    i = {k: int(initial.get(k, 0)) for k in _CAP_COMPONENTS}
    soft = limits.soft_add
    hard = limits.hard_ratio

    def _soft_max(name: str) -> int:
        # `ConstraintLimits.soft_max_for` is the one place the cap is
        # computed, shared with `constraints.check_candidate`, so the number
        # printed here is the number a candidate may actually reach — share,
        # absolute allowance and `soft_floor` alike.
        return limits.soft_max_for(name, p[name]) or p[name]

    def _hard_max(name: str) -> str:
        return str(int(hard * i[name])) if i[name] > 0 else "no cap (initial had none)"

    # The floors come from the same method the check fires on, so the prompt
    # prints the number a candidate may actually reach — never a rule the
    # proposer has to evaluate itself.
    criteria_floor = limits.floor_for("rubric_criteria", i["rubric_criteria"]) or 0
    rules_floor = limits.floor_for("rule_count", i["rule_count"]) or 0
    criteria_allowance = limits.deletion_floor.get("rubric_criteria", 0)
    rules_allowance = limits.deletion_floor.get("rule_count", 0)
    rules_minimum = limits.floor_min.get("rule_count")
    rules_floor_rule = f"initial minus {rules_allowance}"
    if rules_minimum is not None:
        rules_floor_rule += f", never below {rules_minimum}"
    distinct = min(4, n_candidates)
    lines = [
        "## Edit discipline — one mechanism per candidate, bounded growth",
        "",
        f"You are writing exactly {n_candidates} candidates. Together they must test",
        f"at least {distinct} distinct mechanisms — {n_candidates} variations on one idea",
        "is one candidate written several times.",
        "",
        "**One mechanism per candidate.** Every candidate's `rationale.md` carries",
        "exactly one line beginning `Hypothesis:` that states the single mechanism",
        "it tests. State it as",
        f"{hypothesis_source},",
        "and the loop writes it there. A",
        "candidate touching two of rubric.md / rules.py / policy.py must explain",
        "both edits under that one hypothesis — a new criterion plus the policy",
        "branch that reads it is one mechanism; a new criterion plus a retuned",
        "rule threshold is two, and a rationale declaring two hypotheses is",
        "rejected before scoring. Meta-Harness (App. A.2) logged six consecutive",
        "regressions from candidates that changed several things at once before",
        "a single-mechanism candidate won; a confounded edit teaches nothing even",
        "when it wins.",
        "",
        "**At most one candidate per iteration may delete** — remove a rubric",
        "criterion, a `rule_` function or a policy branch (merging two criteria",
        "into one counts as deleting one). That candidate must cite the judged",
        "pairs that motivate the removal. A second deletion candidate is held",
        "back unscored. Renaming a criterion or a rule counts as a deletion —",
        "the old name is gone — so keep names stable unless you mean to remove it.",
        "A removal the observations name under ZERO LIFT is exempt from this",
        "slot and from the deletion floor: the harness measured it at exactly",
        "zero rank shift, so it can hide no damage, and it cites that",
        "measurement instead of judged pairs. Put it in every candidate.",
        "",
        "**Growth caps, as numbers.** The current filter has "
        f"{p['rubric_criteria']} rubric criteria,",
        f"{p['rule_count']} `rule_` functions, {p['rules_loc']} rules.py LOC, "
        f"{p['policy_loc']} policy.py LOC",
        f"and {p['rubric_chars']} rubric.md characters. Per iteration a candidate may add",
        f"at most +{soft.get('rubric_criteria', 0):g} criteria (to "
        f"{_soft_max('rubric_criteria')}), +{soft.get('rule_count', 0):g} rules (to "
        f"{_soft_max('rule_count')}), and",
        f"+{soft.get('rules_loc', 0):.0%} LOC or characters (rules.py to "
        f"{_soft_max('rules_loc')} LOC,",
        f"rubric.md to {_soft_max('rubric_chars')} characters). Candidates over these "
        "soft caps are",
        "still scored but must clear a stricter threshold, one z-unit higher (about",
        "α 0.0003 instead of 0.0076), to be accepted. Across the run",
        f"no capped component may exceed {hard:g}x the initial filter (criteria "
        f"{_hard_max('rubric_criteria')}, rules",
        f"{_hard_max('rule_count')}, rules.py {_hard_max('rules_loc')} LOC, rubric.md",
        f"{_hard_max('rubric_chars')} characters). policy.py LOC is not capped, because a",
        "policy line costs no LLM call — but policy.py composes rubric scores and rule",
        "verdicts and nothing else. **Reading the trajectory from `decide()` is",
        "refused**: a candidate that grades the transcript, the diff or the metadata",
        "in policy.py is invalid and is not scored. Deterministic signal goes in",
        "rules.py, where it is a `rule_` function and where the LOC budget applies.",
        f"Criteria may never fall below the initial count minus {criteria_allowance} "
        f"({criteria_floor} criteria) and",
        f"rules never below the {rules_floor_rule} ({rules_floor} rules) — pruning rules",
        "that do not order pairs is allowed and has won before. A candidate past a",
        "hard cap or under a floor is not scored at all.",
        "",
        "**A criterion at weight 0 is INVALID.** Declaring one in `RUBRIC_CRITERIA`",
        "and mapping it to 0 buys a scorer call on every row for a term the weighted",
        "sum multiplies by nothing. If an axis does not earn its weight, remove it —",
        "the deletion floor does not protect a criterion the current filter already",
        "declares at zero, so that removal is always available. Zeroing a weight to",
        "find out whether an axis matters is not needed either: the rank-lift line",
        "in the observations already answers it, measured, for free. Inheriting a",
        "zero-weight criterion is not held against you; writing one is.",
    ]
    lines.extend(
        [
            "",
            "**This run carries no outcome metadata.** Every row holds the task, the",
            "transcript and the submitted diff, plus process counts where the pool records",
            "them; there is no grading outcome, no test result and no reference fix.",
            "`rules.py`, `policy.py` and `rubric.md` may not name one: a candidate that",
            "does is INVALID at precheck and never scored.",
            "",
            "**Rubric health.** A candidate that introduces a NEW dead axis (>= 95% of",
            "rows at one level) or a NEW criterion pair at |rho| >= 0.70, relative to the",
            "current filter, must clear the stricter one-z-higher threshold; defects the",
            "current filter already has are not held against you.",
            "",
            "**Growth needs evidence; making the filter smaller does not.** A candidate",
            "that makes the filter SMALLER in every component — no more criteria, rules,",
            "rules.py lines, policy.py lines or rubric characters than the current filter",
            "has, and strictly less overall — CLEARS THE GATE whenever it loses no",
            "selection pair (b >= c): no significance test, no discordant-pair floor, it",
            "does not use the iteration's one deletion, and it is not rejected as a no-op",
            "when b = c = 0. Clearing the gate is not promotion: exactly one candidate is",
            "promoted per iteration, so a b = c = 0 prune is promoted only when every",
            "sibling also fails. So pruning rules that never fire or merging two criteria",
            "into one is worth a slot when the shrinking IS the hypothesis — the best",
            "filter this project has measured got there by cutting nine rule functions to",
            "five. Removing something the observations already name under ZERO LIFT is",
            "not that: every sibling can carry it for free, so it never earns a slot of",
            "its own.",
        ]
    )
    if frozen:
        names = ", ".join(f"`{FROZEN_FILES[name]}`" for name in frozen if name in FROZEN_FILES)
        only = [f"`{f}`" for k, f in FROZEN_FILES.items() if k not in frozen] or ["nothing"]
        tool_note = (
            " Score every rewritten or new criterion with the criterion-scoring tool "
            "before you ship it — that is the only way to see what the scorer does "
            "with your anchors."
            if score_tool
            else ""
        )
        lines.extend(
            [
                "",
                f"**Frozen files.** This run freezes {names}: a candidate that changes",
                f"them is INVALID at precheck and never scored, so only {', '.join(only)} may",
                f"change — a smaller filter buys no exception.{tool_note}",
            ]
        )
    if candidates_scored is not None and candidates_scored < n_candidates:
        lines.extend(
            [
                "",
                f"Only {candidates_scored} of your {n_candidates} candidates are fully "
                "scored — the precheck on",
                "already-judged pairs decides which. The rest stay on disk and are recorded",
                "as `gated_out` in the iteration's `precheck.json`, so a held-back idea is",
                "evidence for next iteration, not a wasted slot.",
            ]
        )
    return "\n".join(lines)


# Under the selection gate the misordered pairs are evidence, not the exam;
# said where the pairs are shown, because that is where a proposer decides
# how narrowly to aim.
_PAIRS_ARE_EVIDENCE_NOTE = (
    "\nThese pairs are evidence, not the exam: selection is decided on a fixed "
    "set of\npairs you cannot see, from rows outside this pool, and a change "
    "that only fixes\nthe pairs listed here and nothing general will not move it."
)


# The instruction template.
#
# Every section is forced by what the loop actually computes:
#   * the schema asks for misordered PAIRS, because `targets` names a pair to
#     reorder and there is no per-row counterexample to fix;
#   * there is one pair section rather than FN/FP sections, because
#     `_pair_misorder_examples` writes each inversion twice and rendering the
#     halves apart hides that they are one mistake;
#   * there are no curated trajectories, judge layer blame or regression
#     memory buckets, because the judge produces no per-row verdict and
#     `RegressionMemory` receives only boundary flags — all four sections
#     could only ever render "(none)";
#   * there is no RuleCritic report, for the same reason and a worse one: it
#     does not render "(none)", it renders a NUMBER measured against that
#     same absent per-row map. `precision` divides agreement by EVERY fire,
#     so a fire on a row the map does not cover counts as a wrong one, and a
#     rule with a large measured rank lift can read as "low-precision". The
#     honest measurement of the same rule is already here: `criterion_lift`
#     reports `per rule — <name>: <shift>` over the judged pairs, the
#     instrument criteria are measured with. The critic still runs and still
#     writes `v2_critics/rule_critic.json`;
#   * the two critic quality numbers are labelled DIAGNOSTIC, because their
#     weight in the objective is 0.
#
# `{selection_gate_note}` renders `_PAIRS_ARE_EVIDENCE_NOTE`.
PROPOSER_INSTRUCTIONS = """\
You will produce {n_candidates} candidate filter revisions.

Each candidate must be a complete rewrite of these three files:
  * rubric.md   (the semantic rubric criteria and anchors)
  * rules.py    (deterministic rule callables — pure Python)
  * policy.py   (how the continuous score is composed, and the bookkeeping
                 decision derived from it)

Rules for editing:
  * You may add, delete, split, merge, or rewrite rubric criteria.
  * You may add, delete, or rewrite deterministic rules.
  * You may edit how rubric scores and triggered rules compose into the
    score, and where the bookkeeping decision cut sits.
  * You are free to simplify redundant logic.
  * Every deterministic rule must be a top-level function named `rule_<name>`
    that takes a trajectory-like object and returns None or
    {{"name": str, "verdict": "ACCEPT"|"REJECT"|"REVIEW", "evidence": [str]}}.
  * `policy.py` must expose `decide(trajectory, rubric_scores, triggered_rules)`
    and what it returns must carry `"score": <float 0-100>` for every row.
  * The loop pays for scoring; you do not. Do not optimise for cache hits:
    a rubric edit costs you nothing and is evaluated exactly like a rules
    edit, by the precheck and then by the full pass. Unchanged criteria
    keep their cached cells and changed ones are scored fresh — that is
    bookkeeping the precheck reports per candidate, not a reason to leave a
    criterion you have a hypothesis about untouched.

{edit_discipline}Output format — respond with a single JSON object.

The `diagnosis` block is where you *report* what you inferred from the
misordered pairs below — for each row: its rubric scores, the rules it
triggered, its action histogram, its opening turns and the filter's own
reasoning — and from the two critic reports. Diagnoses should be short
factual observations about what the ordering gets wrong, not open-ended
internal chain-of-thought. Every candidate then declares which misordered
pairs it aims to reorder (`targets`) and a one-line per-file rationale
(`rationale`).

{{
  "diagnosis": [
    "<one factual observation per ordering failure the misordered pairs made visible>",
    ...
  ],
  "candidates": [
    {{
      "name": "<short slug>",
{hypothesis_field}      "notes": "<what this candidate does in one line>",
      "targets": ["<a misordered pair this candidate aims to reorder, written
                   as `<id the judge preferred> over <id this filter put first>`>", ...],
      "rationale": {{
        "rubric.md": "<one sentence — why this rubric change>",
        "rules.py":  "<one sentence — why this rules change>",
        "policy.py": "<one sentence — why this policy change>"
      }},
      "changes": {{
        "rubric.md": "<complete replacement>",
        "rules.py":  "<complete replacement>",
        "policy.py": "<complete replacement>"
      }}
    }}
    ...
  ]
}}

Do not emit anything outside the JSON object. The `diagnosis` and
`rationale` fields ARE where your reasoning goes — do not add extra
prose. If a file is unchanged, still copy its current text into `changes`
and note "no change" in the corresponding `rationale` entry.

META_FILTER::PROPOSER

<STATE_JSON>
{state_json}
</STATE_JSON>

### Current rubric.md
{rubric_md}

### Current rules.py
{rules_py}

### Current policy.py
{policy_py}

### Current-filter decision counts
Bookkeeping only — selection reads the score, not the decision. These are
the counts the filter under edit produced on this iteration's pool.
{metrics_block}

### Misordered pairs — the judge's order against this filter's order
Each block is ONE pair a judge adjudicated in both presentation orders and
this filter scored the other way round. Both trajectories are shown side by
side, with the evidence the filter scored them on.{selection_gate_note}

{pair_misorders}

### RubricCritic report (DIAGNOSTIC — weight 0 in the score)
The rubric prompt is what turns a general-purpose LLM into a SWE
reviewer. This report audits whether rubric.md names the right signals.
Nothing in it is charged to your score: read it as a second reading of your
criteria, not as a term to maximize.
{rubric_critic_block}

### Your prior-iteration diagnoses (self-continuity)
These are the diagnoses and candidate notes YOU emitted in earlier
iterations. Use them to keep your reasoning coherent across iterations:
avoid repeating diagnoses that were already tried and failed, and build
on the ones that led to score improvements.
{past_diagnoses_block}

### Score trajectory (last few iterations)
The selected filter's score and its pair concordance, iteration by
iteration. The two critic quality numbers carry weight 0 and are shown only
so that a wrong turn in the prose stays visible.
{quality_trajectory_block}

### Recent winning edits (what worked, per iteration)
For each recent iteration: which candidate won and why the selector
chose it. Use this to identify edit *directions* (not exact edits) that
are gaining traction.
{winners_block}

### Trajectory pool overview
{pool_overview}

### Iteration history (previous candidates and their outcomes)
{history_block}

### Pareto frontier so far (score, complexity)
You do not have to stay on it, but leaving it usually needs justification.
{pareto_block}
"""


@dataclass
class ProposerResult:
    prompt: str
    response: LLMResponse
    candidate_dirs: list[Path]
    parse_errors: list[str] = field(default_factory=list)
    # How many LLM legs the proposer spent this iteration. One for the
    # generation call; two once a revision or retry leg has run. Recorded in
    # the iteration record because the loop's contract is that a proposer
    # never gets a third — an unbounded revise/re-check cycle would let one
    # iteration spend an arbitrary amount on a candidate set nobody has
    # judged yet.
    legs: int = 1


@dataclass
class Proposer:
    """LLM-driven candidate generator."""

    client: LLMClient = field(default_factory=MockLLMClient)
    n_candidates: int = 3
    temperature: float = 0.4
    # Set by the Optimizer for agentic runs. See meta_filter.visibility.pseudonym.
    pseudonymizer: Pseudonymizer | None = None
    # Set by the Optimizer: the run's initial filter counts, which the
    # edit-discipline block prints the hard caps and floors against. None
    # falls back to the filter under edit.
    initial_complexity: dict[str, int] | None = None
    # Set by the Optimizer: how many candidates the precheck lets through to
    # a full scoring pass. None scores every candidate.
    candidates_scored: int | None = None
    # Set by the Optimizer: the ZERO LIFT finding, repeated in the revision
    # leg so a measured removal is not undone there.
    zero_lift_note: str = ""
    # Set by the Optimizer under `--freeze`: which of rubric / rules / policy
    # the precheck refuses a change to. Empty means nothing is frozen.
    frozen_files: tuple[str, ...] = ()
    # Set by the Optimizer when the tool-using proposer may score a criterion
    # it is drafting (`meta_filter.score_tool`): the command, its budget and
    # what the harness sandbox has to let through. None for the static
    # proposer, which has no shell to run it from.
    score_tool: ScoreToolAccess | None = None
    # Set by the Optimizer so the stated objective tracks the applied one.
    system_prompt: str = PROPOSER_SYSTEM
    # Whether this proposer explores a workspace with tools. The optimizer
    # reads it to decide whether the scoring tool can be advertised at all.
    agentic: ClassVar[bool] = False

    def propose(
        self,
        *,
        loaded_filter: LoadedFilter,
        candidates_root: str | Path,
        iteration: int,
        current_metrics: FilterMetrics,
        recent_false_negatives: list[CounterExample],
        recent_false_positives: list[CounterExample],
        history: list[CandidateEvaluation] | None = None,
        pareto_snapshot: list[CandidateEvaluation] | None = None,
        pool_overview: str | None = None,
        rubric_critic_report: dict | None = None,
        past_diagnoses: list[dict] | None = None,
        quality_trajectory: list[dict] | None = None,
        past_winners: list[dict] | None = None,
        loop_telemetry: dict | None = None,
        ranking_state: dict | None = None,
    ) -> ProposerResult:
        candidates_root = Path(candidates_root)
        candidates_root.mkdir(parents=True, exist_ok=True)

        prompt = self._build_prompt(
            loaded_filter=loaded_filter,
            iteration=iteration,
            current_metrics=current_metrics,
            recent_false_negatives=recent_false_negatives,
            recent_false_positives=recent_false_positives,
            history=history,
            pareto_snapshot=pareto_snapshot,
            pool_overview=pool_overview,
            rubric_critic_report=rubric_critic_report,
            past_diagnoses=past_diagnoses,
            quality_trajectory=quality_trajectory,
            past_winners=past_winners,
            loop_telemetry=loop_telemetry,
            ranking_state=ranking_state,
        )
        response = self.client.generate(
            prompt,
            system=self.system_prompt,
            temperature=self.temperature,
            cache_system=True,  # Proposer system prompt is a constant prefix.
        )

        proposals, diagnosis, parse_errors = _parse_candidates(response.text)
        candidate_dirs: list[Path] = []
        for i, cand in enumerate(proposals):
            slug = _slug(cand.get("name") or f"cand_{i:02d}")
            target = candidates_root / f"iter{iteration:03d}_{i:02d}_{slug}"
            try:
                materialize_candidate(
                    target,
                    rubric_md=cand["changes"]["rubric.md"],
                    rules_py=cand["changes"]["rules.py"],
                    policy_py=cand["changes"]["policy.py"],
                )
                load_filter(target)  # validates syntax + policy.decide
            # policy.py runs at import time, so a bad candidate can raise
            # anything at all. Reject it, never let it abort the run.
            except Exception as exc:
                parse_errors.append(f"candidate {slug}: {exc}")
                continue
            _write_rationale_md(target, cand, diagnosis)
            candidate_dirs.append(target)

        return ProposerResult(
            prompt=prompt,
            response=response,
            candidate_dirs=candidate_dirs,
            parse_errors=parse_errors,
        )

    def revise(
        self,
        *,
        result: ProposerResult,
        precheck_block: str,
        candidates_root: str | Path,
        iteration: int,
    ) -> ProposerResult:
        """The one revision leg, given each candidate's precheck readout.

        The base proposer answers from its prompt alone and writes candidates
        out of a JSON blob, so there is no workspace for it to revise in and
        nothing on disk for it to re-read. It declines the leg and says so by
        returning the result unchanged with `legs` untouched, which is what
        the iteration record then reports. `AgenticProposer` overrides this.
        """

        _ = precheck_block, candidates_root, iteration
        return result

    # -------------------------------------------------------------- helpers

    def _build_prompt(
        self,
        *,
        loaded_filter: LoadedFilter,
        iteration: int,
        current_metrics: FilterMetrics,
        recent_false_negatives: list[CounterExample],
        recent_false_positives: list[CounterExample],
        history: list[CandidateEvaluation] | None = None,
        pareto_snapshot: list[CandidateEvaluation] | None = None,
        pool_overview: str | None = None,
        rubric_critic_report: dict | None = None,
        past_diagnoses: list[dict] | None = None,
        quality_trajectory: list[dict] | None = None,
        past_winners: list[dict] | None = None,
        loop_telemetry: dict | None = None,
        ranking_state: dict | None = None,
    ) -> str:
        rubric_md = loaded_filter.rubric_md
        rules_py = (loaded_filter.directory / "rules.py").read_text(encoding="utf-8")
        policy_py = (loaded_filter.directory / "policy.py").read_text(encoding="utf-8")

        rs = ranking_state or {}
        # The pair terms the selector reads come from `ranking_state` rather
        # than `current_metrics`, which at proposer time carries only the
        # filter's decision counts.
        state = {
            "iteration": iteration,
            "n_candidates": self.n_candidates,
            "ranking_mode": True,
            "n_misordered_pairs": len(recent_false_negatives),
            "pair_concordance": rs.get("pair_concordance"),
            "n_truth_pairs": rs.get("n_truth_pairs", 0),
            "n_tied_pairs": rs.get("n_tied_pairs", 0),
            "redundancy_penalty": rs.get("redundancy_penalty", 0.0),
            "complexity_score": rs.get("complexity_score", current_metrics.complexity_score),
            "min_distinct_levels": rs.get("min_distinct_levels"),
            "rubric_md": rubric_md,
            "rules_py": rules_py,
            "policy_py": policy_py,
        }
        discipline = (
            render_edit_discipline(
                n_candidates=self.n_candidates,
                parent_counts=measure_filter_complexity(loaded_filter.directory)[0],
                initial_counts=self.initial_complexity,
                candidates_scored=self.candidates_scored,
                hypothesis_source="the candidate's `hypothesis` field in your JSON",
                frozen=tuple(self.frozen_files or ()),
                score_tool=self.score_tool is not None,
            )
            + "\n\n"
        )
        hypothesis_field = (
            '      "hypothesis": "<the single mechanism this candidate tests, in one sentence>",\n'
        )
        prompt = PROPOSER_INSTRUCTIONS.format(
            n_candidates=self.n_candidates,
            edit_discipline=discipline,
            hypothesis_field=hypothesis_field,
            selection_gate_note=_PAIRS_ARE_EVIDENCE_NOTE,
            state_json=json.dumps(state),
            rubric_md=rubric_md,
            rules_py=rules_py,
            policy_py=policy_py,
            metrics_block=_format_decision_counts(current_metrics),
            pair_misorders=_format_pair_misorders(
                recent_false_negatives,
                recent_false_positives,
                iteration=iteration,
                limit=5,
                turn_snippets=3,
            ),
            rubric_critic_block=_format_rubric_critic_block(rubric_critic_report),
            past_diagnoses_block=_format_past_diagnoses_block(past_diagnoses),
            quality_trajectory_block=_format_score_trajectory_block(quality_trajectory),
            winners_block=_format_past_winners_block(past_winners),
            history_block=_format_history(history, limit=4),
            pareto_block=_format_pareto_snapshot(pareto_snapshot or history),
            pool_overview=pool_overview or "(pool overview not provided)",
        )
        if loop_telemetry:
            prompt += _format_loop_telemetry(loop_telemetry)
        # Redacting the run directory is not enough on its own: the evidence
        # blocks above carry ids straight into the prompt.
        if self.pseudonymizer is not None:
            prompt = self.pseudonymizer.redact(prompt)
        return prompt


def _format_loop_telemetry(t: dict) -> str:
    """Deterministic stagnation signals, measured by the loop itself.

    The proposer is told to escalate change size when these show stagnation;
    an instruction keyed to signals it cannot see would be vibes. A rubric can
    survive a whole run byte-identical while most candidates edit it and
    every one loses selection, and nothing would say so out loud.
    """

    lines = ["\n## Loop telemetry (measured, not opinions)\n"]
    if (age := t.get("rubric_age")) is not None:
        lines.append(
            f"- rubric.md unchanged for {age} consecutive iteration(s)"
            + (" — the rubric has NEVER changed this run" if t.get("rubric_never_changed") else "")
        )
    if (wins := t.get("rubric_edit_wins")) is not None:
        lines.append(
            f"- candidates that edited rubric.md: {t.get('rubric_edit_candidates', '?')} so far, "
            f"{wins} won selection"
        )
    if nov := t.get("recent_novelty"):
        lines.append(
            "- proposal novelty (share of new lines) over recent iterations: "
            + ", ".join(f"{v:.2f}" for v in nov)
        )
    if lift := t.get("criterion_lift"):
        lines.append(f"- {lift}")
    if note := t.get("critic_note"):
        lines.append(f"- the reviewing critic's search-health read last iteration: {note}")
    if probe := t.get("probe_trend"):
        lines.append(
            "- fixed probe set (100 unlabeled trajectories, constant across "
            "iterations — trend of YOUR filter's own decisions, no ground truth):"
        )
        lines.append(probe)
    if rank := t.get("ranking_block"):
        lines.append(
            "- ranking telemetry (last winner — concordance is the objective's "
            "agreement term; the distribution is a DIAGNOSTIC, not a target):"
        )
        lines.append(rank)
    if health := t.get("criterion_health"):
        # The header describes what the block below it actually carries: the
        # unread clause appears only when the block renders an unread tag.
        lines.append(
            "- criterion health (last winner — measured over its own scores and "
            "read off its policy.py; a DEAD axis and a criterion DECLARED BUT "
            "UNREAD both waste every scorer call they cost, and two axes with "
            "|rho| above the redundancy threshold largely re-measure one "
            "thing, which distorts any aggregate that counts them twice):"
            if UNREAD_TAG_MARK in health
            else "- criterion health (last winner — measured over its own scores; "
            "a DEAD axis wastes every scorer call it costs, and two axes with "
            "|rho| above the redundancy threshold largely re-measure one "
            "thing, which distorts any aggregate that counts them twice):"
        )
        lines.append(health)
    if stability := t.get("criterion_stability"):
        lines.append(
            "- criterion stability (the last promoted filter, scored twice on the "
            "same hidden rows — a criterion whose rows move between two readings "
            "is anchored too loosely for the scorer to read the same way twice; "
            "this is a rubric defect the concordance numbers cannot point at):"
        )
        lines.append(stability)
    if signal := t.get("criterion_signal"):
        lines.append(
            "- criterion signal (each criterion's scores ALONE against the judged "
            "train pairs, beside its rank lift when ablated — an axis near 0.5 alone "
            "orders nothing; one high alone but low lift is redundant with another):"
        )
        lines.append(signal)
    if themes := t.get("rationale_themes"):
        lines.append(
            "- judge-rationale themes (what the judge keeps saying on the pairs the "
            "last winner gets wrong, grouped by theme, and whether any current "
            "criterion asks about it — an uncovered theme with many pairs is the "
            "rubric edit with the most evidence behind it):"
        )
        lines.append(themes)
    if pe := t.get("pair_evidence"):
        lines.append(
            "- pair evidence (judge-adjudicated pairs the selected filter got "
            "backwards — the FAILURE PATTERN is the signal, not these specific "
            "rows: the pair sample rotates every iteration, so fixing only "
            "these exact pairs buys nothing. Do NOT transplant the judge's "
            "phrasing into rubric text; the circularity critic checks for it):"
        )
        lines.append(pe)
    lines.append(
        "\nRead these before choosing how big to go. A stagnant signal here means "
        "small edits are being outcompeted by the incumbent, not that the incumbent "
        "is right."
    )
    return "\n".join(lines) + "\n"


def _action_histogram(trajectory) -> dict[str, int]:
    from collections import Counter

    c = Counter()
    for t in trajectory.turns:
        key = getattr(t, "action", None) or t.role
        c[key] += 1
    return dict(c.most_common(8))


def _row_evidence_lines(ce: CounterExample, turn_snippets: int) -> list[str]:
    """The evidence one trajectory carries, without any verdict framing.

    Used by the pair blocks, which show both rows of a pair. Kept verdict-free on
    purpose: the caller states what the disagreement was, so this function
    cannot smuggle an ACCEPT/REJECT reading into a run that has none.
    """

    traj = ce.raw_trajectory
    decision = ce.solver_decision
    info_lines: list[str] = []
    if traj is not None:
        info_lines.append(f"- action histogram (top): {_action_histogram(traj)}")
        info_lines.append(f"- task: {traj.task[:200]}")
        if traj.metadata:
            keep = {
                k: traj.metadata[k]
                for k in ("api_calls", "n_messages", "submission_bytes")
                if k in traj.metadata
            }
            if keep:
                info_lines.append(f"- metadata: {keep}")
        # A few representative turns to show what the solver actually saw.
        sampled = traj.turns[:turn_snippets]
        for idx, turn in enumerate(sampled):
            snippet = (turn.content or "").replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            info_lines.append(
                f"    [turn {idx:02d}] role={turn.role}"
                f" action={getattr(turn, 'action', None)}: {snippet}"
            )
    if decision is not None:
        info_lines.append(f"- rubric_scores: {decision.rubric_scores}")
        info_lines.append(f"- triggered_rules: {decision.triggered_rules}")
        info_lines.append(f"- solver_reasoning: {decision.reasoning_summary[:200]}")
    return info_lines


_GAP_RE = re.compile(r"inversion gap ([0-9]+(?:\.[0-9]+)?)")
_JUDGE_RE = re.compile(r'Judge: "(.*)"\s*$', re.DOTALL)


def _partner_id(ce: CounterExample) -> str:
    """The other half of the pair, from `pair_misorder_vs:<id>`."""

    prefix = "pair_misorder_vs:"
    subtype = ce.failure_subtype or ""
    return subtype[len(prefix) :] if subtype.startswith(prefix) else ""


# How many pairs the misordered-pair block renders. The optimizer reads it
# too: it decides which of a winner's records count as "already shown".
PROMPT_MISORDER_PAIRS = 5

# Hard character cap on the whole block. Five pairs typically take 15-25 KB,
# so this rarely binds — it exists so the per-pair history line cannot make
# the block grow without someone noticing. Pairs are dropped whole from the tail; the first is
# always kept, because a truncated block is still better than none.
MISORDER_BLOCK_BUDGET = 26_000


def _format_pair_misorders(
    over_ranked: list[CounterExample],
    under_ranked: list[CounterExample],
    *,
    iteration: int,
    limit: int = PROMPT_MISORDER_PAIRS,
    turn_snippets: int = 3,
    budget: int = MISORDER_BLOCK_BUDGET,
) -> str:
    """One block per pair the judge and filter disagree on.

    `_pair_misorder_examples` records every inversion twice — once for the row
    the filter pushed up, once for the row it pushed down — because an
    inversion is two mistakes at once. Rendered as two halves under "false
    negatives" and "false positives", five pairs would arrive as ten separate
    rows whose only link is an id buried in `subtype=pair_misorder_vs:`, and a
    proposer can read that as "no adjudicated mistakes exist".

    Here the pair is the unit. Both trajectories appear side by side, the
    judge's pick first, with the judge's own rationale above them.

    `pair_history`, when the record carries one, renders as one extra bullet
    under the header: which iteration bought the verdict, which stratum it
    came from, and whether an earlier prompt already showed this pair and the
    filter still inverts it. That last part is the point of keeping repeats —
    see `optimizer._select_misordered_pairs`.
    """

    if not over_ranked and not under_ranked:
        if iteration == 0:
            return (
                "no judged pairs yet — the first truth set is built after\n"
                "this iteration's candidates are scored, so there is nothing\n"
                "here to diagnose yet. Read the current filter, the pool\n"
                "overview and `reference/` instead."
            )
        return (
            "no misordered pairs this iteration — either the pair judge\n"
            "resolved nothing (see `ranking.json`) or the selected filter\n"
            "ordered every adjudicated pair the way the judge did."
        )

    by_id = {ce.trajectory_id: ce for ce in under_ranked}
    blocks: list[str] = []
    used = 0
    for i, high in enumerate(over_ranked[:limit]):
        preferred_id = _partner_id(high)
        low = by_id.get(preferred_id)
        gap = _GAP_RE.search(high.why_filter_failed or "")
        judge = _JUDGE_RE.search(high.description or "")
        lines = [
            f"### Misordered pair {i:02d} — the judge preferred {preferred_id} "
            f"over {high.trajectory_id}; this filter scored "
            f"{high.trajectory_id} above {preferred_id}" + (f" (gap {gap.group(1)})" if gap else "")
        ]
        if high.pair_history:
            lines.append(f"- {high.pair_history}")
        if judge:
            lines.append(f'- the judge\'s rationale: "{judge.group(1).strip()}"')
        if high.criterion_breakdown:
            lines.append(f"- {high.criterion_breakdown}")
        lines.append(
            f"- {preferred_id} — the judge's pick, which this filter ranked BELOW"
            + (
                f" (the filter's own confidence on it: {low.confidence:.2f})"
                if low is not None
                else " (its evidence is not in this prompt)"
            )
        )
        if low is not None:
            lines.extend(f"  {line}" for line in _row_evidence_lines(low, turn_snippets))
        lines.append(
            f"- {high.trajectory_id} — which this filter ranked ABOVE "
            f"(the filter's own confidence on it: {high.confidence:.2f})"
        )
        lines.extend(f"  {line}" for line in _row_evidence_lines(high, turn_snippets))
        block = "\n".join(lines)
        # +2 for the blank line that joins this block to the previous one.
        if blocks and used + len(block) + 2 > budget:
            break
        used += len(block) + (2 if blocks else 0)
        blocks.append(block)
    return "\n\n".join(blocks)


def _format_decision_counts(metrics: FilterMetrics) -> str:
    """What the decision field did, and nothing else.

    `FilterMetrics.model_dump()` carries per-row verdict fields: `review_rate`,
    `loss`, `critic_reject_recall` and the four filter-vs-critic cells are
    either uncomputed or derived from candidate consensus rather than from
    the judge — and at proposer time every one of them is a stub zero
    (`_stub_metrics`). Printing them would tell the proposer the loop is
    grading a gate.
    """

    return json.dumps(
        {
            "total": metrics.total,
            "accepted": metrics.accepted,
            "rejected": metrics.rejected,
            "reviewed": metrics.reviewed,
        },
        indent=2,
    )


def _format_history(history: list[CandidateEvaluation] | None, limit: int = 4) -> str:
    """Compact per-iteration history so the proposer avoids repeating past mistakes.

    The FN/FP/review columns are dropped. They are counted against candidate
    consensus, not against the pair judge, and the loop refuses to publish
    that comparison as judge evidence; the concordance column is what these
    candidates were measured on. `critic_agreement` holds it — the optimizer
    writes the pair concordance into that field (`optimizer.py`, the ranking
    scoring block).
    """

    if not history:
        return "(no prior iterations)"
    lines = []
    for i, ev in enumerate(history[-limit:]):
        m = ev.metrics
        lines.append(
            f"- iter {i}: candidate={ev.candidate_id} baseline={ev.is_baseline}"
            f" score={m.score:.3f} pair_concordance={m.critic_agreement:.3f}"
            f" complexity={m.complexity_score:.2f}"
        )
    return "\n".join(lines)


def _format_pareto_snapshot(evaluations: list[CandidateEvaluation] | None) -> str:
    """Non-dominated (score, complexity) points across every past candidate."""

    if not evaluations:
        return "(no candidates evaluated yet)"
    pts = [
        (
            e.candidate_id,
            e.metrics.score,
            e.metrics.complexity_score,
            e.metrics.estimated_filter_cost,
        )
        for e in evaluations
        if e.invalid_reason is None
    ]
    if not pts:
        return "(no valid candidates)"
    # Envelope: sort by complexity asc, then keep the running max score.
    pts.sort(key=lambda p: (p[2], -p[1]))
    frontier = []
    best_score = float("-inf")
    for cid, score, cx, cost in pts:
        if score > best_score:
            frontier.append((cid, score, cx, cost))
            best_score = score
    lines = []
    for cid, score, cx, cost in frontier:
        lines.append(f"- {cid}: score={score:.3f}  complexity={cx:.2f}  cost={cost:.2f}")
    return "\n".join(lines)


def _format_rubric_critic_block(report: dict | None) -> str:
    """The rubric-critic section rendered for the proposer.

    None → "(not available yet)" so iter 0 (before critics have run) does
    not blow up the prompt template.
    """

    if not report:
        return "(not available yet — first iteration or critic disabled)"
    lines = [f"rubric_quality: {report.get('rubric_quality', 0.5):.3f}"]
    issues = report.get("issues") or []
    if issues:
        lines.append("Issues flagged:")
        for issue in issues:
            criterion = issue.get("criterion", "")
            problem = issue.get("problem", "")
            edit = issue.get("suggested_edit", "")
            lines.append(f"  - [{criterion}] {problem}")
            if edit:
                lines.append(f"    suggested edit: {edit}")
    missing = report.get("missing_criteria") or []
    if missing:
        lines.append("Missing criteria (rubric should name these but doesn't):")
        for m in missing:
            lines.append(f"  - {m}")
    return "\n".join(lines) if lines else "(no findings)"


def _format_past_diagnoses_block(items: list[dict] | None) -> str:
    """Self-continuity — the proposer's own past diagnosis lines and
    per-candidate notes, one iter per bullet cluster."""

    if not items:
        return "(no prior iterations)"
    lines = []
    for entry in items:
        k = entry.get("iteration")
        diag = entry.get("diagnosis") or []
        notes = entry.get("candidate_notes") or []
        lines.append(f"iter {k}:")
        if diag:
            for d in diag[:4]:
                lines.append(f"  · diagnosis: {str(d)[:200]}")
        if notes:
            for n in notes[:4]:
                lines.append(f"  · candidate note: {str(n)[:200]}")
    return "\n".join(lines) if lines else "(no prior iterations)"


def _format_score_trajectory_block(items: list[dict] | None) -> str:
    """The score trajectory: one row per past iteration.

    Two choices, both forced by the objective. There is no `review_rate`:
    every row is scored and the decision is bookkeeping, so the column would
    measure nothing. And the agreement column is named for what it holds —
    the optimizer writes pair concordance into `critic_agreement` — with the
    two critic quality numbers kept but marked as the weight-0 diagnostics
    they are.
    """

    if not items:
        return "(no history yet)"
    header = f"{'iter':>4}  {'score':>7}  {'concordance':>11}  {'rubric_q*':>9}  {'rule_q*':>7}"
    rows = [header, "-" * len(header)]
    for row in items:
        rows.append(
            "{iter:>4}  {score:>7}  {conc:>11}  {rq:>9}  {qq:>7}".format(
                iter=row.get("iteration", "?"),
                score=_fmt3(row.get("score")),
                conc=_fmt3(row.get("critic_agreement")),
                rq=_fmt3(row.get("rubric_quality")),
                qq=_fmt3(row.get("rule_quality")),
            )
        )
    rows.append("* diagnostic only — weight 0 in the score")
    return "\n".join(rows)


_LOSS_CLAUSE_RE = re.compile(r"\s*\(loss [^)]*\)|,\s*loss=[-0-9.]+")


def _ranking_selection_reason(reason: str) -> str:
    """The part of a stored selector reason that describes this loop.

    A reason string of the form "... score 0.715 > baseline 0.620 (loss
    26.076 vs 2.280). FN=0, FP=0, review=50, complexity=21.51, cost=0.00"
    carries a `loss` that is not the objective and FN/FP/review counts from
    candidate consensus, not from the pair judge, so those four numbers
    describe nothing the winner was chosen on. The score comparison, the
    complexity and the cost are kept verbatim; the rest is dropped rather
    than reworded.
    """

    text = _LOSS_CLAUSE_RE.sub("", reason)
    head, sep, tail = text.partition("FN=")
    if sep:
        kept = [field for field in tail.split(", ") if field.startswith(("complexity=", "cost="))]
        text = head + ", ".join(kept)
    return " ".join(text.split())


def _format_past_winners_block(items: list[dict] | None) -> str:
    if not items:
        return "(no winners yet)"
    lines = []
    for entry in items:
        k = entry.get("iteration")
        cand = entry.get("winner")
        reason = _ranking_selection_reason(entry.get("reason") or "")[:280]
        notes = entry.get("winner_notes") or ""
        lines.append(f"iter {k} winner: {cand}")
        if notes:
            lines.append(f"  edit notes: {notes[:200]}")
        if reason:
            lines.append(f"  selector reason: {reason}")
    return "\n".join(lines) if lines else "(no winners yet)"


def _fmt3(v) -> str:
    try:
        return f"{float(v):.3f}" if v not in (None, "") else "—"
    except (TypeError, ValueError):
        return "—"


def _slug(value: str) -> str:
    keep = []
    for ch in value.lower().strip():
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "_", "-"):
            keep.append("_")
    slug = "".join(keep).strip("_") or "candidate"
    return slug[:32]


def _parse_candidates(text: str) -> tuple[list[dict], list[str], list[str]]:
    """Parse the JSON response. Returns (candidates, diagnosis, parse_errors)."""

    text = text.strip()
    if not text:
        return [], [], ["empty response"]
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return [], [], [f"JSON decode error: {exc.msg}"]
        else:
            return [], [], [f"JSON decode error: {exc.msg}"]

    diagnosis_raw = obj.get("diagnosis", []) or []
    diagnosis = [str(d) for d in diagnosis_raw if isinstance(d, str)]

    proposals = obj.get("candidates", [])
    errors: list[str] = []
    kept: list[dict] = []
    for i, cand in enumerate(proposals):
        if not isinstance(cand, dict):
            errors.append(f"candidate {i}: not a dict")
            continue
        changes = cand.get("changes")
        if not isinstance(changes, dict):
            errors.append(f"candidate {i}: missing 'changes' object")
            continue
        for required in ("rubric.md", "rules.py", "policy.py"):
            if required not in changes or not isinstance(changes[required], str):
                errors.append(f"candidate {i}: missing complete replacement for {required}")
                break
        else:
            kept.append(cand)
    return kept, diagnosis, errors


_HYPOTHESIS_PREFIX = "Hypothesis:"


def _preserved_hypotheses(candidate_dir: Path) -> list[str]:
    """Every `Hypothesis:` line of a rationale.md the proposer wrote itself.

    The tool-using proposer may author `rationale.md` directly and the loop
    then rewrites the file from `diagnosis.json`. EVERY hypothesis line it
    wrote must survive that rewrite: keeping only the first would collapse a
    candidate that declared two mechanisms into one that declared one, and
    the one-mechanism check (`constraints.count_mechanisms`) — which is the
    reason the line exists — could never fire.
    """

    path = candidate_dir / "rationale.md"
    if not path.exists():
        return []
    return [h for h in hypothesis_lines(path.read_text(encoding="utf-8", errors="replace")) if h]


def _declared_hypotheses(field_value: object) -> list[str]:
    """The hypotheses in a proposer's `hypothesis` field, one per marker.

    Usually one sentence. A field that itself contains more than one
    `Hypothesis:` marker ("Hypothesis: A. Hypothesis: B") is two declared
    mechanisms and is written as two lines, so the check sees both.
    """

    text = " ".join(str(field_value or "").split())
    if not text:
        return []
    parts = re.split(r"\*{0,2}hypothesis\*{0,2}\s*:\s*", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [text]


def _rationale_bullet(text: object) -> str:
    """A `- ` bullet that can never read as a `Hypothesis:` line.

    Diagnosis entries and targets are free text from the proposer; one that
    happens to start with "Hypothesis:" would be counted as a declared
    mechanism and reject the candidate on formatting alone.
    """

    body = " ".join(str(text).split())
    if re.match(r"^\*{0,2}hypothesis\*{0,2}\s*:", body, flags=re.IGNORECASE):
        return f"- note: {body}"
    return f"- {body}"


def _write_rationale_md(candidate_dir: Path, cand: dict, diagnosis: list[str]) -> None:
    """Persist proposer reasoning next to the candidate files for later audit.

    The `Hypothesis:` lines are what the one-mechanism check
    (`constraints.count_mechanisms`) counts, so this writer is faithful
    rather than normalising: every line the proposer wrote into
    `rationale.md` itself is kept, in order; the proposer's `hypothesis`
    field is written only when the file declared none (a field beside a
    written line would double-count one idea); and a field that carries two
    markers is written as two lines. A candidate that stated nothing gets no
    line, which the check reports as unknown rather than as a violation. No
    other line — diagnosis, targets, notes — may begin with the marker.
    """

    lines: list[str] = [f"# {cand.get('name', candidate_dir.name)}"]
    hypotheses = _preserved_hypotheses(candidate_dir) or _declared_hypotheses(
        cand.get("hypothesis")
    )
    for hypothesis in hypotheses:
        lines.append(f"\n{_HYPOTHESIS_PREFIX} {' '.join(hypothesis.split())}")
    if cand.get("notes"):
        lines.append(f"\n**Notes:** {cand['notes']}")
    if diagnosis:
        lines.append("\n## Diagnosis (proposer-reported)")
        for d in diagnosis:
            lines.append(_rationale_bullet(d))
    targets = cand.get("targets") or []
    if targets:
        lines.append("\n## Targets (counterexamples this candidate is meant to fix)")
        for t in targets:
            lines.append(_rationale_bullet(t))
    rationale = cand.get("rationale") or {}
    if isinstance(rationale, dict) and rationale:
        lines.append("\n## Per-file rationale")
        for fname in ("rubric.md", "rules.py", "policy.py"):
            r = rationale.get(fname)
            if r:
                lines.append(f"- **{fname}**: {r}")
    (candidate_dir / "rationale.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
