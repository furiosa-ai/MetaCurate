"""What a candidate would score, computed before the loop pays for it.

The proposer writes candidates blind. It reads the run tree, forms a
hypothesis, edits three files, and then finds out how the edit did one whole
iteration later — by which time the loop has already spent a full solver pass
and a round of judging on it, and the proposer that wrote it is a different
call with a different context. Nothing in that arrangement lets a proposer
notice that its own candidate ranks the pool worse than the filter it started
from.

This module closes that gap with the one measurement that is nearly free:
the truth pairs are already judged (they were bought in earlier iterations
and accumulate — see `Optimizer._accumulate_eval_pairs`), and the scorer
cache already holds a cell for every (row, criterion) the parent used. So a
candidate that edits two criteria out of six needs LLM work for two criteria
on the pair rows and nothing else; the rest of its concordance comes out of
the cache, and its rules and policy are deterministic code that re-runs for
free.

What this is NOT, said plainly because the proposer is told the same thing:

* No new judge verdicts. The pairs are the ones already adjudicated; the
  judge is never called here, and a candidate cannot buy itself supervision.
* Not the final number. It is measured over the accumulated pair set as it
  stands *before* this iteration's acquisition round, over the rows those
  pairs name rather than the whole training pool. The real score is measured
  after this iteration's judging, on the same formula.
* Not a new objective. `score` here is `concordance − λ_cx · min(1,
  complexity / budget)`, evaluated through the same `compute_score` the
  train-side score uses. It is evidence, not the acceptance decision, which
  the paired gate on the hidden selection pairs makes. The other number is
  carried beside it as `score_with_redundancy`, which is what the selected
  filter is finally reported with.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from meta_filter.core.filter_loader import LoadedFilter, load_filter
from meta_filter.core.llm_scorer import (
    _rubric_header,
    _split_rubric_sections,
    criterion_field_sets,
)
from meta_filter.core.schemas import FilterMetrics, Trajectory
from meta_filter.core.solver import Solver
from meta_filter.diagnostics.criterion_stats import (
    charge_redundancy,
    criterion_stats,
    redundancy_penalty,
)
from meta_filter.gate import ranking as ranking_math
from meta_filter.gate.complexity import measure_filter_complexity
from meta_filter.gate.constraints import (
    ConstraintLimits,
    ConstraintReport,
    assess_rubric_health,
    check_candidate,
    record_criteria_defect,
    record_rule_firing,
    rubric_criteria_defect_of,
)
from meta_filter.gate.metrics import DEFAULT_SCORE_WEIGHTS, compute_score
from meta_filter.gate.paired_test import pair_correctness
from meta_filter.llm.cost_meter import LEDGER as COST_LEDGER
from meta_filter.llm.cost_meter import CostLedger, RoleCost
from meta_filter.visibility.view_policy import (
    absent_field_signature,
    field_set_signature,
    scorer_key_fields,
)

logger = logging.getLogger(__name__)

# Terms of `compute_score` the precheck cannot honestly compute. `review_rate`
# would be measured over the pair rows rather than the pool, and
# `estimated_filter_cost` measures cache state across a full solver pass. Both
# are zero-weight by default (`metrics.DEFAULT_SCORE_WEIGHTS`); a run that
# turns one back on gets it named in `omitted_terms` rather than silently
# folded in at zero.
_UNCOMPUTABLE_WEIGHTS = {
    "lambda_review": "review_rate",
    "lambda_cost": "estimated_filter_cost",
    "lambda_rubric_quality": "rubric_quality",
    "lambda_rule_quality": "rule_quality",
}


@dataclass(frozen=True)
class CriterionDelta:
    """Which of a candidate's criteria are new work for the scorer.

    A criterion is UNCHANGED only when everything the scorer cache keys on
    that comes from the filter is byte-identical to the parent's: the
    criterion name, its own rubric section, the rubric preamble, and the
    metadata field set the section declares. That is the whole
    filter-dependent half of `llm_scorer._criterion_cache_key`; the other
    half is the trajectory, the model and the render view, none of which a
    candidate can change.

    The preamble is in there on purpose and it is the expensive part. Editing
    the shared header changes the prompt every criterion is scored under, so
    an unchanged criterion under a rewritten preamble is genuinely a
    different measurement and genuinely needs re-scoring. Reporting it as
    "unchanged" would make the precheck's cost estimate a lie in exactly the
    case where the proposer most needs to know the price.
    """

    changed: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    header_changed: bool = False

    def to_dict(self) -> dict:
        return {
            "changed": list(self.changed),
            "unchanged": list(self.unchanged),
            "added": list(self.added),
            "removed": list(self.removed),
            "header_changed": self.header_changed,
        }


def criterion_identities(
    rubric_md: str, criteria: Sequence[str]
) -> dict[str, tuple[str, str, str]]:
    """`criterion -> (preamble, own section, declared field signature)`.

    The tuple is exactly what the per-criterion cache key hashes out of the
    filter. Two filters agreeing on a criterion's tuple produce the same key
    for every trajectory, so the cell is a hit.
    """

    criteria = list(criteria)
    sections = _split_rubric_sections(rubric_md, criteria)
    header = _rubric_header(rubric_md, sections)
    field_sets = criterion_field_sets(rubric_md, criteria)
    return {
        c: (
            header,
            sections[c],
            field_set_signature("scorer", field_sets[c]),
        )
        for c in criteria
    }


def changed_criteria(
    parent_rubric: str,
    parent_criteria: Sequence[str],
    candidate_rubric: str,
    candidate_criteria: Sequence[str],
) -> CriterionDelta:
    """Partition the candidate's criteria by whether the parent already paid.

    `added` is the subset of `changed` whose name does not appear in the
    parent at all, and `removed` names parent criteria the candidate dropped.
    Both are reported separately because they read very differently to a
    proposer: an added criterion is the cost of a new hypothesis, a removed
    one is a rubric getting smaller, and a *renamed* criterion shows up as
    one of each.
    """

    parent = criterion_identities(parent_rubric, parent_criteria)
    candidate = criterion_identities(candidate_rubric, candidate_criteria)

    changed, unchanged, added = [], [], []
    for name, identity in candidate.items():
        if name not in parent:
            added.append(name)
            changed.append(name)
        elif parent[name] == identity:
            unchanged.append(name)
        else:
            changed.append(name)

    parent_header = next(iter(parent.values()))[0] if parent else ""
    cand_header = next(iter(candidate.values()))[0] if candidate else ""
    return CriterionDelta(
        changed=tuple(changed),
        unchanged=tuple(unchanged),
        added=tuple(added),
        removed=tuple(n for n in parent if n not in candidate),
        header_changed=bool(parent) and bool(candidate) and parent_header != cand_header,
    )


def cache_census(
    scorer,
    trajectories: Sequence[Trajectory],
    criteria: Sequence[str],
    rubric_md: str,
) -> tuple[int, int]:
    """`(cells already cached, cells that need a call)` for this rubric.

    Probes the scorer's own cache rather than inferring from
    `changed_criteria`, because the two answer different questions. The delta
    says what changed since the PARENT; the cache says what this run has ever
    paid for, which also covers a criterion an earlier iteration scored and a
    later one restored. Returns `(0, 0)` for a scorer with no per-criterion
    cache — the heuristic scorer makes no calls, so there is nothing to count.
    """

    read = getattr(scorer, "_read_criterion_entry", None)
    if not callable(read):
        return 0, 0

    criteria = list(criteria)
    sections = _split_rubric_sections(rubric_md, criteria)
    header = _rubric_header(rubric_md, sections)
    field_sets = criterion_field_sets(rubric_md, criteria)
    model = scorer._model_name() if hasattr(scorer, "_model_name") else ""

    cached = missing = 0
    for traj in trajectories:
        for criterion in criteria:
            # Same two components the scorer keys on, including the marker for
            # declared fields THIS row does not carry — a census that predicted
            # hits without it would promise cells that are about to be bought.
            sig = field_set_signature("scorer", field_sets[criterion]) + absent_field_signature(
                scorer_key_fields(field_sets[criterion]), traj.metadata
            )
            if read(traj, criterion, sections[criterion], model, sig, header) is None:
                missing += 1
            else:
                cached += 1
    return cached, missing


@dataclass(frozen=True)
class RuleFiring:
    """How often one `rule_` function fired over the precheck's rows.

    `n_rows` travels with `fires` on purpose. The precheck's rows are the
    rows the accumulated truth pairs name, not the training pool, so `3` on
    its own is unreadable and `3 of 41` is the only honest form — every
    sentence this module renders carries the denominator for that reason.
    """

    name: str
    fires: int
    n_rows: int
    raised: int = 0
    error_types: tuple[str, ...] = ()
    # False when the parent filter already defined a rule of this name. An
    # inherited rule is measured and reported exactly like a new one, but it
    # never escalates: the candidate did not write it.
    inherited: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fires": self.fires,
            "n_rows": self.n_rows,
            "raised": self.raised,
            "error_types": list(self.error_types),
            "inherited": self.inherited,
        }


@dataclass
class RuleFiringReport:
    """What the candidate's own `rules.py` does, measured rather than asked.

    The proposer has a tool that answers this (`meta_filter.rule_tool`) and
    measurably does not call it: zero invocations across two probe
    iterations, because its deliverable is candidate directories and the
    prompt promises the loop will report back. That promise holds for the
    rubric and not for the rules — a rule firing zero times and a rule firing
    and not helping produce the same concordance, the same redundancy and the
    same score. So the loop measures it here instead of asking.

    `rule_check` is the other half: when the proposer does claim it ran the
    checker, `diagnosis.json` carries the counts it says it saw, and those
    are compared against these.
    """

    n_rows: int = 0
    rules: list[RuleFiring] = field(default_factory=list)
    # Whether this candidate's `rules.py` differs from the parent's at all.
    # Nothing about `rule_check` is asked of a candidate that did not touch it.
    rules_changed: bool = False
    # `absent`, `unparseable`, `verified`, `mismatch`, or `not_required` when
    # nothing could be asked. `unparseable` is treated exactly as `absent`.
    rule_check_status: str = "not_required"
    rule_check_raw: str = ""
    # Rule names whose claimed count matched the measured one. Reported even
    # though nothing fires on it, so the signal is visible when it works.
    rule_check_verified: list[str] = field(default_factory=list)
    # Why the rules could not be run at all (an unimportable `rules.py`).
    # The candidate is already `invalid` for that via `load_filter`, so this
    # is a note rather than a second firing.
    error: str = ""
    # One sentence per escalation, in the register `constraints.py` uses.
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "rules": [rule.to_dict() for rule in self.rules],
            "rules_changed": self.rules_changed,
            "rule_check_status": self.rule_check_status,
            "rule_check_raw": self.rule_check_raw,
            "rule_check_verified": list(self.rule_check_verified),
            "error": self.error,
            "reasons": list(self.reasons),
        }


@dataclass
class PrecheckResult:
    """One candidate's pre-scoring readout. Serialized as `precheck.json`."""

    candidate_id: str
    concordance: float = 0.0
    n_pairs: int = 0
    n_tied: int = 0
    redundancy_penalty: float = 0.0
    complexity: float = 0.0
    # `score` leaves the redundancy penalty out; `score_with_redundancy` has
    # it taken off.
    score: float = 0.0
    score_with_redundancy: float = 0.0
    cells_rescored: int = 0
    cells_reused: int = 0
    cost_usd: float = 0.0
    misordered_pairs: list[dict] = field(default_factory=list)
    # Diagnostics that explain the three numbers above.
    n_rows: int = 0
    unscored_pairs: int = 0
    unscored_rows: int = 0
    distinct_levels: int = 0
    criterion_delta: CriterionDelta = field(default_factory=CriterionDelta)
    complexity_breakdown: dict = field(default_factory=dict)
    llm_calls: int = 0
    llm_tokens: int = 0
    cost_source: str = "unpriced"
    omitted_terms: list[str] = field(default_factory=list)
    error: str = ""
    # The growth / deletion / one-mechanism check (`constraints.py`). None
    # only when a result is built outside `run_precheck`.
    constraint: ConstraintReport | None = None
    # Per-pair credit on the judged pairs (`paired_test.pair_correctness`):
    # 1.0 ordered as the judge did, 0.5 own tie, 0.0 reversed or unscored.
    # This is what `concordance` averages; kept whole so the gate can tell
    # two candidates that behave identically on every judged pair apart from
    # two that merely score the same (duplicate collapse).
    # Not serialized — `misordered_pairs` already names the losses — beyond
    # the hash `to_dict` writes.
    pair_credit: dict[tuple[str, str], float] = field(default_factory=dict)
    # The candidate's own deadness / redundancy readout over the judged
    # rows (`criterion_stats.criterion_stats`), the same numbers the
    # redundancy penalty above is computed from. Kept whole so a
    # candidate that introduces a dead axis or a redundant pair can be told
    # so in the revision leg, against the parent's readout when the last
    # full pass left none.
    criterion_stats: dict | None = None
    # Why this candidate's `RUBRIC_CRITERIA` cannot be scored, read off the
    # imported module (`constraints.rubric_criteria_defect_of`) rather than
    # the source. Empty for a healthy candidate. `run_precheck` folds it into
    # `constraint` as an `invalid` firing, which is what makes the gate drop
    # the candidate instead of the solver pass silently losing it.
    criteria_defect: str = ""
    # What this candidate's `rules.py` actually fires on, over the precheck's
    # rows (`measure_candidate_rules`). Free — rules are pure Python — and
    # invisible in every other number here, since a rule that never fires and
    # a rule that fires uselessly produce the same concordance, the same
    # redundancy and the same score. `run_precheck` folds its `reasons` into
    # `constraint` as `needs_strong_evidence`, the way `assess_rubric_health`
    # folds in what scoring revealed. None when the measurement was not run.
    rule_firing: RuleFiringReport | None = None

    def correctness_key(self) -> tuple | None:
        """A hashable form of `pair_credit`; None when nothing was measured.

        Two candidates with equal keys ordered every judged pair the same way
        (including the same own ties and the same unscored rows), so on the
        evidence the precheck has they are one behaviour.
        """

        if self.error or not self.pair_credit:
            return None
        return tuple(sorted(self.pair_credit.items()))

    def correctness_hash(self) -> str | None:
        key = self.correctness_key()
        if key is None:
            return None
        return hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "correctness_hash": self.correctness_hash(),
            "concordance": self.concordance,
            "n_pairs": self.n_pairs,
            "n_tied": self.n_tied,
            "redundancy_penalty": self.redundancy_penalty,
            "complexity": self.complexity,
            "score": self.score,
            "score_with_redundancy": self.score_with_redundancy,
            "cells_rescored": self.cells_rescored,
            "cells_reused": self.cells_reused,
            "cost_usd": self.cost_usd,
            "misordered_pairs": self.misordered_pairs,
            "n_rows": self.n_rows,
            "unscored_pairs": self.unscored_pairs,
            "unscored_rows": self.unscored_rows,
            "distinct_levels": self.distinct_levels,
            "criterion_delta": self.criterion_delta.to_dict(),
            "complexity_breakdown": self.complexity_breakdown,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "cost_source": self.cost_source,
            "omitted_terms": self.omitted_terms,
            "error": self.error,
            "criteria_defect": self.criteria_defect,
            "rule_firing": self.rule_firing.to_dict() if self.rule_firing is not None else None,
            "constraint": self.constraint.to_dict() if self.constraint is not None else None,
            "criterion_stats": self.criterion_stats,
        }


def _metrics_shell(concordance: float, complexity_score: float, breakdown: dict) -> FilterMetrics:
    """A `FilterMetrics` carrying only the terms the precheck can measure.

    `compute_score` is called on this rather than reimplemented so the
    precheck cannot drift from selection: if a weight moves, both move. The
    counting fields are zero because there is no per-row adjudication here at
    all — the loop's supervision is pairwise, and every term those fields
    feed is zero-weight (and named in `omitted_terms` when it is not).
    """

    return FilterMetrics(
        total=0,
        critic_accept_total=0,
        critic_reject_total=0,
        accepted=0,
        rejected=0,
        reviewed=0,
        agree_accept=0,
        agree_reject=0,
        filter_accept_critic_reject=0,
        filter_reject_critic_accept=0,
        critic_reject_recall=0.0,
        critic_accept_retention=0.0,
        disputed_accept_rate=0.0,
        disputed_reject_rate=0.0,
        review_rate=0.0,
        critic_agreement=concordance,
        complexity=dict(breakdown),
        complexity_score=complexity_score,
    )


def _misorder_records(
    scores: dict[str, float],
    pairs: Sequence[tuple[str, str]],
    limit: int,
) -> list[dict]:
    """The pairs this candidate does not order the way the judge did.

    `hi` / `lo` are the JUDGE's order, so `direction` reads as what the
    candidate did to it. Three things can go wrong and they need different
    fixes, so they are named rather than merged: `inverted` is a real
    ordering mistake, `tied` is a scorer that cannot separate two rows the
    judge could, and `unscored` is a ranking-contract violation that costs
    the pair outright under `unscored="loss"`.
    """

    out: list[dict] = []
    for hi, lo in pairs:
        hi_score = scores.get(hi)
        lo_score = scores.get(lo)
        if hi_score is None or lo_score is None:
            direction = "unscored"
        elif hi_score > lo_score:
            continue
        elif hi_score == lo_score:
            direction = "tied"
        else:
            direction = "inverted"
        out.append(
            {
                "hi": hi,
                "lo": lo,
                "hi_score": hi_score,
                "lo_score": lo_score,
                "direction": direction,
            }
        )
        if len(out) >= limit:
            break
    return out


# --- what the candidate's rules actually fire on ----------------------------

# How a proposer reports a rule's firing count, which is not one form but a
# family of them. A parser that wants the count immediately after the name
# cannot read a large share of real `rule_check` strings, almost all of them
# correct measurements phrased as sentences.
#
#   rule_broad_sprawl fired 1/20 and ordered 3/3 split judged pairs correctly
#   rule_repeated_editing fired on 3 of 20 sampled train rows
#   rule_repeated_editing unchanged; current telemetry measured 13/50
#
# So the name and its count are matched separately, with filler allowed
# between them. The filler may not contain a digit, which is what keeps the
# count bound to the name rather than to the next number in the sentence: in
# the first example above the rule fired 1 of 20 times and ordered 3 of 3
# pairs, and only the digit rule tells those apart.
_RULE_NAME = re.compile(r"\b(rule_\w+)")
# `3/20` or `3 of 20`; the denominator is captured and then ignored, because
# the proposer samples its own rows through `rule_tool` and its `n` is not the
# precheck's `n`.
_COUNT = re.compile(r"\b(\d+)\s*(?:/|\s+of\s+)\s*(\d+)\b")
# How far past the name the count may sit. Long enough for "fired on" and for
# "unchanged; current telemetry measured", short enough that a count belonging
# to a later clause is not swept up.
_FILLER_LIMIT = 44

# A candidate that keeps no rules has nothing to count, so demanding a `k/n`
# from it would demand a fabrication. It says so instead, and the claim is
# checked against the measurement like any other. Every phrasing the three
# runs produced is here: "none", "No deterministic rules", "No rule_*
# functions", "no deterministic rule is defined", and the report that
# `rule_tool` refused because there was nothing to check.
_NO_RULES_CLAIM = re.compile(
    # `\brule` without a trailing boundary on purpose: the runs wrote "No
    # rule_* functions", and `rule_*` has no word boundary after "rule".
    r"\bnone\b|\bno\b[^.;]{0,40}?\brule|\bnothing to (?:check|replay)\b",
    re.IGNORECASE,
)

_NO_RULE_CHECK_REASON = (
    "rules.py changed without a rule_check in diagnosis.json — "
    "the firing counts were never looked at"
)


def parse_rule_check(raw: str, known_rules: Collection[str] = ()) -> dict[str, int] | None:
    """`{rule name: claimed firing count}` from a `rule_check` string.

    Args:
        raw: The free-text field, e.g. `"rule_trivial 1/20, rule_repeat 2/20"`.
            Counts are read wherever they appear, so a claim that carries its
            own supporting detail parses as the claim it is.
        known_rules: The rules this candidate actually defines. Used for one
            thing only: a claim that reports a count without naming anything
            is attributable when there is exactly one rule it could be about.

    Returns:
        The claimed counts; an empty dict when the text declares this
        candidate has no rules to count, which is a claim and is checked; or
        None when no claim can be read out of the text at all. An empty or
        whitespace-only string is None as well: it claims nothing, and a
        claim of nothing is the absence of a claim.

    Never raises. The text is written by a model into a free-text field, so
    every shape of malformation has to come back as "no usable claim" rather
    than as an exception that takes the iteration down.
    """

    if not isinstance(raw, str) or not raw.strip():
        return None

    names = list(_RULE_NAME.finditer(raw))
    claims: dict[str, int] = {}
    for index, name in enumerate(names):
        stop = names[index + 1].start() if index + 1 < len(names) else len(raw)
        window = raw[name.end() : stop]
        count = _COUNT.search(window)
        # The filler between the name and its count must carry no digit, or
        # the count belongs to something else in the sentence.
        if (
            count
            and count.start() <= _FILLER_LIMIT
            and not any(ch.isdigit() for ch in window[: count.start()])
        ):
            claims[name.group(1)] = int(count.group(1))
    if claims:
        return claims

    if _NO_RULES_CLAIM.search(raw):
        return {}

    # No name anywhere, but a count: "Replayed on 50/50 train rows; fired 2".
    # Attributable only when the candidate has exactly one rule to attribute
    # it to, and then it is not a guess. With two or more, the claim really is
    # unreadable and says so.
    if len(known_rules or ()) == 1:
        # Only when the number is unambiguous. "fired 2" names which number is
        # the firing count; a string with exactly one count has no other
        # candidate. A string with several and no anchor —
        # "random sample 0/20; ... full-pool check 3/50" — is three readings
        # of the rule, and picking one of them invented a mismatch against a
        # claim that was true.
        fired = re.search(r"\bfired\b\D{0,20}?(\d+)", raw, re.IGNORECASE)
        counts = _COUNT.findall(raw)
        if fired:
            return {next(iter(known_rules)): int(fired.group(1))}
        if len(counts) == 1:
            return {next(iter(known_rules)): int(counts[0][0])}
    return None


def read_rule_checks(candidates_root: Path) -> dict[str, str]:
    """`{candidate slug: rule_check text}` from `candidates/diagnosis.json`.

    `agentic_proposer._read_diagnosis` parses the same file and is reused for
    everything it returns, but it returns `diagnosis`, `notes` and
    `hypothesis` only — `rule_check` is new and does not come back through
    it, and that module is off limits to edit. So this reads the one extra
    field, under the same tolerances: a missing or unreadable file, or a
    payload of the wrong shape, is no claims rather than an error.
    """

    path = Path(candidates_root) / "diagnosis.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("unreadable diagnosis.json at %s; no rule_check claims", path)
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for entry in payload.get("candidates") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        claim = entry.get("rule_check")
        if name and isinstance(claim, str):
            out[name] = claim
    return out


def _rules_text(filter_dir: Path) -> str:
    path = Path(filter_dir) / "rules.py"
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _check_rule_claim(report: RuleFiringReport, raw: str, *, diagnosis_present: bool) -> None:
    """Compare the proposer's `rule_check` against what was just measured.

    Only the rule name and the count are compared. The denominator is not:
    the proposer runs `rule_tool` over a sample of the TRAIN pool and the
    precheck runs over the rows the accumulated truth pairs name, so the two
    `n`s are different by construction and an honest claim would fail a
    strict comparison.

    Three situations ask nothing, and all three report `not_required`: a
    candidate whose rules.py is byte-identical to its parent's made no claim
    to check; a precheck with no rows measured nothing to check a claim
    against; and a run whose proposer writes no `diagnosis.json` at all has
    no channel to make the claim in, so its silence is not evasion. Only the
    agentic proposer writes that file, and faulting the scripted proposers
    for not writing one would fire on every candidate of every mock run.
    """

    if not report.rules_changed or not report.n_rows or not diagnosis_present:
        report.rule_check_status = "not_required"
        return
    if not raw.strip():
        report.rule_check_status = "absent"
        report.reasons.append(_NO_RULE_CHECK_REASON)
        return

    report.rule_check_raw = raw
    claims = parse_rule_check(raw, [rule.name for rule in report.rules])
    if claims is None:
        report.rule_check_status = "unparseable"
        report.reasons.append(
            f"rule_check in diagnosis.json could not be read ({raw.strip()!r}); expected "
            "a '<rule name> <k>/<n>' count for each rule, or 'none' when the candidate "
            "keeps no rules — treated as absent, so the firing counts were never "
            "looked at"
        )
        return

    measured = {rule.name: rule for rule in report.rules}
    if not claims:
        if measured:
            report.rule_check_status = "mismatch"
            report.reasons.append(
                "rule_check says this candidate has no rules, but its rules.py defines "
                + ", ".join(sorted(measured))
            )
        else:
            report.rule_check_status = "verified"
        return

    mismatched = False
    for name, claimed in claims.items():
        rule = measured.get(name)
        if rule is None:
            mismatched = True
            report.reasons.append(
                f"rule_check claims {name} fired {claimed} time(s), but this candidate's "
                "rules.py defines no rule of that name"
            )
            continue
        if bool(rule.fires) != bool(claimed):
            # Only the categorical disagreement is a finding. The proposer
            # samples its own rows through `rule_tool` and the precheck holds
            # a different set, so `1/20` against a measured `2/50` is two
            # honest readings of the same rule and faulting it would punish
            # the candidate that did the work. What no sampling explains is a
            # rule reported as firing that fires on nothing here, or reported
            # as dead that fires — that is a claim about the rule, not about
            # a row count, and the two measurements contradict each other.
            mismatched = True
            report.reasons.append(
                f"rule_check claims {name} fired {claimed} time(s); measured "
                f"{rule.fires} of {rule.n_rows} precheck rows"
            )
            continue
        report.rule_check_verified.append(name)
    report.rule_check_status = "mismatch" if mismatched else "verified"


def measure_candidate_rules(
    candidate_dir: Path,
    *,
    parent_filter: LoadedFilter,
    trajectories: Sequence[Trajectory],
    rule_check: str = "",
    diagnosis_present: bool = True,
) -> RuleFiringReport:
    """Run the candidate's rules over the precheck's rows and grade the result.

    Two degenerate shapes are escalations, and only for a rule this candidate
    introduced: a rule that fires on none of the rows orders nothing, and a
    rule that fires on all of them separates no pair. Both are
    `needs_strong_evidence` and never `invalid` — a few dozen rows cannot
    prove what a rule does on the pool, so the honest reading is a raised bar
    and not a rejection. A rule the parent already had is measured and
    reported the same way and escalates nothing: it is not this candidate's
    doing, and charging it would make every child of a filter with a dead
    rule look guilty.

    Args:
        candidate_dir: The candidate directory, holding `rules.py`.
        parent_filter: The filter this candidate was proposed from; its rule
            names decide which of the candidate's rules are new, and its
            `rules.py` text decides whether a `rule_check` is owed at all.
        trajectories: The precheck's rows, and therefore the denominator of
            every count reported here.
        rule_check: The candidate's `rule_check` claim from `diagnosis.json`,
            empty when it made none.
        diagnosis_present: Whether a `diagnosis.json` exists for this batch at
            all. False for every proposer that does not write one, and then
            no claim is owed.

    Returns:
        The report, with `reasons` empty when nothing fired. Never raises: a
        `rules.py` that will not import comes back in `error`, because
        `precheck_candidate` already fails that candidate through
        `load_filter` and a second reading of the same break helps nobody.
    """

    from meta_filter.rule_tool import RuleToolRefusal, measure_rules

    rows = list(trajectories)
    report = RuleFiringReport(n_rows=len(rows))
    candidate_dir = Path(candidate_dir)
    parent_dir = Path(parent_filter.directory)
    report.rules_changed = _rules_text(candidate_dir) != _rules_text(parent_dir)
    inherited = {getattr(fn, "__name__", "") for fn in parent_filter.rule_functions}

    rules_path = candidate_dir / "rules.py"
    if not rules_path.is_file():
        report.error = "the candidate has no rules.py"
        return report
    try:
        outcomes = measure_rules(rules_path, rows)
    except RuleToolRefusal as exc:
        report.error = str(exc)
        logger.warning("precheck %s: rules not measurable — %s", candidate_dir.name, exc)
        return report

    for outcome in outcomes:
        rule = RuleFiring(
            name=outcome.name,
            fires=outcome.fire_count,
            n_rows=len(rows),
            raised=outcome.error_count,
            error_types=tuple(outcome.error_types),
            inherited=outcome.name in inherited,
        )
        report.rules.append(rule)
        if rule.inherited or not rows:
            continue
        if rule.raised == rule.n_rows and rule.n_rows:
            # A rule that raises on every row also fires zero times, but the
            # two want opposite repairs: one is a wrong predicate, the other
            # is a bug. Saying "fired on 0 rows" would send the proposer to
            # the wrong one.
            report.reasons.append(
                f"{rule.name} raised on all {rule.n_rows} precheck rows "
                f"({', '.join(rule.error_types) or 'unknown error'}) — "
                "it never returns a verdict"
            )
        elif rule.fires == 0:
            report.reasons.append(
                f"{rule.name} fired on 0 of {rule.n_rows} precheck rows — it cannot order anything"
            )
        elif rule.fires == rule.n_rows:
            report.reasons.append(
                f"{rule.name} fired on {rule.n_rows} of {rule.n_rows} precheck rows — "
                "it separates no pair"
            )

    _check_rule_claim(report, rule_check, diagnosis_present=diagnosis_present)
    return report


def precheck_candidate(
    candidate_dir: Path,
    *,
    parent_filter: LoadedFilter,
    trajectories_by_id: dict[str, Trajectory],
    truth_pairs: Sequence[tuple[str, str]],
    scorer,
    tied_pairs: Iterable[tuple[str, str]] = (),
    lazy_scoring: bool = True,
    score_weights: dict[str, float] | None = None,
    redundancy_weight: float = 0.0,
    redundancy_threshold: float = 0.7,
    cost_ledger: CostLedger | None = COST_LEDGER,
    cost_role: str = "solver",
    max_misorders: int = 40,
) -> PrecheckResult:
    """Score one candidate against the already-judged pairs. No judge calls.

    Raises nothing: a candidate whose own code is broken is a broken
    candidate, and the precheck says so in `error` rather than taking the
    iteration down with it — the same containment `_evaluate_candidate` uses
    around the real solver pass.
    """

    result = PrecheckResult(candidate_id=candidate_dir.name)
    weights = {**DEFAULT_SCORE_WEIGHTS, **(score_weights or {})}
    result.omitted_terms = [term for key, term in _UNCOMPUTABLE_WEIGHTS.items() if weights.get(key)]

    breakdown, complexity_score = measure_filter_complexity(candidate_dir)
    result.complexity = complexity_score
    result.complexity_breakdown = breakdown

    try:
        loaded = load_filter(candidate_dir)
    except Exception as exc:  # noqa: BLE001 — a candidate is LLM-written code
        result.error = f"{type(exc).__name__}: {exc}"
        logger.warning("precheck %s: candidate does not load — %s", candidate_dir.name, exc)
        return result

    declared = getattr(loaded.policy_module, "RUBRIC_CRITERIA", None)
    result.criteria_defect = rubric_criteria_defect_of(declared) or ""
    criteria = (
        [c for c in declared if isinstance(c, str) and c.strip()]
        if isinstance(declared, list | tuple)
        else []
    )
    parent_criteria = list(getattr(parent_filter.policy_module, "RUBRIC_CRITERIA", []) or [])
    result.criterion_delta = changed_criteria(
        parent_filter.rubric_md, parent_criteria, loaded.rubric_md, criteria
    )
    if result.criteria_defect:
        # Same contract `constraints.rubric_criteria_defect` fires on from the
        # source, said again from the value the interpreter produced so a
        # declaration built at import time is caught too. `run_precheck` turns
        # it into an `invalid` firing; the error here is what the proposer
        # reads in the precheck block.
        result.error = result.criteria_defect
        logger.warning("precheck %s: %s", candidate_dir.name, result.error)
        return result

    pairs = [
        (hi, lo) for hi, lo in truth_pairs if hi in trajectories_by_id and lo in trajectories_by_id
    ]
    row_ids = sorted({tid for pair in pairs for tid in pair})
    rows = [trajectories_by_id[tid] for tid in row_ids]
    result.n_rows = len(rows)
    if not rows:
        result.error = "no already-judged pair lies in this pool — nothing to check"
        logger.info("precheck %s: %s", candidate_dir.name, result.error)
        return result

    cached_before, missing_before = cache_census(scorer, rows, criteria, loaded.rubric_md)
    result.cells_reused = cached_before
    result.cells_rescored = missing_before

    # What the precheck itself spends, taken off the same ledger the run
    # bills every other role against. A delta rather than an estimate: the
    # precheck runs between the proposer's two legs with nothing else in
    # flight, so everything the scorer role bills in this window is this
    # candidate's.
    mark = cost_ledger.snapshot() if cost_ledger is not None else None
    try:
        # Warm the per-criterion cache in parallel first: Solver.run scores
        # rows one at a time, and a candidate that rewrote its rubric needs
        # every cell fresh — serial, that was ~2 s per cell, half an hour per
        # iteration on the 50-row pool before this call was added.
        if hasattr(scorer, "prefetch_parallel") and criteria:
            scorer.prefetch_parallel(rows, criteria, loaded.rubric_md)
        decisions = Solver(rubric_scorer=scorer, lazy_scoring=lazy_scoring).run(loaded, rows)
    except Exception as exc:  # noqa: BLE001 — candidate code runs in here
        result.error = f"{type(exc).__name__}: {exc}"
        logger.warning("precheck %s: candidate raised while scoring — %s", candidate_dir.name, exc)
        return result

    result.llm_calls = sum(d.llm_calls for d in decisions)
    result.llm_tokens = sum(d.llm_tokens for d in decisions)
    if mark is not None:
        spend = cost_ledger.since(mark).get(cost_role, RoleCost())
        # The parallel warm-up paid for the cells before Solver.run read them
        # back as hits, so the decisions undercount; the ledger saw it all.
        result.llm_calls = spend.calls
        result.llm_tokens = spend.tokens
        result.cost_usd = spend.cost_usd
        # A model with no published price bills 0.0, which is a hole rather
        # than a fact — say which one this is.
        result.cost_source = "unpriced" if spend.unpriced_calls else "metered"

    scores = {d.trajectory_id: float(d.rank_score) for d in decisions if d.rank_score is not None}
    detail = ranking_math.concordance_detail(
        scores, list(pairs), unscored="loss", ties=list(tied_pairs)
    )
    result.pair_credit = pair_correctness(scores, pairs, ties=tied_pairs)
    result.concordance = detail.concordance
    result.n_pairs = detail.n_offered
    result.n_tied = detail.n_tied
    result.unscored_pairs = detail.n_unscored
    result.unscored_rows = detail.missing_ids
    result.distinct_levels = ranking_math.distinct_levels(scores)

    result.criterion_stats = criterion_stats(decisions)
    result.redundancy_penalty = redundancy_penalty(
        result.criterion_stats,
        weight=redundancy_weight,
        threshold=redundancy_threshold,
    )

    # The score, then the reported one with the redundancy penalty taken off;
    # both are written to precheck.json.
    result.score, result.score_with_redundancy = charge_redundancy(
        compute_score(
            _metrics_shell(result.concordance, complexity_score, breakdown),
            weights=weights,
        ),
        result.redundancy_penalty,
    )

    tie_keys = {frozenset(p) for p in tied_pairs}
    result.misordered_pairs = _misorder_records(
        scores,
        [p for p in pairs if frozenset(p) not in tie_keys],
        max_misorders,
    )
    return result


def run_precheck(
    candidate_dirs: Sequence[Path],
    *,
    parent_filter: LoadedFilter,
    trajectories_by_id: dict[str, Trajectory],
    truth_pairs: Sequence[tuple[str, str]],
    scorer,
    tied_pairs: Iterable[tuple[str, str]] = (),
    lazy_scoring: bool = True,
    score_weights: dict[str, float] | None = None,
    initial_dir: Path,
    redundancy_weight: float = 0.0,
    redundancy_threshold: float = 0.7,
    cost_ledger: CostLedger | None = COST_LEDGER,
    cost_role: str = "solver",
    write: bool = True,
    limits: ConstraintLimits | None = None,
    baseline_stats: dict | None = None,
    freeze: Sequence[str] = (),
    zero_lift_criteria: Sequence[str] = (),
    zero_lift_rules: Sequence[str] = (),
    hard_caps: Mapping[str, int] | None = None,
) -> dict[str, PrecheckResult]:
    """Precheck every candidate and write `<candidate>/precheck.json`.

    Sequential on purpose. The expensive part is the scorer, which fans out
    internally over rows (`prefetch_parallel`), and candidates share a cache
    — running them in a second pool would mostly race two candidates onto the
    same missing cell.

    Every candidate is also measured against the growth caps and deletion
    floor (`initial_dir` is the run's starting filter), with the parent as
    the per-iteration baseline. The check reads files rather than importing
    code, so it is reported even for a candidate that does not load. It does
    not skip scoring for an `invalid` candidate: the revision leg still needs
    the number to judge whether the fix kept what worked. `hard_caps` is
    passed straight through to `check_candidate`, where it replaces the run
    hard cap — and nothing else — with explicit numbers.

    Any outcome field named in rules.py, policy.py OR rubric.md is
    `invalid`, the initial filter's own references included.

    A policy.py that declares no scoreable `RUBRIC_CRITERIA` is `invalid`
    too: `check_candidate` fires on the source, and whatever only the
    interpreter can see is folded in from `PrecheckResult.criteria_defect`
    (`record_criteria_defect`). Both matter — otherwise such a candidate
    would be reported `ok` here and then dropped inside the solver pass with
    nothing but a WARNING.

    `freeze` (`--freeze`) makes a changed frozen file `invalid`,
    and once a candidate is scored its own readout is compared against
    `baseline_stats` (the parent's `criterion_stats` as the proposer saw
    them) and folded back into the report (`assess_rubric_health`), so a NEW
    dead axis or redundant pair raises its bar the way a soft cap does.
    """

    from meta_filter.propose.agentic_proposer import _match_notes

    tied = list(tied_pairs)
    rows = list(trajectories_by_id.values())
    # One read of `candidates/diagnosis.json` for the whole batch: every
    # candidate of an iteration lives under the same root, and the file is the
    # proposer's, not any one candidate's.
    roots = {Path(cand_dir).parent for cand_dir in candidate_dirs}
    diagnosis_present = any((root / "diagnosis.json").is_file() for root in roots)
    rule_checks: dict[str, str] = {}
    for root in roots:
        rule_checks.update(read_rule_checks(root))

    results: dict[str, PrecheckResult] = {}
    for cand_dir in candidate_dirs:
        cand_dir = Path(cand_dir)
        constraint = check_candidate(
            cand_dir,
            baseline_dir=Path(parent_filter.directory),
            initial_dir=Path(initial_dir),
            limits=limits if limits is not None else ConstraintLimits.DEFAULT,
            freeze=freeze,
            zero_lift_criteria=zero_lift_criteria,
            zero_lift_rules=zero_lift_rules,
            hard_caps=hard_caps,
        )
        result = precheck_candidate(
            cand_dir,
            parent_filter=parent_filter,
            trajectories_by_id=trajectories_by_id,
            truth_pairs=truth_pairs,
            scorer=scorer,
            tied_pairs=tied,
            lazy_scoring=lazy_scoring,
            score_weights=score_weights,
            redundancy_weight=redundancy_weight,
            redundancy_threshold=redundancy_threshold,
            cost_ledger=cost_ledger,
            cost_role=cost_role,
        )
        if constraint is not None and result.criteria_defect:
            # The source-level reading inside `check_candidate` cannot
            # evaluate every declaration; this one was read off the imported
            # module. Folding it in here is what makes `gate_candidates` drop
            # the candidate rather than leaving it to be lost inside the
            # solver pass, where nothing but a log line recorded it.
            record_criteria_defect(constraint, result.criteria_defect)
        if constraint is not None and result.criterion_stats is not None:
            assess_rubric_health(constraint, result.criterion_stats, baseline_stats)
        # Measured whether or not the candidate scored: rules are pure Python
        # and cost nothing, and a candidate whose rubric broke still wrote
        # rules that the revision leg has to be told about.
        result.rule_firing = measure_candidate_rules(
            cand_dir,
            parent_filter=parent_filter,
            trajectories=rows,
            rule_check=_match_notes(cand_dir.name, rule_checks),
            diagnosis_present=diagnosis_present,
        )
        if constraint is not None:
            record_rule_firing(constraint, result.rule_firing.reasons)
        result.constraint = constraint
        results[cand_dir.name] = result
        if write:
            if not cand_dir.is_dir():
                # The proposer removed the directory between writing it and
                # now; recreating it just to hold a readout would fake a
                # candidate. The result still goes back to the caller.
                logger.warning(
                    "precheck %s: directory gone, not writing precheck.json", cand_dir.name
                )
            else:
                (cand_dir / "precheck.json").write_text(
                    json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        logger.info(
            "precheck %s: score %.4f (concordance %.4f over %d pairs, %d TIE; "
            "redundancy -%.3f%s; complexity %.1f); %d cells re-scored, %d reused",
            result.candidate_id,
            result.score,
            result.concordance,
            result.n_pairs,
            result.n_tied,
            result.redundancy_penalty,
            ", reported not charged",
            result.complexity,
            result.cells_rescored,
            result.cells_reused,
        )
    return results


def _constraint_line(constraint: ConstraintReport | None) -> str:
    """One prompt line when a cap fired; empty otherwise.

    `needs_strong_evidence` tells the proposer the acceptance bar for this
    candidate is one z-unit higher; `INVALID` (upper-cased so it stands out in a
    block of lower-case numbers) tells it the candidate will not be scored
    at all unless the revision brings it back under the cap.
    """

    if constraint is None:
        return ""
    if constraint.simplification:
        line = (
            "  SIMPLIFICATION (smaller than the current filter in every component): "
            "accepted if it loses no hidden pair (b >= c), outside the significance "
            "test and the deletion limit"
        )
        if constraint.status != "ok":
            line += f"\n  constraint: {constraint.summary()}"
        return line
    if constraint.status == "ok":
        return ""
    return f"  constraint: {constraint.summary()}"


def _rubric_health_lines(constraint: ConstraintReport | None) -> list[str]:
    """What this candidate's rubric costs for nothing, fired or not.

    Three readouts, always printed when non-empty, whatever the status: the
    criteria policy.py declares and never reads
    (`constraints.declared_unread_criteria`), the criteria it declares at
    weight exactly zero (`constraints.zero_weight_criteria` — read, so
    the first readout stays silent, and free to vary, so the third does too),
    and the axes whose scores sit at one level
    (`criterion_stats.dead_axes`). All three are one scorer call per
    row that changes no ordering, and all are usually INHERITED — which is
    exactly why they cannot be reported through `reasons`, where everything
    is something the candidate did wrong. A proposer that only ever sees its
    own new defects will carry an old one to the end of the run.
    """

    if constraint is None:
        return []
    lines: list[str] = []
    if constraint.declared_unread:
        lines.append(
            "  declared but never read by policy.py: "
            + ", ".join(constraint.declared_unread)
            + " — one scorer call per row each, and no effect on the ordering"
        )
    if constraint.zero_weight:
        lines.append(
            "  declared at weight 0: "
            + ", ".join(constraint.zero_weight)
            + " — read, but the weighted sum multiplies it by nothing; the same "
            "bill again. Remove it, or give it weight. The deletion floor does "
            "not protect a criterion the current filter already declares at zero"
        )
    if constraint.dead_axes:
        lines.append(
            "  dead axes (one level on almost every row): "
            + ", ".join(constraint.dead_axes)
            + " — same bill, measured in the scores rather than in policy.py"
        )
    return lines


def _rule_firing_lines(report: RuleFiringReport | None) -> list[str]:
    """What each rule fired on, always with its denominator.

    Printed for every candidate whether or not anything fired, and for
    inherited rules as well as new ones, for the reason
    `_rubric_health_lines` exists: a rule the parent contributed is still a
    rule that orders nothing, and a proposer only ever shown its own new
    defects carries an old one to the end of the run. The escalations
    themselves come through `constraint.summary()` and are not repeated here.
    """

    if report is None:
        return []
    if report.error:
        return [f"  rules not measured: {report.error}"]
    if not report.rules:
        return []
    lines = [f"  rules, over the {report.n_rows} precheck rows (pure Python, no cost):"]
    for rule in report.rules:
        note = " [inherited]" if rule.inherited else ""
        if rule.raised:
            note += f", raised {', '.join(rule.error_types)} on {rule.raised} row(s)"
        lines.append(f"    {rule.name}: fired on {rule.fires} of {rule.n_rows}{note}")
    if report.rule_check_status == "verified" and report.rule_check_verified:
        lines.append(
            "  rule_check in diagnosis.json matches the measurement for "
            + ", ".join(report.rule_check_verified)
        )
    return lines


def render_precheck_block(
    results: dict[str, PrecheckResult],
    *,
    parent: PrecheckResult | None = None,
    max_misorders: int = 8,
    redundancy_threshold: float = 0.7,
) -> str:
    """The revision leg's prompt block: what each candidate actually scored.

    Written as prose the proposer reads rather than raw JSON, because the
    same JSON is already on disk beside every candidate and the prompt's job
    is to say what it means and what may be done about it.

    The number is the formula the precheck reports, not the one that picks
    the winner — a paired test on hidden pairs does that — and the block
    says so.

    Under every candidate, fired or not, `_rubric_health_lines` names the
    criteria its policy declares and never reads and the axes its scores
    leave dead — the two ways a rubric can charge a scorer call per row for
    an ordering it does not change.
    """

    lines = [
        "## PRECHECK — what your candidates score right now",
        "",
        "Every candidate you just wrote was run against the pairs a judge has",
        "ALREADY adjudicated in earlier iterations of this run, using the",
        "cached rubric cells wherever your edit left a criterion untouched.",
        "The number below is the formula the precheck reports — evidence for",
        "your revision, not the acceptance decision, which is made on a fixed",
        "set of pairs you cannot see, from rows outside this pool:",
        "",
        "    score = concordance - complexity_term",
        "",
        f"The redundancy figure printed for each candidate (|rho| > "
        f"{redundancy_threshold:g} between two of its criteria) is measured and",
        "shown here but does NOT enter this iteration's selection; it is charged",
        "to the final filter's reported score, so avoid it where you can without",
        "giving up ordering power.",
        "",
    ]
    lines.extend(
        [
            "What it is not: no judge was called, so no new verdict exists; the",
            "pairs are the ones already bought, over the rows those pairs name,",
            "not the whole training pool. A criterion you did not touch was not",
            "re-scored — its cells came from the cache unchanged.",
            "",
        ]
    )
    if parent is not None:
        lines.append(
            f"Parent filter (`{parent.candidate_id}`) on the same pairs: "
            f"score {parent.score:.4f}, concordance {parent.concordance:.4f}."
        )
        lines.append("")

    for result in results.values():
        # The caps are a file-level fact, so they are reported for a broken
        # candidate too: the revision leg should not fix the import and then
        # learn about the cap one leg later.
        constraint_line = _constraint_line(result.constraint)
        health_lines = _rubric_health_lines(result.constraint) + _rule_firing_lines(
            result.rule_firing
        )
        if result.error:
            lines.append(f"### {result.candidate_id} — NOT SCOREABLE")
            lines.append(f"  {result.error}")
            if constraint_line:
                lines.append(constraint_line)
            lines.extend(health_lines)
            lines.append("")
            continue
        delta = ""
        if parent is not None:
            delta = f" ({result.score - parent.score:+.4f} vs the parent)"
        lines.append(f"### {result.candidate_id} — score {result.score:.4f}{delta}")
        if constraint_line:
            lines.append(constraint_line)
        lines.extend(health_lines)
        lines.append(
            f"  concordance {result.concordance:.4f} over {result.n_pairs} pairs "
            f"({result.n_tied} judge-TIE excluded), redundancy "
            f"-{result.redundancy_penalty:.3f}"
            f" (not in the score above; reported score {result.score_with_redundancy:.4f})"
            f", complexity {result.complexity:.1f}"
        )
        lines.append(
            f"  criteria re-scored: {len(result.criterion_delta.changed)} of "
            f"{len(result.criterion_delta.changed) + len(result.criterion_delta.unchanged)} "
            f"({result.cells_rescored} cells paid for, {result.cells_reused} reused"
            + (
                ", rubric preamble edited so every criterion re-scored"
                if result.criterion_delta.header_changed
                else ""
            )
            + ")"
        )
        if result.unscored_pairs:
            lines.append(
                f"  RANKING CONTRACT VIOLATION: no score for "
                f"{result.unscored_rows} row(s), costing {result.unscored_pairs} pairs"
            )
        if result.misordered_pairs:
            lines.append("  pairs you order against the judge (judge ranked hi > lo):")
            for m in result.misordered_pairs[:max_misorders]:
                lines.append(
                    f"    {m['direction']}: {m['hi']} ({m['hi_score']}) vs "
                    f"{m['lo']} ({m['lo_score']})"
                )
            extra = len(result.misordered_pairs) - max_misorders
            if extra > 0:
                lines.append(f"    ... and {extra} more, in precheck.json")
        lines.append("")
    return "\n".join(lines)
