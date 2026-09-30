"""Growth and deletion constraints on a candidate filter.

A complexity penalty weight invites farming: the cheapest gain it offers is
pruning rubric axes to shrink the complexity score, whatever that does to
the ordering. With no weight at all, nothing resists growth. A cap is a
different kind of instrument. It does not trade off against concordance, so
there is no exchange rate to tune and nothing to farm. It
says: within one iteration a candidate may grow this much on its own say-so
(the soft cap — beyond it the acceptance gate demands stronger evidence);
across the whole run it may never exceed this multiple of where it started
(the hard cap); and it may never fall more than this far below the initial
filter's axis count (the deletion floor, which is what actually stops the
pruning). Each is stated per component of `complexity.measure_filter_complexity`
so a rubric edit and a rules edit are bounded separately — except
`policy_loc`, which is measured and reported but capped by nothing (see
`ConstraintLimits`).

Deletion is detected by NAME, not by count, because the count cannot see a
merge: two criteria folded into one new one leaves `rubric_criteria` down by
one, exactly like a plain deletion, and it is treated as one. The names
also feed "at most one deletion candidate per iteration".

The rationale check (one mechanism per candidate) lives here too
because it is the same shape of thing — a structural fact about the candidate
directory that the precheck can read without an LLM call.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from meta_filter.complexity import DEFAULT_COMPLEXITY_COEFFICIENTS, measure_filter_complexity
from meta_filter.criterion_stats import REDUNDANT_RHO, dead_axes, redundant_pairs

logger = logging.getLogger(__name__)

COMPONENTS = ("rubric_criteria", "rule_count", "rules_loc", "policy_loc", "rubric_chars")

# Components whose soft cap is an absolute headcount rather than a share.
# Two more criteria is two more hypotheses whatever the rubric's size; 15 %
# more criteria on a six-axis rubric would be 0.9 of one, which is nothing.
_ABSOLUTE_COMPONENTS = frozenset({"rubric_criteria", "rule_count"})

# One line per declared hypothesis, in any of the ways a proposer writes one:
# `Hypothesis: ...`, `- Hypothesis: ...`, `**Hypothesis**: ...`, or a Markdown
# heading `## Hypothesis: ...`. The heading form is not in the interface note
# but is how the proposers actually title sections, and missing it would let a
# two-mechanism candidate through on formatting alone.
_HYPOTHESIS_LINE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?\*{0,2}hypothesis\*{0,2}\s*:", re.IGNORECASE
)

ConstraintStatus = Literal["ok", "needs_strong_evidence", "invalid"]


# Worst wins. `invalid` is a hard stop, `needs_strong_evidence` a raised bar.
_SEVERITY: dict[str, int] = {"ok": 0, "needs_strong_evidence": 1, "invalid": 2}

# No-loss simplification, and rubric health.
#
# A narrow licence for shrinking (drop a dead axis with every other file
# untouched) is a lane a proposer has to hit byte for byte, and it will not.
# The general rule is the one that matters: complexity is out of the
# acceptance decision, so nothing in the loop rewards making the filter
# smaller, and a pure removal earns nothing — it changes no ordering, reads
# as `b = c = 0`, and costs the proposer a candidate slot for free.
#
# So: a candidate that is SMALLER than its baseline in every component and
# strictly smaller in total (`is_simplification`) is accepted whenever it
# loses no hidden pair (`b >= c`), outside the significance test, the
# one-deletion limit and the no-op rejection (`selector.choose_paired`,
# `optimizer.gate_candidates`). It does not matter whether what it removed
# was a dead axis, a redundant criterion or a rule that never fired: the
# filter got smaller and the hidden pairs did not notice, which is the whole
# argument for the edit.
#
# A candidate that INTRODUCES a dead axis or a
# redundant pair, relative to its baseline, needs strong evidence (one
# z-unit higher, like a soft cap); an inherited defect is not the
# candidate's fault and fires nothing.
#
# A NEW dead axis is held to a stricter definition than the health block's
# 0.90 (`criterion_stats.DEAD_MODAL_SHARE`), and only over enough rows for
# the share to mean something: a fresh criterion at 92 % on 30 judged rows is
# a criterion that has not found its range yet, not a defect.
NEW_DEAD_SHARE = 0.95
NEW_DEAD_MIN_ROWS = 40

# `--freeze` names, and the file each one pins.
FROZEN_FILES: dict[str, str] = {"rubric": "rubric.md", "rules": "rules.py", "policy": "policy.py"}


@dataclass(frozen=True)
class ConstraintLimits:
    """The caps, as numbers the proposer prompt can print verbatim.

    `soft_add` is per iteration and measured against the BASELINE (the
    parent the candidate was edited from): an absolute headcount for the
    components in `_ABSOLUTE_COMPONENTS`, a relative share for the rest.
    `hard_ratio` and `deletion_floor` are per run and measured against the
    INITIAL filter, so a chain of just-under-the-soft-cap steps still has a
    ceiling, and a chain of one-at-a-time deletions still has a floor.

    Two components are treated differently:

    * `policy_loc` is in neither cap (`hard_exempt`, and absent from
      `soft_add`). A policy line costs no LLM call, and a hard cap would
      throw out a policy-only candidate however good its ordering. It stays
      in every breakdown and report.
    * the rule floor is `initial − 5`, but never below `floor_min`
      (3 functions): pruning a nine-rule filter to five is a legitimate
      edit that `initial − 1` would forbid. The rubric floor stays at
      `initial − 1`. `floor_for` is the one place the floor is computed, so
      the check and the proposer prompt print the same number.
    """

    soft_add: dict[str, float] = field(
        default_factory=lambda: {
            "rubric_criteria": 2,
            "rule_count": 2,
            "rules_loc": 0.15,
            "rubric_chars": 0.15,
        }
    )
    # Absolute headroom a relative `soft_add` may never fall below, per
    # iteration and against the baseline. A share is worthless once the
    # baseline is small: a filter whose rules.py is 2 lines gets `2 * 15 //
    # 100 == 0` extra lines, so ANY candidate that writes a rule body is over
    # the soft cap and is held to the strong-evidence threshold — while
    # `rule_count` cheerfully allows +2 rules — a contradiction the proposer
    # prompt would print verbatim ("+2 rules (to 2), rules.py to 2 LOC").
    # Two rule bodies is what `rule_count` licenses, and a rule body is about
    # 16 lines at the median (p75 17), so 35 is two of them. Above a 233-line
    # rules.py the share binds again and the brake on large filters is
    # unchanged.
    soft_floor: dict[str, int] = field(default_factory=lambda: {"rules_loc": 35})
    hard_ratio: float = 1.5
    # Components the run hard cap does not apply to.
    hard_exempt: frozenset[str] = frozenset({"policy_loc"})
    deletion_floor: dict[str, int] = field(
        default_factory=lambda: {"rubric_criteria": 1, "rule_count": 5}
    )
    # Absolute minimum a floored component may be asked to keep, whatever
    # `initial - deletion_floor` says: a small initial filter is not licence
    # to delete everything.
    floor_min: dict[str, int] = field(default_factory=lambda: {"rule_count": 3})

    DEFAULT: ClassVar[ConstraintLimits]

    def soft_max_for(self, name: str, base: int) -> int | None:
        """The largest `name` value a candidate may reach without needing
        strong evidence, given the baseline's count; None when uncapped.

        The one place the soft cap is computed, so `check_candidate` fires on
        exactly the number `proposer` prints — the reason `floor_for` exists
        for the other direction. An absolute component is `base + allowance`;
        a relative one is the share, never less than `soft_floor` of headroom.
        """

        allowance = self.soft_add.get(name)
        if allowance is None:
            return None
        if name in _ABSOLUTE_COMPONENTS:
            return base + int(allowance)
        return max(soft_cap(base, allowance), base + self.soft_floor.get(name, 0))

    def floor_for(self, name: str, initial: int) -> int | None:
        """The smallest `name` count a candidate may have, given the initial
        filter's count; None when the component has no floor.

        ``max(floor_min, initial - deletion_floor)``, clamped at 0. When the
        minimum is what binds it is itself capped at ``initial - 1``: a filter
        that started with two rules may still delete one, rather than every
        candidate reading as under a floor of 3. An explicit
        ``deletion_floor`` of 0 (no deletions) is not touched by that cap.
        With the defaults, rules: ``max(3, initial - 5)`` (``initial - 1``
        below four rules); criteria: ``initial - 1``.
        """

        allowance = self.deletion_floor.get(name)
        if allowance is None:
            return None
        floor = max(0, initial - allowance)
        minimum = self.floor_min.get(name)
        if minimum is not None and minimum > floor:
            floor = min(minimum, max(floor, initial - 1))
        return floor

    def to_dict(self) -> dict:
        return {
            "soft_add": dict(self.soft_add),
            "soft_floor": dict(self.soft_floor),
            "hard_ratio": self.hard_ratio,
            "hard_exempt": sorted(self.hard_exempt),
            "deletion_floor": dict(self.deletion_floor),
            "floor_min": dict(self.floor_min),
        }


ConstraintLimits.DEFAULT = ConstraintLimits()


@dataclass
class ConstraintReport:
    """What the caps say about one candidate. Serialized inside `precheck.json`.

    `reasons` is one human-readable string per firing so the revision leg can
    quote them back to the proposer unedited; `status` is the worst of them.
    `deleted` / `added` are names (criteria and `rule_` functions) so a merge
    reads as what it is — two names gone, one new — rather than as a count
    that moved by one.
    """

    status: ConstraintStatus
    reasons: list[str]
    deletion: bool
    deleted: list[str]
    added: list[str]
    mechanism_count: int | None
    candidate: dict[str, int]
    baseline: dict[str, int]
    initial: dict[str, int]
    # Names in `deleted` the harness measured at exactly zero rank shift.
    # `deletion` excludes them: they cannot hide damage, so they do not
    # consume the iteration's one deletion slot (see `_inert_removals`).
    deleted_inert: list[str] = field(default_factory=list)
    # `simplification` says the candidate is smaller than its
    # baseline in every complexity component and strictly smaller in total
    # (`is_simplification`); the gate then accepts it on `b >= c` alone.
    # `new_dead_axes` and `new_redundant_pairs` are defects the candidate
    # introduced (measured after scoring, `assess_rubric_health`); either one
    # fires `needs_strong_evidence`. `frozen_changed` lists files a
    # `--freeze` run pinned that the candidate changed anyway (`invalid`).
    simplification: bool = False
    new_dead_axes: list[str] = field(default_factory=list)
    new_redundant_pairs: list[list[str]] = field(default_factory=list)
    frozen_changed: list[str] = field(default_factory=list)
    # Rubric bookkeeping, reported on EVERY candidate whether or not
    # anything fired. `declared_unread` is what the candidate's own policy.py
    # declares in RUBRIC_CRITERIA and never reads
    # (`declared_unread_criteria`); `dead_axes` is what its scores say
    # (`criterion_stats.dead_axes`, filled by `assess_rubric_health`). Both
    # cost one scorer call per row and order nothing, and both are inherited
    # far more often than they are introduced — so they are a readout here,
    # and only the introduction of the first one fires a status.
    declared_unread: list[str] = field(default_factory=list)
    # Criteria the candidate's policy.py declares and maps to a weight of
    # exactly 0 (`zero_weight_criteria`). Filled on every report whether or
    # not it fired, so the precheck block can name an INHERITED one without
    # the candidate being charged for it — the rule `declared_unread` and
    # `new_dead_axes` already follow.
    zero_weight: list[str] = field(default_factory=list)
    dead_axes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "deletion": self.deletion,
            "deleted": list(self.deleted),
            # Why `deletion` can be False while `deleted` is not empty.
            "deleted_inert": list(self.deleted_inert),
            "added": list(self.added),
            "mechanism_count": self.mechanism_count,
            "candidate": dict(self.candidate),
            "baseline": dict(self.baseline),
            "initial": dict(self.initial),
            "simplification": self.simplification,
            "new_dead_axes": list(self.new_dead_axes),
            "new_redundant_pairs": [list(p) for p in self.new_redundant_pairs],
            "frozen_changed": list(self.frozen_changed),
            "declared_unread": list(self.declared_unread),
            "zero_weight": list(self.zero_weight),
            "dead_axes": list(self.dead_axes),
        }

    def summary(self) -> str:
        """The one-line form the precheck block prints: `status — reasons`."""

        label = "INVALID" if self.status == "invalid" else self.status
        text = f"{label} — {'; '.join(self.reasons)}" if self.reasons else label
        if self.deleted:
            text += f"; deleted: {', '.join(self.deleted)}"
        if self.simplification:
            text += "; SIMPLIFICATION"
        return text


def criterion_names(rubric_path: Path) -> list[str]:
    """Criterion names in rubric order, from the same lines complexity counts.

    Mirrors `complexity._count_rubric_criteria` line for line so the names
    and the count can never disagree: a heading `## Criterion: Foo` or a bare
    `criterion: Foo` line both count there, so both name a criterion here.
    """

    if not rubric_path.exists():
        return []
    names: list[str] = []
    for line in rubric_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## Criterion"):
            rest = stripped[len("## Criterion") :]
        elif stripped.lower().startswith("criterion:"):
            rest = stripped[len("criterion:") :]
        else:
            continue
        names.append(rest.lstrip(":").strip().rstrip("#").strip())
    return names


def rule_names(rules_path: Path) -> list[str]:
    """Top-level `rule_*` function names, the units `rule_count` counts."""

    if not rules_path.exists():
        return []
    try:
        tree = ast.parse(rules_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("rule_")
    ]


def _named_units(filter_dir: Path) -> list[str]:
    return criterion_names(filter_dir / "rubric.md") + rule_names(filter_dir / "rules.py")


def _inert_removals(
    component: str,
    baseline_dir: Path,
    candidate_dir: Path,
    zero_lift_criteria: Iterable[str],
    zero_lift_rules: Iterable[str],
) -> int:
    """How many of this component's removals the harness measured as inert.

    Not what the candidate claims: both name sets come from ablating the
    BASELINE and re-ranking, so a candidate cannot talk its way into the
    credit — it can only earn one by deleting something already measured at
    exactly zero rank shift.
    """

    if component == "rubric_criteria":
        gone = set(declared_criteria(baseline_dir)) - set(declared_criteria(candidate_dir))
        # Measured inert, OR declared by the baseline at weight exactly zero.
        # The second half is what makes the credit available at iteration 0
        # and on any run whose lift measurement has not landed yet: without
        # it a filter at the criteria floor carrying a zero-weight criterion
        # can neither delete it (floor) nor keep it (`newly_zero_weight`),
        # which is a position with no legal move. Both halves are facts about
        # the BASELINE, so neither is farmable by a candidate.
        inert = set(zero_lift_criteria) | set(zero_weight_criteria(baseline_dir))
        return len(gone & inert) if inert else 0
    if component == "rule_count" and zero_lift_rules:
        gone = set(rule_names(baseline_dir / "rules.py")) - set(
            rule_names(candidate_dir / "rules.py")
        )
        return len(gone & set(zero_lift_rules))
    return 0


def _percent(share: float) -> int:
    """A relative allowance as whole percent (0.15 -> 15), for integer caps."""

    return round(share * 100)


def soft_cap(base: int, share: float) -> int:
    """The largest value within `share` of `base`: 100 -> 115, 3000 -> 3450.

    The same integer arithmetic `check_candidate` fires on, so the number the
    proposer prompt prints is exactly the number a candidate may reach.
    """

    return base + base * _percent(share) // 100


def hypothesis_lines(text: str) -> list[str]:
    """The body of every `Hypothesis:` line in `text`, in order.

    Matches every way a proposer writes one (`_HYPOTHESIS_LINE`), so the
    writer that produces `rationale.md` and the check that counts its lines
    agree on what a hypothesis line is.
    """

    out: list[str] = []
    for line in text.splitlines():
        match = _HYPOTHESIS_LINE.match(line)
        if match:
            out.append(line[match.end() :].strip())
    return out


def count_mechanisms(rationale_path: Path) -> int | None:
    """How many hypotheses `rationale.md` declares; None when there is none.

    A heading count, deliberately not an LLM call: the rule is "one mechanism
    per candidate" and the cheapest way to hold a proposer to it is to have
    it write `Hypothesis:` once and count. Zero is not a firing — a candidate
    without a rationale is a documentation gap the prompt handles, not a
    confounded edit.
    """

    if not rationale_path.exists():
        return None
    text = rationale_path.read_text(encoding="utf-8", errors="replace")
    return len(hypothesis_lines(text))


def parse_freeze(spec: str | Iterable[str] | None) -> tuple[str, ...]:
    """`--freeze rules,policy` → `("rules", "policy")`, validated and de-duplicated.

    Raises `ValueError` naming the unknown token, so the CLI can refuse a
    typo before a run starts under a freeze that pins nothing.
    """

    if spec is None:
        return ()
    tokens = spec.split(",") if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for token in tokens:
        name = token.strip().lower()
        if not name:
            continue
        if name not in FROZEN_FILES:
            raise ValueError(
                f"unknown --freeze target {token.strip()!r}; choose from {', '.join(FROZEN_FILES)}"
            )
        if name not in out:
            out.append(name)
    return tuple(out)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def frozen_changed(candidate_dir: Path, baseline_dir: Path, freeze: Iterable[str]) -> list[str]:
    """Which of the frozen files the candidate's bytes differ from the baseline's.

    `--freeze rubric` additionally pins `RUBRIC_CRITERIA`, which lives in
    policy.py rather than rubric.md. Without that, freezing the rubric does
    not close the rubric channel: with rubric.md frozen and policy.py
    editable, a candidate can put a criterion name back into
    `RUBRIC_CRITERIA`, and the scorer then bills a criterion rubric.md does
    not define. The list is reported under policy.py because that is the
    file a candidate has to edit to break it.
    """

    out: list[str] = []
    freeze = list(freeze)
    for name in freeze:
        filename = FROZEN_FILES[name]
        if _read(Path(candidate_dir) / filename) != _read(Path(baseline_dir) / filename):
            out.append(filename)
    if "rubric" in freeze and "policy" not in freeze:
        declared = _declared_criteria(candidate_dir)
        # None on either side means the declaration could not be read at all,
        # which the loader and `rubric_criteria_defect` already refuse on their
        # own terms. Adding "RUBRIC_CRITERIA changed" on top would be a second,
        # misleading reason for one fault.
        if declared is not None:
            baseline_declared = _declared_criteria(Path(baseline_dir))
            if baseline_declared is not None and declared != baseline_declared:
                out.append("policy.py:RUBRIC_CRITERIA")
    return out


def _declared_criteria(filter_dir: Path) -> tuple[str, ...] | None:
    """`RUBRIC_CRITERIA` as declared, or None when it cannot be read."""

    try:
        from meta_filter.filter_loader import load_filter

        declared = getattr(load_filter(Path(filter_dir)).policy_module, "RUBRIC_CRITERIA", None)
    except Exception:
        return None
    if declared is None:
        return None
    try:
        return tuple(str(name) for name in declared)
    except TypeError:
        return None


# --- declared but unread criteria ---------------------------------------------
#
# The solver issues one LLM call per (row, criterion in `RUBRIC_CRITERIA`). A
# criterion declared there that `decide()` never reads is therefore bought on
# every row and spent on nothing: no arithmetic touches it, so it cannot move
# the ordering by a single pair. A candidate that drops a dead axis from its
# formula but leaves it declared keeps paying for it on every row.
#
# The detector is deliberately one-sided. Its job is to refuse to miss a
# generic read, never to catch every literal one: everything it cannot follow
# — a mapping that is iterated, summed, `.values()`-ed, keyed by a computed
# name, or enumerated through `RUBRIC_CRITERIA` — makes EVERY declared
# criterion count as read and the answer empty. A policy that takes a mean
# over the whole mapping must come back empty for exactly that reason
# (`tests/test_declared_unread_criteria.py`).

# Attributes that hand the whole mapping to the caller rather than one cell.
# `.get` is the one attribute that names a single criterion, so it is the one
# attribute that is not generic — and only when its key resolves to a literal.
_CELL_ATTR = "get"

# `decide(trajectory, rubric_scores, triggered_rules)` — the mapping is the
# parameter with this name, or failing that the second positional one.
MAPPING_PARAM = "rubric_scores"


def _policy_tree(filter_dir: Path) -> ast.Module | None:
    """policy.py parsed, or None when it is missing or does not parse."""

    path = Path(filter_dir) / "policy.py"
    if not path.is_file():
        return None
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None


def _module_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    out: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            out[node.target.id] = node.value
    return out


def _declared_in(tree: ast.Module | None) -> list[str]:
    """The string literals of the module-level `RUBRIC_CRITERIA` sequence.

    Empty when the name is absent or is built from anything but string
    literals — an unreadable declaration is one this check says nothing
    about.
    """

    if tree is None:
        return []
    value = _module_assignments(tree).get("RUBRIC_CRITERIA")
    if not isinstance(value, ast.List | ast.Tuple):
        return []
    names: list[str] = []
    for element in value.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            names.append(element.value)
        else:
            return []
    return names


def declared_criteria(filter_dir: Path) -> list[str]:
    """`RUBRIC_CRITERIA` as policy.py declares it, in declaration order."""

    return _declared_in(_policy_tree(Path(filter_dir)))


# What a candidate is told when its policy.py cannot name a criterion for the
# solver to score. One sentence per shape, shared by the two readers below so
# the proposer sees the same words whether the source or the imported value
# caught it.
NO_CRITERIA_REASON = "policy.py declares no RUBRIC_CRITERIA — the solver has nothing to score"
EMPTY_CRITERIA_REASON = (
    "policy.py declares an empty RUBRIC_CRITERIA — the solver has nothing to score"
)
BAD_CRITERIA_REASON = (
    "policy.py declares a RUBRIC_CRITERIA that is not a list of criterion-name "
    "strings — the solver has nothing to score"
)


def policy_reads_trajectory(filter_dir: Path) -> str | None:
    """Why this filter's policy.py is not a policy, read from the source.

    The interface note the proposer is given splits the three files by job:
    `rules.py` holds "deterministic rule callables", `policy.py` holds "how
    the continuous score is composed, and the bookkeeping decision derived
    from it". `decide()` is handed the trajectory so that a policy may cite a
    row, not so that it may grade one — and because `policy_loc` is the one
    component with no growth cap (a policy line costs no LLM call), a policy
    that grades is deterministic logic bought outside every budget.

    A candidate can arrive at an empty rules.py and a policy.py that parses
    `+++ b/` paths out of the diff, walks the assistant turns for `pytest` /
    `ruff` / `python -c` markers and scores the ORDER of edits against tests
    — many rule bodies' worth of transcript grading, never once charged
    against the rules budget or slowed by the gate the way a rule would be.

    Certain-only, like everything else in this module: the check is whether
    `decide` references its own `trajectory` parameter at all, passing it to
    a helper included. It does not chase a trajectory obtained some other
    way, and it does not read intent — a policy that takes the argument and
    ignores it (every shipped seed) is what "composes" looks like.

    Args:
        filter_dir: The candidate filter's directory.

    Returns:
        The reason, naming where the logic belongs, or None when policy.py
        composes and nothing more.
    """

    path = filter_dir / "policy.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "decide":
            continue
        args = node.args
        names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        if "trajectory" not in names:
            return None
        reads = sum(
            isinstance(inner, ast.Name) and inner.id == "trajectory" for inner in ast.walk(node)
        )
        if reads:
            return (
                f"policy.py reads the trajectory in decide() ({reads} references) — "
                "policy.py composes rubric scores and rule verdicts; deterministic "
                "signal from the transcript, diff or metadata belongs in rules.py "
                "as a rule_ function, where the rules budget applies to it"
            )
        return None
    return None


def rubric_criteria_defect(filter_dir: Path) -> str | None:
    """Why this filter's `RUBRIC_CRITERIA` cannot be scored, read from the source.

    `Solver.run` scores exactly the names in `RUBRIC_CRITERIA`; with none,
    every row gets the same constant out of `decide()` and the candidate
    ranks nothing. That is not a weak candidate, it is an unscoreable one,
    and it has to be caught where a proposer can still repair it.

    Certain-only, like everything else in this module: it fires for a
    declaration this reader can evaluate and finds unusable, and says
    nothing otherwise. A declaration built at import time
    (`RUBRIC_CRITERIA = list(WEIGHTS)`) or bound anywhere but the module's top
    level is left to `rubric_criteria_defect_of`, which sees the value the
    interpreter actually produced.
    """

    tree = _policy_tree(Path(filter_dir))
    if tree is None:
        return None  # no policy.py, or one that does not parse: the loader says so first
    value = _module_assignments(tree).get("RUBRIC_CRITERIA")
    if value is None:
        # `_module_assignments` reads top-level assignments only. A name bound
        # inside an `if` or a `try` is not a declaration this reader resolves,
        # but neither is it an absence — only a policy.py that never mentions
        # the name at all is certainly missing it.
        if any(isinstance(n, ast.Name) and n.id == "RUBRIC_CRITERIA" for n in ast.walk(tree)):
            return None
        return NO_CRITERIA_REASON
    if not isinstance(value, ast.List | ast.Tuple):
        return None  # built from something this reader cannot evaluate
    if not value.elts:
        return EMPTY_CRITERIA_REASON
    if any(isinstance(e, ast.Constant) and not isinstance(e.value, str) for e in value.elts):
        return BAD_CRITERIA_REASON
    return None


def rubric_criteria_defect_of(criteria: object) -> str | None:
    """The same defect, read off the imported module's value.

    The half `rubric_criteria_defect` cannot do from the source. A blank
    string is a criterion with no name — the scorer would look for it in
    rubric.md and find nothing — so it counts as unusable here.
    """

    if criteria is None:
        return NO_CRITERIA_REASON
    if not isinstance(criteria, list | tuple):
        return BAD_CRITERIA_REASON
    if not criteria:
        return EMPTY_CRITERIA_REASON
    if any(not isinstance(c, str) or not c.strip() for c in criteria):
        return BAD_CRITERIA_REASON
    return None


def record_criteria_defect(report: ConstraintReport, reason: str) -> ConstraintReport:
    """Fire `invalid` on an already-built report for an unscoreable declaration.

    The source-level check inside `check_candidate` runs before the
    candidate is imported, so a declaration only the interpreter can
    evaluate reaches `precheck_candidate` instead. This is where that second
    reading is folded back in, the way `assess_rubric_health` folds in what
    scoring revealed. Idempotent.
    """

    if reason and reason not in report.reasons:
        report.reasons.append(reason)
    if reason and _SEVERITY["invalid"] > _SEVERITY[report.status]:
        report.status = "invalid"
    return report


def record_rule_firing(report: ConstraintReport, reasons: Sequence[str]) -> ConstraintReport:
    """Fold what running the candidate's rules revealed into its report.

    The rules are pure Python, so unlike the rubric they can be measured for
    nothing — but only once the precheck holds the rows, which is after
    `check_candidate` has already built the report off the files. A rule that
    fires on none of those rows, or on all of them, orders nothing; that
    raises the bar the way a soft cap does and never invalidates, because a
    few dozen rows cannot prove what the pool does.

    Args:
        report: The report to fold into, modified in place.
        reasons: One sentence per firing, already phrased for the proposer.

    Returns:
        The same report. Idempotent: a repeated reason is not added twice.
    """

    fired = [reason for reason in reasons if reason]
    for reason in fired:
        if reason not in report.reasons:
            report.reasons.append(reason)
    if fired and _SEVERITY["needs_strong_evidence"] > _SEVERITY[report.status]:
        report.status = "needs_strong_evidence"
    return report


def _string_constants(node: ast.AST) -> set[str]:
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _parents(node: ast.AST) -> dict[int, ast.AST]:
    out: dict[int, ast.AST] = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            out[id(child)] = parent
    return out


def _positional_names(function: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in function.args.posonlyargs + function.args.args]


def _param_names(function: ast.FunctionDef) -> list[str]:
    """Every parameter, positional and keyword-only."""

    return _positional_names(function) + [arg.arg for arg in function.args.kwonlyargs]


def _mapping_param(decide: ast.FunctionDef) -> str | None:
    """The parameter carrying the rubric-score mapping, or None.

    The name is looked for first, among keyword-only parameters too: a
    mapping passed by keyword is still the mapping, and `rubric_scores` is
    what every policy this project has run calls it.

    The positional fallback is a GUESS, and a wrong guess is the worst answer
    this scan can give. Nothing would then be seen to consume the real
    mapping, so no read would look generic and every criterion whose name is
    not a literal would be reported unread. A guess is therefore only taken
    when `decide()` actually loads the name it guessed; otherwise this
    returns None and the caller says nothing at all.
    """

    if MAPPING_PARAM in _param_names(decide):
        return MAPPING_PARAM
    names = _positional_names(decide)
    if len(names) >= 2 and names[1] in _loaded_names(decide):
        return names[1]
    return None


def _reachable(
    entry: ast.FunctionDef, functions: dict[str, ast.FunctionDef]
) -> list[ast.FunctionDef]:
    """`entry` and every module-level function it calls, transitively."""

    seen = {entry.name}
    order = [entry]
    queue = [entry]
    while queue:
        current = queue.pop()
        called = sorted(
            child.func.id
            for child in ast.walk(current)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in functions
        )
        for name in called:
            if name not in seen:
                seen.add(name)
                order.append(functions[name])
                queue.append(functions[name])
    return order


def _module_literals(tree: ast.Module, referenced: Iterable[str]) -> set[str] | None:
    """String literals of the module constants the reachable code names.

    A weights dict keyed by criterion name lives at module level and is
    applied inside `decide()`; its keys are reads, and a scan of function
    bodies alone would call every one of them unread.

    None means "these cannot be enumerated", and a caller that gets None must
    say nothing at all. A module constant built from anything this walk does
    not follow — `AXES = _axes()`, `KEYS = frozenset(imported)` — has string
    contents the walk never sees; dropping such a reference silently would
    report a policy reading all six criteria through a helper keyed by that
    constant as six "unread" findings. Half
    enumerated is the one state this set may not be in, because it is only
    ever used to conclude that a criterion is NOT read.
    """

    assignments = _module_assignments(tree)
    out: set[str] = set()
    seen: set[str] = set()
    pending = [name for name in referenced if name in assignments]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        value = assignments[name]
        out |= _string_constants(value)
        for ref in _loaded_names(value):
            if ref not in assignments:
                return None
            if ref not in seen:
                pending.append(ref)
    # A constant the module mutates after assigning it is half enumerated,
    # which is the one state this set may not be in: the literals we walked
    # are real but the rest are invisible, so a key built from it would read
    # as covering fewer criteria than it does. Say nothing instead.
    if seen & _mutated(tree):
        return None

    return out


@dataclass(frozen=True)
class _PolicyScan:
    """What the mapping walk needs to know about the module around it.

    `module_names` are the module-level assignments, every one of which
    `_module_literals` has enumerated whole — the walk bails otherwise — so a
    value drawn from one of them can only be a string this scan has already
    collected.
    """

    functions: dict[str, ast.FunctionDef]
    module_names: frozenset[str]


def _tainted(node: ast.AST, names: set[str]) -> bool:
    """Whether `node` evaluates to the rubric-score mapping itself."""

    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load) and node.id in names
    if isinstance(node, ast.BoolOp):
        return any(_tainted(value, names) for value in node.values)
    if isinstance(node, ast.IfExp):
        return _tainted(node.body, names) or _tainted(node.orelse, names)
    if isinstance(node, ast.Starred):
        return _tainted(node.value, names)
    return False


# Methods that hand back something the container already held. Any other
# method may BUILD a string this scan never saw (`", ".join(parts)`,
# `name.upper()`), so calling one makes both the result and the receiver
# unenumerable.
_CONTAINER_READS = frozenset({"get", "items", "keys", "values", "copy"})


def _enumerated(node: ast.expr, module_names: frozenset[str], keys: set[str]) -> bool:
    """Whether every string this expression can evaluate to is one the scan
    already holds — a literal in the reachable code, or a module constant's.

    True for a literal, for a module constant, for a name already established
    as enumerable, and for indexing or reading a container that is itself
    enumerable. False for everything else: an import, a call this walk does
    not enter, a field of the trajectory, a concatenation. Those produce
    strings that never reached `literals`, and a criterion KEY drawn from one
    is a key this scan cannot resolve.
    """

    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in module_names or node.id in keys
    if isinstance(node, ast.Attribute | ast.Subscript | ast.Starred):
        return _enumerated(node.value, module_names, keys)
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return all(_enumerated(element, module_names, keys) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            _enumerated(element, module_names, keys)
            for element in [*node.keys, *node.values]
            if element is not None
        )
    if isinstance(node, ast.BoolOp):
        return all(_enumerated(value, module_names, keys) for value in node.values)
    if isinstance(node, ast.IfExp):
        return _enumerated(node.body, module_names, keys) and _enumerated(
            node.orelse, module_names, keys
        )
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _CONTAINER_READS
            and _enumerated(node.func.value, module_names, keys)
        )
    return False


def _mutated(function: ast.FunctionDef) -> set[str]:
    """Names the function can change in place, so no binding describes them.

    `parts = []` is enumerable and `parts.append(trajectory.task)` makes it
    anything at all; the binding on its own would say the opposite.
    """

    out: set[str] = set()
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr not in _CONTAINER_READS
            and isinstance(node.func.value, ast.Name)
        ):
            out.add(node.func.value.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
        ):
            out.add(node.value.id)
    return out


def _target_names(target: ast.expr) -> set[str]:
    return {node.id for node in ast.walk(target) if isinstance(node, ast.Name)}


def _bindings(function: ast.FunctionDef) -> dict[str, list[ast.expr]]:
    """Every name the function's OWN scope binds, and what it binds it from.

    Comprehension targets are deliberately absent: a comprehension has its
    own scope, so `[_num(rubric_scores, name) for name in WEIGHTS]` and a
    `name = rule.get("name")` in the same body are two different variables,
    and folding them together would let the rule name disqualify the weights
    key. `_comprehension_keys` binds those where they belong.

    A name bound anywhere this does not look — `with ... as`, `except ... as`,
    a nested `def` — simply never becomes enumerable, which is the safe end.
    """

    out: dict[str, list[ast.expr]] = {}

    def bind(target: ast.expr, value: ast.expr) -> None:
        for name in _target_names(target):
            out.setdefault(name, []).append(value)

    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(node.target, node.value)
        elif isinstance(node, ast.For | ast.AsyncFor):
            bind(node.target, node.iter)
        elif isinstance(node, ast.NamedExpr):
            bind(node.target, node.value)
    return out


# Every expression form that opens a scope of its own for its `for` targets.
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _comprehension_keys(
    node: ast.AST,
    parents: dict[int, ast.AST],
    module_names: frozenset[str],
    outer: set[str],
) -> set[str]:
    """`outer`, corrected for the comprehension scopes `node` sits inside.

    A target drawn from something enumerable is a key inside that
    comprehension and nowhere else; a target drawn from anything else hides
    an outer name of the same spelling for as long as the comprehension
    lasts. Written as a walk up the parent chain because that is the only
    place the enclosing comprehensions are known.
    """

    chain: list[ast.AST] = []
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, _COMPREHENSIONS):
            chain.append(current)
        current = parents.get(id(current))
    if not chain:
        return outer
    keys = set(outer)
    for comprehension in reversed(chain):
        for generator in comprehension.generators:
            targets = _target_names(generator.target)
            if _enumerated(generator.iter, module_names, keys):
                keys |= targets
            else:
                keys -= targets
    return keys


def _enumerable_keys(
    function: ast.FunctionDef, module_names: frozenset[str], inherited: set[str]
) -> set[str]:
    """The names inside `function` that may be trusted as a criterion key.

    `inherited` are the parameters the call site bound to something already
    enumerable — a literal, or one of its own enumerable names. To those this
    adds the locals whose every binding is enumerable and which are never
    mutated: the weights-dict loop (`for name, weight in WEIGHTS.items()`)
    and the local lookup table (`check = confirmation_axis.get(rule)`) are
    both that shape, and both read criteria whose names are in `literals`.

    A key that is NOT one of these makes the read generic, which empties the
    whole finding. That is the intended failure: `for name in AXES` over an
    imported list reads criteria this walk cannot name, and trusting it is
    what reported all six of them unread.
    """

    mutated = _mutated(function)
    bindings = _bindings(function)
    keys = {name for name in inherited if name not in mutated}
    for _ in range(4):
        grew = False
        for name, values in bindings.items():
            if name in keys or name in mutated:
                continue
            if all(_enumerated(value, module_names, keys) for value in values):
                keys.add(name)
                grew = True
        if not grew:
            break
    return keys


def _resolvable_key(key: ast.expr, keys: set[str]) -> bool:
    """Whether a subscript / `.get` key names criteria this scan has seen.

    A string literal does. So does a name in `keys`: a helper parameter the
    call site fed something enumerable, or a local drawn from a module
    constant this scan enumerated whole (`_enumerable_keys`). Anything else —
    a loop variable over an import, a computed name, an attribute, an
    expression — is a key whose values never reached `literals`, and an
    unresolvable key is a generic read.
    """

    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return True
    return isinstance(key, ast.Name) and key.id in keys


def _generic_use(
    node: ast.AST,
    parents: dict[int, ast.AST],
    keys: set[str],
    scan: _PolicyScan,
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> bool:
    """Whether this one use of the mapping consumes it generically."""

    parent = parents.get(id(node))
    if parent is None:
        return True
    keys = _comprehension_keys(node, parents, scan.module_names, keys)
    if isinstance(parent, ast.Assign) and parent.value is node:
        return False  # an alias; the taint follows it
    if isinstance(parent, ast.Compare):
        return False  # `"X" in rubric_scores`, `rubric_scores == {}`
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return not _resolvable_key(parent.slice, keys)
    if isinstance(parent, ast.Attribute) and parent.value is node:
        if parent.attr != _CELL_ATTR:
            return True  # .values(), .items(), .keys(), .copy(), anything
        call = parents.get(id(parent))
        if not (isinstance(call, ast.Call) and call.func is parent and call.args):
            return True
        return not _resolvable_key(call.args[0], keys)
    if isinstance(parent, ast.Call):
        return _generic_call(node, parent, keys, scan, seen)
    if isinstance(parent, ast.keyword):
        # `_criterion(rubric_scores=scores, name="X")` — the keyword node sits
        # between the mapping and its call, so resolve one level up.
        call = parents.get(id(parent))
        return not isinstance(call, ast.Call) or _generic_call(node, call, keys, scan, seen)
    # for/comprehension iteration, sum(), len(), a return, an f-string — every
    # remaining shape hands on more than one cell.
    return True


def _generic_call(
    node: ast.AST,
    call: ast.Call,
    keys: set[str],
    scan: _PolicyScan,
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> bool:
    """The mapping passed as an argument: follow a helper, else give up.

    Two things are carried across the call: which of the helper's parameters
    hold the mapping, and which hold something enumerable. The second is what
    lets `_criterion(rubric_scores, name)` resolve its `name` — and what
    stops it resolving when the caller's `name` came from somewhere this scan
    never read.
    """

    if not isinstance(call.func, ast.Name) or call.func.id not in scan.functions:
        return True  # sum(), dict(), statistics.mean(), anything imported
    target = scan.functions[call.func.id]
    names = _positional_names(target)
    tainted: set[str] = set()
    passed: set[str] = set()
    for index, arg in enumerate(call.args):
        if index < len(names) and not isinstance(arg, ast.Starred):
            if _enumerated(arg, scan.module_names, keys):
                passed.add(names[index])
        if arg is node:
            if isinstance(arg, ast.Starred) or index >= len(names):
                return True  # `helper(*rubric_scores)` unpacks the keys
            tainted.add(names[index])
    for keyword in call.keywords:
        if keyword.arg is not None and _enumerated(keyword.value, scan.module_names, keys):
            passed.add(keyword.arg)
        if keyword.value is node:
            if keyword.arg is None:
                return True  # **kwargs
            tainted.add(keyword.arg)
    if not tainted:
        return True
    return _consumes_generically(scan, call.func.id, tainted, passed, seen)


def _consumes_generically(
    scan: _PolicyScan,
    name: str,
    tainted: set[str],
    passed: set[str],
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> bool:
    """Whether `name`, called with the mapping bound to `tainted`, reads it whole."""

    key = (name, tuple(sorted(tainted)), tuple(sorted(passed)))
    if key in seen:
        return False
    seen.add(key)
    function = scan.functions[name]
    keys = _enumerable_keys(function, scan.module_names, passed)
    parents = _parents(function)
    names = set(tainted)
    # `scores = rubric_scores or {}` aliases the mapping; follow it to a
    # fixpoint so the alias is classified like the parameter.
    for _ in range(4):
        grew = False
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and _tainted(node.value, names):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in names:
                        names.add(target.id)
                        grew = True
        if not grew:
            break
    for node in ast.walk(function):
        if not _tainted(node, names):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.BoolOp | ast.IfExp) and _tainted(parent, names):
            continue  # classified at the enclosing expression instead
        if _generic_use(node, parents, keys, scan, seen):
            return True
    return False


def declared_unread_criteria(filter_dir: Path) -> list[str]:
    """Criteria policy.py DECLARES in `RUBRIC_CRITERIA` and `decide()` never reads.

    In declaration order, so the list reads against rubric.md. A criterion
    counts as READ when its name appears as a string literal anywhere in
    `decide()` or a module-level helper it calls, or in a module constant
    that code names (a weights dict keyed by criterion). That is generous on
    purpose: an unread criterion is a real cost — one scorer call per row,
    for nothing — but calling a read criterion unread would push a proposer
    into deleting an axis the policy uses.

    Empty, meaning "nothing to say", whenever the scan cannot follow the
    policy: no policy.py, a syntax error, no `decide()`, no parameter that
    can be shown to carry the mapping, a `RUBRIC_CRITERIA` that is not a list
    of literals, a module constant whose contents this walk cannot enumerate
    (`_module_literals`), or any GENERIC consumption of the mapping —
    iterating it, `.values()`/`.items()`/`.keys()`, handing it to a call this
    scan cannot enter (`sum`, `dict`, an import), keying it with something
    that does not resolve to names the scan has read, or naming
    `RUBRIC_CRITERIA` inside the reachable code.
    """

    filter_dir = Path(filter_dir)
    tree = _policy_tree(filter_dir)
    declared = _declared_in(tree)
    if tree is None or not declared:
        return []
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    decide = functions.get("decide")
    if decide is None:
        return []
    mapping = _mapping_param(decide)
    if mapping is None:
        return []

    literals: set[str] = set()
    referenced: set[str] = set()
    for function in _reachable(decide, functions):
        literals |= _string_constants(function)
        referenced |= _loaded_names(function)
    if "RUBRIC_CRITERIA" in referenced:
        return []  # the policy enumerates its own declaration
    module_literals = _module_literals(tree, referenced)
    if module_literals is None:
        return []  # a module constant this walk cannot read to the bottom
    literals |= module_literals

    scan = _PolicyScan(functions=functions, module_names=frozenset(_module_assignments(tree)))
    if _consumes_generically(scan, "decide", {mapping}, set(), set()):
        return []
    return [name for name in declared if name not in literals]


def _numeric(node: ast.expr) -> float | None:
    """A weight literal as a float, or None when this is not one.

    `ast.UnaryOp` is here because a negative weight is not an `ast.Constant`
    — `-1.0` parses as `USub` over `1.0`. Reading only `Constant` made a
    single negative entry hide the WHOLE table, and negative weights are
    ordinary in these policies.
    """

    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return None if isinstance(node.value, bool) else float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        inner = _numeric(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _weight_tables(tree: ast.Module, declared: Iterable[str]) -> list[dict[str, float]]:
    """Module-level dict literals that price at least one declared criterion.

    One unreadable entry skips that ENTRY, not the table: a weights dict with
    a computed value for one criterion still prices the others, and dropping
    the whole table on the first expression made the scan silent on 935 of
    this repo's 1,211 filter directories.
    """

    names = set(declared)
    tables = []
    for value in _module_assignments(tree).values():
        if not isinstance(value, ast.Dict):
            continue
        priced: dict[str, float] = {}
        for key, item in zip(value.keys, value.values, strict=False):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            number = _numeric(item)
            if number is not None:
                priced[key.value] = number
        if priced.keys() & names:
            tables.append(priced)
    return tables


def _named_outside_tables(tree: ast.Module) -> set[str]:
    """Criterion-shaped string literals appearing outside the priced tables.

    A criterion the policy names ANYWHERE else — a tuple the floor reads, a
    branch, a second mapping — is not merely a weighted term, so a weight of
    zero is a DEMOTION rather than a defect. A policy can move a criterion
    out of the weighted sum and leave it in, say, an `EXECUTION_CRITERIA`
    tuple a score floor reads. Calling that inert, and telling the proposer
    to delete it, produces a policy that raises `KeyError` on the tuple the
    floor still reads.
    """

    inside = set()
    for value in _module_assignments(tree).values():
        if isinstance(value, ast.Dict):
            for key in value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    inside.add(id(key))
    # Only `RUBRIC_CRITERIA` is a declaration. Every OTHER module-level
    # sequence of criterion names is a use — excluding bare lists wholesale
    # would put a criterion demoted into one back in the defect set.
    declaration = _module_assignments(tree).get("RUBRIC_CRITERIA")
    if isinstance(declaration, ast.List | ast.Tuple):
        for element in declaration.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                inside.add(id(element))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in inside
    }


def priced_criteria(filter_dir: Path) -> set[str]:
    """Declared criteria some readable module-level table gives a number to.

    The coverage predicate. A criterion outside it is one this scan cannot
    price — not one priced at zero — and the two must never be confused: the
    `invalid` fires off the candidate's scan and the floor credit off the
    baseline's, so a baseline the reader cannot price has to silence BOTH or
    it charges a candidate for a defect it inherited and then refuses the
    removal the reason tells it to make.
    """

    tree = _policy_tree(Path(filter_dir))
    declared = _declared_in(tree)
    if tree is None or not declared:
        return set()
    priced: set[str] = set()
    for table in _weight_tables(tree, declared):
        priced |= table.keys() & set(declared)
    return priced


def zero_weight_criteria(filter_dir: Path) -> list[str]:
    """Criteria policy.py declares, prices at exactly 0, and reads nowhere else.

    The fourth health finding, and the one that had no check. A criterion at
    weight zero is not UNREAD (the policy names it, so
    `declared_unread_criteria` says nothing) and not DEAD (its scores can
    spread across every level), yet the solver buys one LLM call per row for
    a term the aggregation multiplies by nothing, and without this check a
    run spends a whole iteration measuring the rank lift before anything can
    name it. This is static: it fires on the candidate that writes it, in the
    precheck, before a single row is scored.

    Three ways to be silent, all deliberate, all in the module's
    certain-only direction:

      * the criterion carries a non-zero number in some readable table —
        including a table that is not the weights, where 0 in one and 3.0 in
        another is ambiguous and says nothing;
      * the scan cannot price it at all (`priced_criteria`);
      * policy.py NAMES it outside the tables (`_named_outside_tables`).
        That last one is a demotion, not a defect: the axis has been moved
        from the weighted sum to a floor or a gate and still moves the score,
        and both "remove it" and "give it weight" would undo a legitimate
        mechanism — the first by raising `KeyError` in the code that still
        reads it.
    """

    tree = _policy_tree(Path(filter_dir))
    declared = _declared_in(tree)
    if tree is None or not declared:
        return []

    tables = _weight_tables(tree, declared)
    elsewhere = _named_outside_tables(tree)
    values: dict[str, list[float]] = {}
    for table in tables:
        for name in declared:
            if name in table:
                values.setdefault(name, []).append(table[name])

    return [
        n
        for n in declared
        if values.get(n) and all(v == 0.0 for v in values[n]) and n not in elsewhere
    ]


def newly_zero_weight_criteria(candidate_dir: Path, baseline_dir: Path) -> list[str]:
    """Criteria this candidate put at weight zero that its baseline did not.

    Inheriting one fires nothing, the way an inherited unread criterion and
    an inherited dead axis do not: the run should not refuse to score every
    candidate because the filter they were all edited from carries a defect.
    Writing one is `invalid` — it is the move a proposer makes when the
    deletion floor leaves it nothing else, and the floor credits a measured
    zero-lift removal back, so the proper exit is open.

    Restricted to criteria the BASELINE scan could actually price. Without
    that restriction the fire and the escape hatch read different scans: any
    baseline the reader cannot price (a computed weights table, or
    `RUBRIC_CRITERIA = list(WEIGHTS)`) makes
    an INHERITED zero look new, charges every candidate for it at once, and
    then refuses the deletion the reason names because the floor credit reads
    the same silent baseline. That is an emptied iteration, from an edit
    nobody made.
    """

    baseline_priced = priced_criteria(Path(baseline_dir))
    baseline_zero = set(zero_weight_criteria(Path(baseline_dir)))
    return [
        n
        for n in zero_weight_criteria(Path(candidate_dir))
        if n in baseline_priced and n not in baseline_zero
    ]


def newly_unread_criteria(
    candidate_dir: Path,
    baseline_dir: Path,
    *,
    candidate_unread: Sequence[str] | None = None,
) -> list[str]:
    """Criteria the BASELINE's policy reads that the candidate's declares and does not.

    Only an introduced mismatch counts. A candidate that inherits an unread
    criterion fires nothing: it did not create the defect, and in a run whose
    initial filter already carries two dead axes, charging the inheritance
    would raise the bar for every candidate in the run for something none of
    them did. A criterion the baseline did not declare at all is likewise not
    an introduction by this rule — the reason string says "no longer reads",
    and it means it.
    """

    unread = (
        list(candidate_unread)
        if candidate_unread is not None
        else declared_unread_criteria(candidate_dir)
    )
    if not unread:
        return []
    baseline_declared = set(declared_criteria(baseline_dir))
    baseline_unread = set(declared_unread_criteria(baseline_dir))
    return [name for name in unread if name in baseline_declared and name not in baseline_unread]


def complexity_score(breakdown: Mapping[str, int]) -> float:
    """The scalar `measure_filter_complexity` would give this breakdown.

    Re-derived from the breakdown rather than re-measured so the total and
    the per-component test below read the same numbers, and so a caller that
    already has a `ConstraintReport` needs no second pass over the files.
    Default coefficients only: the simplification rule is not a weight to
    tune, and a run that re-weighted complexity would change which edits
    count as smaller without changing anything the gate measures.
    """

    return sum(
        DEFAULT_COMPLEXITY_COEFFICIENTS.get(name, 0.0) * int(breakdown.get(name, 0))
        for name in COMPONENTS
    )


def is_simplification(candidate: Mapping[str, int], baseline: Mapping[str, int]) -> bool:
    """Whether the candidate is smaller than the baseline, with no trade.

    Both halves are required: the weighted total is strictly lower AND no
    single component is higher. The total alone would let a candidate buy
    300 rubric characters with a deleted rule and still call itself a
    simplification, which is the confounded edit the one-mechanism rule
    exists to refuse; the per-component half alone would admit a candidate
    identical to its baseline. Together they mean exactly "this filter does
    less, in every way it is measured".

    This is deliberately a structural test over the complexity breakdown and
    nothing else — no rubric statistics, no diff shape. A narrow version
    (dead axes and redundant merges, recognised by a byte-level diff test)
    would almost never fire: a proposer cannot aim at a lane it has to hit
    exactly. Pruning a rule that never
    fires, folding two criteria into one, dropping an axis the scorer reads
    the same way on every row — all of them are one thing here, and the gate
    asks the only question that matters about them (`b >= c`, did the hidden
    pairs notice).
    """

    if any(int(candidate.get(name, 0)) > int(baseline.get(name, 0)) for name in COMPONENTS):
        return False
    return complexity_score(candidate) < complexity_score(baseline)


def rubric_defects(
    candidate_stats: Mapping | None, baseline_stats: Mapping | None
) -> tuple[list[str], list[list[str]]]:
    """Dead axes and redundant pairs the candidate has and the baseline did not.

    A dead axis here is `>= NEW_DEAD_SHARE` of rows at one level over at
    least `NEW_DEAD_MIN_ROWS` rows. A baseline axis already at or above the
    health block's `DEAD_MODAL_SHARE` is inherited and never counts, and so
    is a pair correlated in the baseline. Without candidate stats there is
    nothing to report; without baseline stats every defect is new.
    """

    if not candidate_stats:
        return [], []
    base_per = (baseline_stats or {}).get("per_criterion") or {}
    new_dead: list[str] = []
    if int(candidate_stats.get("n_rows", 0)) >= NEW_DEAD_MIN_ROWS:
        for name, pc in sorted((candidate_stats.get("per_criterion") or {}).items()):
            if float(pc.get("modal_share", 0.0)) < NEW_DEAD_SHARE:
                continue
            inherited = base_per.get(name)
            if inherited is not None and inherited.get("dead"):
                continue
            new_dead.append(name)
    inherited_pairs = redundant_pairs(baseline_stats)
    new_pairs = sorted(
        sorted(pair) for pair in redundant_pairs(candidate_stats) if pair not in inherited_pairs
    )
    return new_dead, new_pairs


def assess_rubric_health(
    report: ConstraintReport,
    candidate_stats: Mapping | None,
    baseline_stats: Mapping | None,
) -> ConstraintReport:
    """Fold the candidate's own rubric statistics into its constraint report.

    Called once the candidate has been scored (the precheck's judged rows,
    or the full pass): a NEW dead axis or a NEW redundant pair raises the
    status to `needs_strong_evidence` with a reason the proposer can read,
    exactly as a soft cap does. Inherited defects are the baseline's and
    fire nothing. Idempotent — a second call with the same stats adds no
    duplicate reason.

    `report.dead_axes` is set here too, and unlike `new_dead_axes` it lists
    EVERY dead axis the candidate has: the report is the readout the
    precheck block prints, beside `declared_unread`, and the two together
    are what the block calls inert.
    """

    new_dead, new_pairs = rubric_defects(candidate_stats, baseline_stats)
    report.new_dead_axes = new_dead
    report.new_redundant_pairs = new_pairs
    # Every dead axis, not only the ones this candidate introduced: the
    # report is also the readout the precheck block prints, and an inherited
    # dead axis is the thing the proposer most needs named.
    report.dead_axes = dead_axes(candidate_stats)
    per = (candidate_stats or {}).get("per_criterion") or {}
    reasons: list[str] = []
    for name in new_dead:
        pc = per.get(name) or {}
        reasons.append(
            f"new dead axis {name} ({pc.get('modal_share', 0.0):.0%} at "
            f"{float(pc.get('modal_value', 0.0)):+g} over {candidate_stats['n_rows']} rows)"
        )
    live = list((candidate_stats or {}).get("live_criteria") or [])
    matrix = (candidate_stats or {}).get("abs_corr") or []
    for a, b in new_pairs:
        rho = ""
        if a in live and b in live and matrix:
            rho = f" ({float(matrix[live.index(a)][live.index(b)]):.2f})"
        reasons.append(f"new |rho| >= {REDUNDANT_RHO:.2f} pair {a} vs {b}{rho}")
    for reason in reasons:
        if reason not in report.reasons:
            report.reasons.append(reason)
    if reasons and _SEVERITY["needs_strong_evidence"] > _SEVERITY[report.status]:
        report.status = "needs_strong_evidence"
    return report


def parse_complexity_ceiling(spec: str | None) -> dict[str, int]:
    """`--complexity-ceiling rubric_criteria=9,rule_count=13,...` -> `{name: cap}`.

    Each value is the largest count a candidate may reach for that component
    over the whole run. Names are `COMPONENTS` minus the exempt `policy_loc`;
    a component left out has no hard cap. Raises `ValueError` naming the bad
    token, so a typo is refused at boot rather than silently uncapped.
    """

    caps: dict[str, int] = {}
    exempt = ConstraintLimits.DEFAULT.hard_exempt
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        name, sep, value = token.partition("=")
        name = name.strip()
        if not sep or name not in COMPONENTS or name in exempt:
            allowed = ", ".join(c for c in COMPONENTS if c not in exempt)
            raise ValueError(
                f"--complexity-ceiling takes NAME=INT with NAME in {allowed}; got {token!r}"
            )
        try:
            cap = int(value)
        except ValueError:
            raise ValueError(
                f"--complexity-ceiling {name} needs an integer, got {value!r}"
            ) from None
        if cap < 0:
            raise ValueError(f"--complexity-ceiling {name} must be non-negative, got {cap}")
        caps[name] = cap
    return caps


def _breakdown(filter_dir: Path) -> dict[str, int]:
    breakdown, _ = measure_filter_complexity(filter_dir)
    return {name: int(breakdown.get(name, 0)) for name in COMPONENTS}


def check_candidate(
    candidate_dir: Path,
    *,
    baseline_dir: Path,
    initial_dir: Path,
    limits: ConstraintLimits = ConstraintLimits.DEFAULT,
    freeze: Iterable[str] = (),
    zero_lift_criteria: Iterable[str] = (),
    zero_lift_rules: Iterable[str] = (),
    hard_caps: Mapping[str, int] | None = None,
) -> ConstraintReport:
    """Measure one candidate against the soft caps, hard caps and floor.

    Reads files only — no import of the candidate's code, no LLM — so it is
    safe to run on a candidate that does not load, and the precheck runs it
    on every candidate before deciding which ones are worth scoring.

    Precedence is worst-of: a policy.py with no scoreable `RUBRIC_CRITERIA`
    (`rubric_criteria_defect`), any hard-cap or floor breach, a rationale
    with two or more hypotheses, an outcome field named in rules.py,
    policy.py or rubric.md, a criterion the baseline read that
    this candidate declares and no longer reads (`newly_unread_criteria`),
    or a changed file the run froze (`freeze`, see `FROZEN_FILES`), is
    `invalid`; otherwise any soft-cap breach is `needs_strong_evidence`;
    otherwise `ok`. Every firing is listed, not just the first, so a
    proposer sees the whole bill in one revision.

    The `RUBRIC_CRITERIA` check is unconditional because the declaration is
    what the solver scores.

    `declared_unread` is filled on every report whether or not it fired, so
    the precheck block can name an INHERITED unread criterion without the
    candidate being charged for it.

    A `simplification` (`is_simplification`) is recorded but never
    fires a status of its own: it is a licence the acceptance gate reads,
    not a breach. `freeze` is checked first and wins over it — a rubric-only
    run that let a candidate rewrite the frozen `policy.py` because the
    filter got smaller would not be a rubric-only run.

    `hard_caps` (`--complexity-ceiling`) replaces the RUN HARD CAP, and only
    it, with explicit numbers. The default cap is a ratio of the initial
    filter, so a run that starts from a small filter is otherwise held to a
    small multiple of small forever: a one-criterion seed puts its ceiling
    at 1.5 and makes the most ordinary proposal there is — add a criterion —
    `invalid` in iteration 0. Explicit caps give runs from different seeds
    the same ABSOLUTE room and leave the seed as the only difference between
    them. The floor is deliberately not moved with it: a blank seed asked to
    keep criteria it never had could not produce a candidate at all.
    """

    candidate_dir = Path(candidate_dir)
    candidate = _breakdown(candidate_dir)
    baseline = _breakdown(Path(baseline_dir))
    initial = _breakdown(Path(initial_dir))

    reasons: list[str] = []
    status: ConstraintStatus = "ok"

    def fire(level: ConstraintStatus, reason: str) -> None:
        nonlocal status
        reasons.append(reason)
        if _SEVERITY[level] > _SEVERITY[status]:
            status = level

    # First, because it is the one breach that makes every number below it
    # meaningless: a candidate the solver cannot score has no concordance to
    # weigh a cap against.
    criteria_defect = rubric_criteria_defect(candidate_dir)
    if criteria_defect is not None:
        fire("invalid", criteria_defect)

    policy_defect = policy_reads_trajectory(candidate_dir)
    if policy_defect is not None:
        fire("invalid", policy_defect)

    for name in COMPONENTS:
        value, base, init = candidate[name], baseline[name], initial[name]

        # Per-iteration soft cap, against the parent.
        allowance = limits.soft_add.get(name)
        soft_max = limits.soft_max_for(name, base)
        if soft_max is not None and value > soft_max:
            # Integer arithmetic throughout (`soft_cap`): `base * 1.15` is
            # 3449.9999999999995 for 3000, so the float form fired at exactly
            # the cap the prompt promised. The reason prints the cap rather
            # than the rule, because a share and a floor do not read the same.
            if name in _ABSOLUTE_COMPONENTS:
                fire(
                    "needs_strong_evidence",
                    f"{name} {value} > baseline {base} + {allowance:g} (soft)",
                )
            else:
                growth = f", +{(value - base) / base:.0%}" if base else ""
                # The share is what binds in every ordinary case, and its
                # wording is what the proposer prompt has always printed; the
                # floor only speaks up when it is the thing that bound.
                rule = (
                    f"{allowance:.0%}"
                    if soft_max == soft_cap(base, allowance)
                    else f"{limits.soft_floor.get(name, 0)} lines, the floor under {allowance:.0%}"
                )
                fire(
                    "needs_strong_evidence",
                    f"{name} {value} > baseline {base} + {rule} (soft{growth})",
                )

        # Run hard cap, against the initial filter. A component the initial
        # filter did not have at all (0) has no meaningful multiple, and an
        # exempt one (policy_loc) has no cap. `hard_caps`
        # (`--complexity-ceiling`) states the cap as a number instead of a
        # multiple of the initial filter's size; for integer counts
        # `value > cap` with `cap = floor(1.5 x size)` is exactly
        # `value > 1.5 x size`, so the two spellings refuse the same
        # candidates.
        if hard_caps:
            cap = hard_caps.get(name)
            if name not in limits.hard_exempt and cap is not None and value > cap:
                fire("invalid", f"{name} {value} > run cap {cap} (hard)")
        else:
            ceil = initial[name]
            if name not in limits.hard_exempt and ceil > 0 and value > limits.hard_ratio * ceil:
                fire("invalid", f"{name} {value} > {limits.hard_ratio:g} x initial {ceil} (hard)")

        # Deletion floor, against the initial filter. The reason names the
        # arithmetic that bound: `initial - allowance`, or the absolute
        # minimum when the initial filter was too small for the allowance.
        floor = limits.floor_for(name, init)
        if floor is not None and _inert_removals(
            name, Path(baseline_dir), candidate_dir, zero_lift_criteria, zero_lift_rules
        ):
            # A criterion the harness MEASURED as moving the ordering by
            # exactly zero is not discriminating power the floor is there to
            # protect — it is a scorer call per row buying nothing. Removing
            # one is credited back, so the floor binds on axes that carry
            # something.
            #
            # Without the credit, a run at the floor that is told which axis
            # contributes least cannot delete it, so it sets the weight to
            # 0.0 instead — and ends carrying a criterion that varies, is
            # read, changes no pair's ordering, and still costs a scorer call
            # per row. The floor would turn the correct repair into a half
            # one.
            #
            # Not farmable: `zero_lift_criteria` comes from ablating the
            # BASELINE and re-ranking, so it is the harness's measurement of
            # the filter the candidate was edited from, not anything the
            # candidate declares. Zeroing an axis's weight to earn a deletion
            # slot costs whatever that axis was contributing, which is the
            # trade we want made.
            #
            # Rules get the same credit as criteria. The finding names both
            # kinds in one sentence and the action asks for both, so
            # crediting only criteria would write a cheque the rule floor
            # bounces — and since the removal rides in EVERY candidate, that
            # bounce would invalidate a whole iteration.
            floor = max(
                0,
                floor
                - _inert_removals(
                    name, Path(baseline_dir), candidate_dir, zero_lift_criteria, zero_lift_rules
                ),
            )
        if floor is not None and value < floor:
            allowance = limits.deletion_floor[name]
            if floor == max(0, init - allowance):
                fire("invalid", f"{name} {value} < initial {init} - {allowance} (deletion floor)")
            else:
                fire("invalid", f"{name} {value} < {floor} (deletion floor minimum)")

    baseline_names = _named_units(Path(baseline_dir))
    candidate_names = _named_units(candidate_dir)
    deleted = [n for n in baseline_names if n not in candidate_names]
    added = [n for n in candidate_names if n not in baseline_names]
    # The one-deletion-per-iteration slot exists so a removal that hides
    # damage is taken one at a time and attributed. A removal the harness
    # MEASURED at exactly zero rank shift hides nothing by construction, so
    # it does not consume the slot — the same reasoning that already exempts
    # a `simplification`. Without this, "fold the zero-lift removal into
    # every candidate" makes every candidate a deletion candidate and
    # `gate_candidates` holds all but one back UNSCORED, which costs two
    # slots where the bug it fixes cost one. `deleted` stays the full truth
    # for the record; only the flag the slot reads is narrowed.
    _inert = set(zero_lift_criteria) | set(zero_lift_rules)
    deleted_inert = [n for n in deleted if n in _inert]
    deleted_live = [n for n in deleted if n not in _inert]

    mechanism_count = count_mechanisms(candidate_dir / "rationale.md")
    if mechanism_count is None:
        logger.info("constraints %s: no rationale.md, mechanism count unknown", candidate_dir.name)
    elif mechanism_count == 0:
        logger.info("constraints %s: rationale declares no Hypothesis: line", candidate_dir.name)
    elif mechanism_count >= 2:
        fire(
            "invalid",
            f"rationale declares {mechanism_count} hypotheses; one mechanism per candidate",
        )

    # The rubric included and no initial-filter exemption: the rows carry no
    # outcome field, so any read of one is either dead code or an attempt to
    # recover what the strip removed.
    from meta_filter.metadata_free import filter_outcome_reads

    for filename, names in filter_outcome_reads(candidate_dir).items():
        fire(
            "invalid",
            f"{filename} reads outcome metadata ({', '.join(names)}) — this run is "
            "metadata-free and its rows carry no such field",
        )

    # A declared criterion the policy stopped reading is a scorer call per
    # row that buys nothing. Only the candidate that stops reading it pays:
    # inheriting one fires nothing (`newly_unread_criteria`), it is merely
    # reported.
    zero_weight = zero_weight_criteria(candidate_dir)
    for name in newly_zero_weight_criteria(candidate_dir, Path(baseline_dir)):
        fire(
            "invalid",
            f"rubric_criteria declares {name!r} at weight 0 — one scorer call on "
            "every row for a term the weighted sum multiplies by nothing. Remove "
            "the criterion, or give it weight. The deletion floor does not protect "
            "a criterion the current filter already declares at zero",
        )

    declared_unread = declared_unread_criteria(candidate_dir)
    for name in newly_unread_criteria(
        candidate_dir, Path(baseline_dir), candidate_unread=declared_unread
    ):
        fire(
            "invalid",
            f"policy.py no longer reads declared criterion {name} — remove it "
            "from RUBRIC_CRITERIA and rubric.md, or read it",
        )

    simplification = is_simplification(candidate, baseline)
    frozen = frozen_changed(candidate_dir, Path(baseline_dir), freeze)
    for filename in frozen:
        fire("invalid", f"frozen file changed: {filename}")

    report = ConstraintReport(
        status=status,
        reasons=reasons,
        deletion=bool(deleted_live),
        deleted=deleted,
        deleted_inert=deleted_inert,
        added=added,
        mechanism_count=mechanism_count,
        candidate=candidate,
        baseline=baseline,
        initial=initial,
        simplification=simplification,
        frozen_changed=frozen,
        declared_unread=declared_unread,
        zero_weight=zero_weight,
    )
    if simplification:
        logger.info(
            "constraints %s: simplification — complexity %.2f < baseline %.2f, no component up",
            candidate_dir.name,
            complexity_score(candidate),
            complexity_score(baseline),
        )
    if status != "ok":
        logger.log(
            logging.WARNING if status == "invalid" else logging.INFO,
            "constraints %s: %s",
            candidate_dir.name,
            report.summary(),
        )
    return report
