"""Per-criterion health diagnostics computed from already-paid rubric scores.

Two failure shapes motivate this module:

* redundancy: the objective exerts no pressure against criteria that
  re-measure an existing axis, so a rubric's effective rank can fall while
  agreement rises, yet duplicating a live axis hurts the aggregate.
  ``effective_rank`` and ``max_abs_corr`` make that visible per candidate
  and feed the reported redundancy penalty.
* dead axes: a criterion whose scores pile onto one anchor orders nothing
  and only costs calls; its removal is a rank no-op. ``per_criterion`` modal
  shares expose that to the proposer.

Everything here reuses the FilterDecision.rubric_scores that the solver
already produced — no new I/O, no LLM calls. Spearman uses tie-averaged
ranks: with integer anchors ties dominate, and index-order ranking hands
any two near-constant axes a spurious rho of ~1.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# An axis with at least this share of its scores on one value is "dead":
# it cannot order the pool, whatever its prose says.
DEAD_MODAL_SHARE = 0.90

# Two axes at or above this |rho| are printed with a `!` in the correlation
# matrix and are what `constraints.rubric_defects` calls a redundant pair. It is
# the run default of `--redundancy-threshold`, restated here so the matrix
# marks the same pairs the penalty would charge.
REDUNDANT_RHO = 0.70

# The criterion-health block's size ceiling, in characters. The block carries
# the full |rho| matrix, because a proposer shown only the single
# most-correlated pair is told about one of two redundant pairs. The matrix is
# paid for by trimming the per-level distribution lines, live axes first.
#
# The ceiling leaves room for the inert-axis lines, which are the only lines
# in this block that state what to DO. Paying for them out of the matrix would
# trade the diagnosis for the instruction; the ceiling is a prompt-size guard,
# not a scarce resource.
HEALTH_BLOCK_BUDGET = 2400

# What an inert axis — dead, or declared in RUBRIC_CRITERIA and never read by
# `decide()` — costs. Identical for both, which is why they are named
# together.
INERT_AXIS_COST = "costs one scorer call per row and contributes nothing to the ordering"

# The repair, which is NOT the same for the two. A dead axis produces one
# level, so its anchors are what is broken; rewriting them is a repair a
# proposer can actually make.
DEAD_AXIS_ACTION = (
    "rewrite its anchors so they separate these trajectories (for example, "
    "judge the defect from the issue, the diff and the observed checks when a "
    "reference the criterion asks for is absent)"
)

# A declared-but-unread axis is the opposite case. Its scores are usually
# fine — spread over several levels — and rewriting them changes nothing,
# because `decide()` never looks. The cheap repair is to look: the cell is already
# bought, and `policy_loc` is in neither growth cap (`ConstraintLimits`), so
# the line that reads it is free.
UNREAD_AXIS_ACTION = (
    "read it in decide() — the cell is already bought and policy.py LOC is "
    "not capped, so reading it is free. Rewriting its anchors fixes nothing "
    "here: the scores already separate trajectories, and the arithmetic "
    "ignores them"
)

# The licence that already exists (`constraints.is_simplification`), because a
# proposer that does not know removal is free will keep an axis it cannot use.
# Worded as `proposer.render_edit_discipline` words it: losing no selection
# pair is the ACCEPTANCE condition, and not using the iteration's one deletion
# is a consequence of it, not a second condition the removal would violate.
REMOVAL_LICENCE = (
    "removal is a no-loss simplification: accepted whenever it loses no "
    "selection pair, and it does not use the iteration's one deletion"
)

# Why a dead axis is dead, when the cause is legible from the files alone.
# `absent_input` is the `dead_axis` module's case: the criterion asks the
# scorer about a field no row carries, so the scorer
# answers "absent → 0" on every row, correctly. `one_level` is everything
# else: the anchors are there, the input is there, and the scorer still
# gives one level.
DEAD_REASON_ABSENT = "absent_input"
DEAD_REASON_ONE_LEVEL = "one_level"

# How the reference fields are named in prose. A criterion can ask for the
# gold patch in words and spell `metadata.` only in the rubric header, so a
# scan that knows only the `metadata.<field>` form would stay silent on
# exactly the axis that is dead.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "gold_patch": ("gold_patch", "gold patch", "gold-patch", "reference patch"),
    "problem_statement": ("problem_statement", "problem statement"),
}


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with tied values averaged (scipy-free)."""

    sorter = np.argsort(x, kind="mergesort")
    xs = x[sorter]
    ranks = np.arange(len(x), dtype=float)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2
        i = j + 1
    out = np.empty(len(x))
    out[sorter] = ranks
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def criterion_stats(decisions: Iterable) -> dict | None:
    """Redundancy + deadness diagnostics for one candidate's rubric scores.

    ``decisions`` are FilterDecision-likes exposing ``rubric_scores``.
    Returns a JSON-safe dict, or None when fewer than two criteria carry
    scores on at least 10 shared rows.
    """

    rows = [d.rubric_scores for d in decisions if getattr(d, "rubric_scores", None)]
    criteria = sorted({c for r in rows for c in r})
    rows = [r for r in rows if all(c in r for c in criteria)]
    if len(criteria) < 2 or len(rows) < 10:
        return None

    mat = np.array([[float(r[c]) for r in rows] for c in criteria])

    per_criterion: dict[str, dict] = {}
    for c, vec in zip(criteria, mat):
        dist = Counter(round(float(v), 4) for v in vec)
        modal_value, modal_n = dist.most_common(1)[0]
        per_criterion[c] = {
            "modal_value": modal_value,
            "modal_share": modal_n / len(vec),
            "dead": modal_n / len(vec) >= DEAD_MODAL_SHARE,
            "distribution": {str(k): v for k, v in sorted(dist.items())},
        }

    live = [i for i, vec in enumerate(mat) if vec.std() > 1e-9]
    out: dict = {
        "n_rows": len(rows),
        "n_criteria": len(criteria),
        "n_live": len(live),
        "per_criterion": per_criterion,
        "effective_rank": None,
        "max_abs_corr": None,
        "max_pair": None,
        "mean_abs_corr": None,
    }
    if len(live) < 2:
        return out

    k = len(live)
    corr = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            corr[a, b] = corr[b, a] = _spearman(mat[live[a]], mat[live[b]])
    eig = np.clip(np.linalg.eigvalsh(corr), 0, None)
    iu = np.triu_indices(k, 1)
    off = np.abs(corr[iu])
    hi = int(np.argmax(off))
    ai, bi = int(iu[0][hi]), int(iu[1][hi])
    out.update(
        effective_rank=float((eig.sum() ** 2) / (eig**2).sum()),
        max_abs_corr=float(off.max()),
        max_pair=[criteria[live[ai]], criteria[live[bi]]],
        mean_abs_corr=float(off.mean()),
        # The whole matrix, not only its largest entry. A rubric with two
        # redundant pairs would otherwise be reported as having one, and a proposer
        # deciding which of six axes to merge needs every |rho|, not the max.
        live_criteria=[criteria[i] for i in live],
        abs_corr=[[round(abs(float(v)), 4) for v in row] for row in corr],
    )
    return out


def redundant_pairs(stats: Mapping | None, *, rho: float = REDUNDANT_RHO) -> set[frozenset[str]]:
    """Every unordered pair of live axes whose |rho| is at or above `rho`.

    Read off the matrix `criterion_stats` records, so a caller that needs
    the set — the new-defect check, the prompt's `!` marks — does not
    recompute a correlation from scores it may no longer have.
    """

    if not stats or not stats.get("abs_corr") or not stats.get("live_criteria"):
        return set()
    names = list(stats["live_criteria"])
    matrix = stats["abs_corr"]
    out: set[frozenset[str]] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if float(matrix[i][j]) >= rho:
                out.add(frozenset((names[i], names[j])))
    return out


def dead_axes(stats: Mapping | None) -> list[str]:
    """Criteria `criterion_stats` flagged dead, in name order."""

    if not stats:
        return []
    return sorted(name for name, pc in (stats.get("per_criterion") or {}).items() if pc.get("dead"))


def redundancy_penalty(stats: dict | None, *, weight: float, threshold: float) -> float:
    """Graded score penalty for inter-criterion redundancy.

    0 below ``threshold``; ramps linearly to ``weight`` at |rho| = 1. This
    deliberately scores a shape property, so it is OFF (weight 0) unless the
    run opts in, and even then it is only reported (`charge_redundancy`).
    """

    if weight <= 0 or not stats or stats.get("max_abs_corr") is None:
        return 0.0
    excess = max(0.0, stats["max_abs_corr"] - threshold)
    if excess <= 0:
        return 0.0
    return weight * excess / max(1e-9, 1.0 - threshold)


def charge_redundancy(score: float, penalty: float) -> tuple[float, float]:
    """Split one candidate's score into the selecting half and the reported one.

    Returns ``(selection_score, reported_score)``. The penalty is only in the
    second, so it is charged to the filter the run reports as deployable and
    not to the ordering candidates are picked in: `|rho|` does not exist
    until every row is scored, so a proposer has nothing to optimize against
    it until an iteration after it ships a criterion.

    One function rather than the same two-line conditional in the optimizer
    and in the precheck: the precheck's whole purpose is to show a proposer
    the number selection will use, and a precheck that charges the penalty
    where selection does not is worse than no precheck.
    """

    return score, score - penalty


def _saturation_line(pc: dict, *, max_levels: int = 6) -> str:
    """Share of rows at EACH score level, busiest first.

    The modal share alone says a criterion is saturated; it does not say
    whether the other rows are spread across the remaining anchors or piled
    onto one neighbour. That distinction is the difference between "widen the
    anchors" and "this axis has two states, not five", and only the full
    distribution shows it.
    """

    dist = pc.get("distribution") or {}
    total = sum(dist.values())
    if not total:
        return ""
    ranked = sorted(dist.items(), key=lambda kv: (-kv[1], float(kv[0])))
    shown = ", ".join(f"{float(v):+g}:{n / total:.0%}" for v, n in ranked[:max_levels])
    rest = len(ranked) - max_levels
    return shown + (f", +{rest} more" if rest > 0 else "")


def short_names(names: Sequence[str]) -> dict[str, str]:
    """Two-to-four-letter row and column headers for the |rho| matrix.

    Initials of the words in the name (`Task Grounding` → `TG`, `Query/PR
    Quality` → `QPQ`); a collision gets a digit. The legend line the matrix
    prints maps them back, so the abbreviation never has to be guessable on
    its own — it only has to fit a six-by-six table into a prompt.
    """

    out: dict[str, str] = {}
    taken: set[str] = set()
    for name in names:
        words = re.findall(r"[A-Za-z0-9]+", name)
        base = "".join(w[0].upper() for w in words)[:4] or name[:2].upper()
        short = base
        counter = 2
        while short in taken:
            short = f"{base}{counter}"
            counter += 1
        taken.add(short)
        out[name] = short
    return out


def render_corr_matrix(stats: Mapping | None, *, threshold: float = REDUNDANT_RHO) -> list[str]:
    """The pairwise |rho| table over the live axes, `!` on cells at or above
    `threshold`, plus a legend line. Empty when fewer than two axes are live.
    """

    names = list((stats or {}).get("live_criteria") or [])
    matrix = (stats or {}).get("abs_corr") or []
    if len(names) < 2 or not matrix:
        return []
    short = short_names(names)
    width = max(len(v) for v in short.values())
    cell = max(width, 5)
    header = " " * (width + 2) + " ".join(short[n].rjust(cell) for n in names)
    lines = [
        f"  * |rho| between every pair of live axes (`!` marks >= {threshold:.2f}, "
        "the pairs that largely re-measure one thing):",
        "    " + header,
    ]
    for i, row_name in enumerate(names):
        cells = []
        for j in range(len(names)):
            if i == j:
                cells.append("-".rjust(cell - 1) + " ")
                continue
            value = float(matrix[i][j])
            mark = "!" if value >= threshold else " "
            cells.append(f"{value:.2f}{mark}".rjust(cell))
        lines.append("    " + short[row_name].ljust(width + 2) + " ".join(cells))
    lines.append("    legend: " + ", ".join(f"{short[n]}={n}" for n in names))
    return lines


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _section_inputs(section: str, declared: Iterable[str]) -> set[str]:
    """Metadata fields one criterion's section asks the scorer to read.

    Declared (`<!-- reads: -->`), referenced as `metadata.<field>`, or named
    in prose through `_FIELD_ALIASES`. The header is deliberately not in
    scope: `initial_ranking_nolabel` names fields there precisely to say
    they are absent.
    """

    from meta_filter.view_policy import METADATA_REF_RE

    fields = set(declared) | set(METADATA_REF_RE.findall(section))
    lowered = section.lower()
    for field, aliases in _FIELD_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            fields.add(field)
    return fields


def dead_axis_reasons(
    rubric_md: str,
    criteria: Sequence[str],
    dead: Iterable[str],
    rows: Iterable,
) -> dict[str, dict]:
    """Why each dead criterion is dead, as far as the files can say.

    For every name in `dead`: the fields its own rubric section asks for
    (`_section_inputs`) are looked up on every row's metadata. If the
    section asks for at least one field and every such field is missing or
    empty on EVERY row, the reason is `DEAD_REASON_ABSENT` with the field
    names — the scorer is answering, correctly, that its input is not there,
    and the fix is the data or the declaration, not the anchors. Otherwise it
    is `DEAD_REASON_ONE_LEVEL`: the input exists and the anchors still do not
    separate these trajectories.

    A filter can carry two axes at 100 % `+0` for iterations with "DEAD AXIS"
    beside each and nothing saying why, and a proposer then neither fixes nor
    removes them. A reason is what makes the line actionable.
    """

    from meta_filter.llm_scorer import _split_rubric_sections
    from meta_filter.view_policy import declared_fields

    dead = list(dead)
    if not dead:
        return {}
    rows = list(rows)
    criteria = list(criteria)
    sections = _split_rubric_sections(rubric_md, criteria) if criteria else {}
    out: dict[str, dict] = {}
    for name in dead:
        section = sections.get(name) or ""
        asked = _section_inputs(section, declared_fields(section))
        absent = sorted(
            f
            for f in asked
            if rows and not any(_present(getattr(t, "metadata", {}).get(f)) for t in rows)
        )
        if asked and rows and len(absent) == len(asked):
            out[name] = {"reason": DEAD_REASON_ABSENT, "fields": absent}
        else:
            out[name] = {"reason": DEAD_REASON_ONE_LEVEL, "fields": []}
    return out


def _dead_tag(reason: Mapping | None) -> str:
    """Why this axis is dead.

    The repair belongs to `render_inert_axes`, which states it once for every
    inert axis instead of repeating it per criterion, and can say the
    different thing a declared-but-unread axis needs.
    """

    if reason and reason.get("reason") == DEAD_REASON_ABSENT:
        fields = ", ".join(f"metadata.{f}" for f in reason.get("fields") or [])
        return (
            f" — DEAD AXIS: its input ({fields}) is absent on every row, so the "
            "scorer answers 0 correctly"
        )
    return (
        " — DEAD AXIS: scorer gives one level on every row (anchors do not "
        "separate these trajectories)"
    )


# The words the unread tag uses, so a caller holding a rendered block can see
# whether it carries one without re-deriving the finding.
UNREAD_TAG_MARK = "DECLARED BUT UNREAD"

# The same shape for the other way an axis can be inert: it is scored on every
# row and the policy's arithmetic never looks at the number.
_UNREAD_TAG = (
    f" — {UNREAD_TAG_MARK}: policy.py lists it in RUBRIC_CRITERIA and "
    "decide() never reads its score"
)


def render_inert_axes(
    dead: Sequence[str],
    declared_unread: Sequence[str],
    *,
    removals: int | None = None,
    floor: int | None = None,
) -> str:
    """The lines that name every inert axis and say what may be done.

    Empty when nothing is inert. One line per KIND, not one per axis: the
    repair is identical within a kind and repeating it four times in a prompt
    is four times the tokens for one sentence — but it differs BETWEEN the
    kinds, and saying "rewrite it so it separates trajectories" to an axis
    whose scores already separate the pool is telling the proposer to fix the
    half that is not broken. An axis that is both dead and unread is listed
    under both, because it needs both repairs or removal.

    `removals` is how many criteria the run's deletion floor still allows
    (`ConstraintLimits.floor_for`), with `floor` the count it may not go
    below. It is what keeps "remove it" honest: a run that starts at six
    criteria may never go below five, so "for each of them: remove it" with
    two inert axes asks for a candidate the precheck then rejects as
    `rubric_criteria 4 < initial 6 - 1 (deletion floor)`. At `removals=0` the
    removal option is not offered at all. None means the caller does not know
    the floor, and then the option is stated without a count.
    """

    dead = list(dict.fromkeys(dead))
    unread = list(dict.fromkeys(declared_unread))
    if not dead and not unread:
        return ""
    removable = removals is None or removals > 0
    remove = ". Or remove it from rubric.md and RUBRIC_CRITERIA" if removable else ""
    lines = [f"  * inert axes — each {INERT_AXIS_COST}:"]
    if dead:
        lines.append(f"      - dead: {', '.join(dead)} — {DEAD_AXIS_ACTION}{remove}.")
    if unread:
        lines.append(
            f"      - declared but unread: {', '.join(unread)} — {UNREAD_AXIS_ACTION}{remove}."
        )
    if not removable:
        lines.append(
            f"      - none of them may be removed: the run's deletion floor "
            f"({floor} criteria) is already reached, so they are fixed in place or paid for."
        )
    elif removals is None:
        lines.append(f"      - {REMOVAL_LICENCE}.")
    else:
        lines.append(
            f"      - the deletion floor leaves room to remove {removals} of them "
            f"(never below {floor} criteria), and {REMOVAL_LICENCE}."
        )
    return "\n".join(lines)


def render_criterion_health(
    stats: dict | None,
    *,
    threshold: float,
    dead_reasons: Mapping[str, Mapping] | None = None,
    declared_unread: Sequence[str] = (),
    removals_left: int | None = None,
    criteria_floor: int | None = None,
    budget: int = HEALTH_BLOCK_BUDGET,
) -> str:
    """One prompt-ready block: deadness and redundancy, framed as findings.

    Deterministic and free — every number here is arithmetic over rubric
    scores the solver already paid for. Unlike the LLM rubric critic it reads
    behaviour rather than prose.

    Four parts, in this order: one line per criterion (modal share, a DEAD
    AXIS tag with its reason from `dead_reasons` where one is known, a
    DECLARED BUT UNREAD tag for the names in `declared_unread`, the per-level
    distribution beneath it), the inert-axis lines that group every axis
    carrying either tag by which tag it carries and say what may be done
    about each kind (`render_inert_axes`), the effective rank, and the full
    pairwise |rho|
    matrix with `!` on every pair at or above `threshold`
    (`render_corr_matrix`), followed by the most-correlated pair and what it
    costs. Over `budget` characters the per-level
    lines are dropped, live axes first, then the matrix legend — the matrix,
    the dead-axis reasons and the inert-axis lines are never trimmed, because
    those are the things this block exists to say.

    `declared_unread` comes from `constraints.declared_unread_criteria` over
    the same filter these scores came from. A dead axis and an unread one are
    the same bill — one scorer call per row for no ordering — and differ only
    in who can see it: deadness is in the scores, unreadness is in policy.py.
    They are named together for that reason, and repaired differently.

    `removals_left` is how
    many criteria this run's deletion floor still allows the filter to lose
    and `criteria_floor` the count it may not go below
    (`ConstraintLimits.floor_for`); both are measured on rubric.md, like the
    floor itself, and are passed rather than derived here because
    `stats["n_criteria"]` counts what was SCORED, which is the policy's
    declaration and not the file the floor is checked against.

    The correlated-pair line says the pair costs only the final filter's
    reported score, not this iteration's selection: stating the wrong
    consequence is how a diagnostic gets optimized against as if it were a
    term.
    """

    if not stats:
        return ""
    dead_reasons = dead_reasons or {}
    unread = list(dict.fromkeys(declared_unread))
    lines: list[str] = [
        f"criterion health over {stats['n_rows']} rows, {stats['n_criteria']} criteria:"
    ]
    # (index of the levels line, modal share) for the trim below.
    level_lines: list[tuple[int, float, bool]] = []
    for c, pc in sorted(stats["per_criterion"].items(), key=lambda kv: -kv[1]["modal_share"]):
        tag = _dead_tag(dead_reasons.get(c)) if pc["dead"] else ""
        if c in unread:
            tag += _UNREAD_TAG
        lines.append(
            f"  * {c}: {pc['modal_share']:.0%} of scores sit at {pc['modal_value']:+g}{tag}"
        )
        levels = _saturation_line(pc)
        if levels:
            lines.append(f"      score levels: {levels}")
            level_lines.append((len(lines) - 1, pc["modal_share"], bool(pc["dead"]) or c in unread))
    # Dead by measurement or unread by declaration — named together, once,
    # each with the repair its own defect takes.
    inert_line = render_inert_axes(
        dead_axes(stats), unread, removals=removals_left, floor=criteria_floor
    )
    if inert_line:
        lines.append(inert_line)
    if stats.get("effective_rank") is not None:
        lines.append(
            f"  * effective rank {stats['effective_rank']:.1f} of {stats['n_live']} live axes"
        )
    matrix_lines = render_corr_matrix(stats, threshold=threshold)
    legend_index = None
    if matrix_lines:
        lines.extend(matrix_lines)
        legend_index = len(lines) - 1
    if stats.get("max_abs_corr") is not None and stats["max_pair"]:
        a, b = stats["max_pair"]
        note = ""
        if stats["max_abs_corr"] > threshold:
            note = (
                " — these two largely re-measure one axis; merging or "
                "differentiating them frees budget without losing ordering"
            )
            note += (
                ", and it is charged to the final filter's reported "
                "score rather than to this iteration's selection"
            )
        lines.append(
            f"  * most correlated pair: {a} vs {b} (|rho|={stats['max_abs_corr']:.2f}){note}"
        )

    # Trim to budget: levels lines of live axes first (the most balanced
    # distribution is the least informative one to keep), then the levels of
    # dead axes, then the legend.
    drop: set[int] = set()
    ordered = sorted(level_lines, key=lambda item: (item[2], item[1]))
    while sum(len(line) + 1 for i, line in enumerate(lines) if i not in drop) > budget:
        if ordered:
            drop.add(ordered.pop(0)[0])
        elif legend_index is not None and legend_index not in drop:
            drop.add(legend_index)
        else:
            break
    return "\n".join(line for i, line in enumerate(lines) if i not in drop)


def criterion_stability(
    winning_cells: dict[str, dict[str, float]],
    fresh_cells: dict[str, dict[str, float]],
) -> dict:
    """Per criterion, the share of rows whose score moved between two scorings.

    The promotion re-score scores the accepted candidate
    twice on the same selection rows — the draw that won and a fresh one —
    and the two draws are otherwise thrown away once the gate has confirmed
    or rejected. Kept side by side they measure something no other channel
    does: how consistently the scorer reads each criterion's anchors. About
    a fifth of cells move on an identical re-score pool-wide; a criterion
    whose rows move at twice that rate is being read inconsistently, and
    that is a rubric defect the proposer can fix by tightening the anchors,
    where nothing in the concordance numbers would ever point at it.

    Only rows scored in both draws count. Returns ``{"n_rows", "per_criterion":
    {name: {"n", "moved", "share_moved", "mean_abs_delta"}}}``; empty
    `per_criterion` when the draws share no row.
    """

    shared = sorted(set(winning_cells) & set(fresh_cells))
    per_criterion: dict[str, dict] = {}
    for tid in shared:
        before, after = winning_cells[tid], fresh_cells[tid]
        for criterion in set(before) & set(after):
            entry = per_criterion.setdefault(criterion, {"n": 0, "moved": 0, "abs_delta_sum": 0.0})
            delta = float(after[criterion]) - float(before[criterion])
            entry["n"] += 1
            entry["abs_delta_sum"] += abs(delta)
            if delta != 0.0:
                entry["moved"] += 1
    out = {}
    for criterion, entry in sorted(per_criterion.items()):
        n = entry["n"]
        out[criterion] = {
            "n": n,
            "moved": entry["moved"],
            "share_moved": round(entry["moved"] / n, 4) if n else 0.0,
            "mean_abs_delta": round(entry["abs_delta_sum"] / n, 4) if n else 0.0,
        }
    return {"n_rows": len(shared), "per_criterion": out}


# A criterion whose rows move at or above this share between two scorings
# is called out by name; the pool-wide rate on an identical re-score is
# about 0.20 (`.claude` memory, scorer cells are samples), so this is
# roughly twice the background.
UNSTABLE_SHARE = 0.35


def render_criterion_stability(stability: dict | None, *, candidate: str = "") -> str:
    """One prompt-ready block from `criterion_stability`, busiest axis first.

    Empty when nothing was measured — the block sits beside the criterion
    health block in the proposer prompt and an empty string renders nothing
    there, so a run whose gate never promoted reads exactly as before.
    """

    if not stability or not stability.get("per_criterion"):
        return ""
    who = f" of {candidate}" if candidate else ""
    lines = [
        f"criterion stability{who}: the promoted filter was scored twice on "
        f"{stability['n_rows']} hidden rows (the winning draw and a fresh one); "
        "share of rows whose score moved, per criterion:"
    ]
    ranked = sorted(stability["per_criterion"].items(), key=lambda kv: -kv[1]["share_moved"])
    for criterion, entry in ranked:
        tag = ""
        if entry["share_moved"] >= UNSTABLE_SHARE:
            tag = (
                " — anchors are read inconsistently; tighten what separates "
                "adjacent levels or the axis is scorer noise"
            )
        lines.append(
            f"  * {criterion}: {entry['share_moved']:.0%} of rows moved "
            f"(mean |delta| {entry['mean_abs_delta']:.2f}){tag}"
        )
    return "\n".join(lines)
