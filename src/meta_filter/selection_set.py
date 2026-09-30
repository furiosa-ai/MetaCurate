"""Hidden fixed selection set — the pairs the acceptance gate is graded on.

A loop graded against its own train pairs shows an internal objective that
keeps rising while an external yardstick peaks early and then moves inside
its own noise. Two mechanisms behind that are about *what* gets measured and
*who* sees it, not about the objective formula:

* grading candidates on the same truth pairs the proposer is shown next
  iteration makes the exam memorisable, and it grows every iteration;
* four noisy measurements with the max taken is a winner's curse, and a
  candidate re-scored on the pairs that made it win keeps the lucky draw.

This module is the exam half of the answer: a set of rows disjoint from
train and probe, a FIXED set of pairs
over them built once from the seed, judged once in both orders with ties
dropped, and persisted with provenance so the baseline's number is a constant
rather than a moving target. Nothing here is ever rendered into a proposer
prompt; `selection_row_ids_in_text` exists so the optimizer wiring can prove
that.

The optimizer wiring (per-iteration scoring of every candidate, the paired
gate, the promotion re-score) lives elsewhere; this module has no LLM cost of
its own beyond the one-off judge call.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from meta_filter.filter_loader import LoadedFilter
from meta_filter.pair_judge import PairResolution
from meta_filter.ranking import ConcordanceResult, _sign, concordance_detail
from meta_filter.schemas import FilterDecision, Trajectory

logger = logging.getLogger(__name__)

SELECTION_SET_FILE = "selection_set.json"


def selection_set_path(run_dir: str | Path) -> Path:
    """Where a run's selection set lives: ``runs/_selection/<run_id>.json``.

    Outside the run directory on purpose. The agentic proposer's workspace is
    the run directory, so anything in it is one ``cat`` away; the set holds the
    judge's order for every selection pair, which is exactly the answer key
    the proposer must never see. Same arrangement as ``runs/_idmap/``.
    """

    run_dir = Path(run_dir)
    return run_dir.parent / "_selection" / f"{run_dir.name}.json"


def baseline_scores_path(run_dir: str | Path) -> Path:
    """Where the baseline's selection scores live.

    ``runs/_selection/<run_id>.baseline_scores.json``, beside the selection
    set and for the same reason: the map is the current filter's score on
    every hidden row, which the proposer must not read. It is rewritten at
    every promotion (with the fresh, canonical draw) and once at set build,
    so a resumed run reads back the exact numbers the gate was last measured
    against instead of re-scoring.
    """

    run_dir = Path(run_dir)
    return run_dir.parent / "_selection" / f"{run_dir.name}.baseline_scores.json"


# `n_pairs` value meaning "every unordered pair, no sampling, no stratification".
ALL_PAIRS: Literal["all"] = "all"
PairCount = int | Literal["all"]

# Share of `n_pairs` drawn from the small-gap quartile when stratifying.
HARD_SHARE = 0.5
# The hard stratum is the smallest-gap quarter of the pair universe.
HARD_QUANTILE = 0.25


class _PairJudgeLike(Protocol):
    """The one method of `PairJudge` this module calls.

    Named so tests can hand in a fake without standing up an LLM client, and
    so the optimizer's real `PairJudge` type-checks against it unchanged.
    """

    def resolve_pairs_detail(
        self,
        pairs: list[tuple[str, str]],
        traj_by_id: dict[str, Trajectory],
        *,
        max_workers: int = 8,
    ) -> PairResolution: ...


class _SolverLike(Protocol):
    def run(
        self, loaded_filter: LoadedFilter, trajectories: list[Trajectory]
    ) -> list[FilterDecision]: ...


@dataclass(frozen=True)
class SelectionPlan:
    """The unordered pairs chosen for judging, and how they were chosen.

    Kept separate from the judged `SelectionSet` because the plan is pure
    arithmetic on ids and a seed — cheap, deterministic, testable without a
    judge — while judging costs money and happens once per run.
    """

    pairs: list[tuple[str, str]]
    n_requested: int
    seed: int
    stratified: bool
    n_hard: int
    n_uniform: int
    # "all" when the plan is the whole pair universe, "sample" otherwise.
    pairs_mode: str = "sample"

    def provenance(self) -> dict[str, Any]:
        return {
            "n_requested": self.n_requested,
            "seed": self.seed,
            "stratified": self.stratified,
            "n_hard": self.n_hard,
            "n_uniform": self.n_uniform,
            "pairs_mode": self.pairs_mode,
        }


def _pair_universe(ids: Iterable[str]) -> list[tuple[str, str]]:
    """Every unordered pair over the distinct ids, as sorted tuples, in sorted order."""

    ordered = sorted(set(ids))
    return [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1 :]]


def plan_selection_pairs(
    ids: list[str],
    *,
    n_pairs: PairCount,
    seed: int,
    initial_scores: Mapping[str, float] | None = None,
) -> SelectionPlan:
    """Choose the unordered pairs the selection set will be judged on.

    Deterministic in (sorted ids, seed, initial_scores): every run that shares
    the selection rows regenerates the identical list, so the judge cache is
    shared and two runs' selection concordances are measured on the same
    exam. That determinism has a hole, though: `initial_scores` is a scorer
    draw, and two runs launched in parallel score the initial filter before
    the cache holds it, get different cells, and so plan different hard
    halves that share only a fraction of their pairs. ``n_pairs="all"``
    closes it by taking the whole universe — no seed, no scores, no
    stratification — which is what `scripts.build_selection_set` uses to
    build one set every run reads.

    With `initial_scores` covering every id, half the pairs come from the
    quartile of pairs with the smallest |score gap| under the initial filter.
    Those are the pairs the starting filter is least sure about, so they are
    where a candidate's edit can actually change a verdict; a uniform draw
    over C(60, 2) pairs is dominated by pairs any filter orders the same way
    and carries little signal per judged dollar. The other half stays
    uniform over the remaining pairs so the set still reads as a sample of
    the whole ordering, not only of the boundary. Without full score
    coverage the draw is uniform, like `probe.fixed_validation_pairs`.

    A universe no larger than `n_pairs` is returned whole; a pair is never
    drawn twice.
    """

    universe = _pair_universe(ids)
    if n_pairs == ALL_PAIRS:
        return SelectionPlan(
            pairs=universe,
            n_requested=len(universe),
            seed=seed,
            stratified=False,
            n_hard=0,
            n_uniform=len(universe),
            pairs_mode="all",
        )
    if not isinstance(n_pairs, int) or n_pairs < 0:
        raise ValueError(f"n_pairs must be 'all' or a non-negative int, got {n_pairs!r}")

    stratified = bool(initial_scores) and all(tid in initial_scores for tid in set(ids))

    if len(universe) <= n_pairs:
        return SelectionPlan(
            pairs=universe,
            n_requested=n_pairs,
            seed=seed,
            stratified=False,
            n_hard=0,
            n_uniform=len(universe),
        )

    rng = random.Random(seed)
    if not stratified:
        sampled = sorted(rng.sample(universe, n_pairs))
        return SelectionPlan(
            pairs=sampled,
            n_requested=n_pairs,
            seed=seed,
            stratified=False,
            n_hard=0,
            n_uniform=len(sampled),
        )

    assert initial_scores is not None
    # Rank-based quartile, with the pair itself as a tiebreaker so equal
    # gaps (a common case: integer-valued policies) still split the same way
    # on every machine.
    by_gap = sorted(
        universe,
        key=lambda p: (abs(float(initial_scores[p[0]]) - float(initial_scores[p[1]])), p),
    )
    cut = max(1, int(len(by_gap) * HARD_QUANTILE))
    hard_pool = by_gap[:cut]
    rest_pool = by_gap[cut:]

    want_hard = min(int(round(n_pairs * HARD_SHARE)), len(hard_pool))
    want_rest = min(n_pairs - want_hard, len(rest_pool))
    # If the rest is too small to fill its half, the hard pool tops it up;
    # the universe is larger than n_pairs so between them there is enough.
    want_hard = min(n_pairs - want_rest, len(hard_pool))

    hard = rng.sample(hard_pool, want_hard)
    uniform = rng.sample(rest_pool, want_rest)
    pairs = sorted(hard + uniform)
    return SelectionPlan(
        pairs=pairs,
        n_requested=n_pairs,
        seed=seed,
        stratified=True,
        n_hard=len(hard),
        n_uniform=len(uniform),
    )


def build_selection_pairs(
    ids: list[str],
    *,
    n_pairs: PairCount,
    seed: int,
    initial_scores: Mapping[str, float] | None = None,
) -> list[tuple[str, str]]:
    """The unordered pairs of `plan_selection_pairs`, without the bookkeeping."""

    return plan_selection_pairs(
        ids, n_pairs=n_pairs, seed=seed, initial_scores=initial_scores
    ).pairs


@dataclass
class SelectionSet:
    """Judged selection pairs plus the rows they live on and how they were made.

    `pairs` are ordered ``(hi, lo)`` in the judge's order — both presentation
    orders agreed. `tied` are pairs the two orders disagreed on and
    `unresolved` are pairs no verdict came back for; both are kept so the
    record shows how much of the paid-for judging produced supervision, but
    neither is ever scored (half credit on a pair with no verdict is a score
    a candidate can farm by tying it too).

    `row_ids` is every selection row, not only the rows that appear in a
    pair: the exposure guard hides the whole file from the proposer.
    """

    row_ids: list[str]
    pairs: list[tuple[str, str]]
    tied: list[tuple[str, str]] = field(default_factory=list)
    unresolved: list[tuple[str, str]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pairs)

    def pair_signature(self) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
        """What makes two sets the same exam: the judged orders and the ties.

        Provenance (timestamps, file paths) is deliberately left out — a set
        copied into a run directory is the same exam as its source whatever
        the copy's metadata says, and a set with one pair judged the other
        way is not, however alike the metadata looks.
        """

        return (
            tuple(sorted(self.pairs)),
            tuple(sorted(tuple(sorted(p)) for p in self.tied)),
        )

    def pair_ids(self) -> set[str]:
        """Every id that appears in a judged, tied or unresolved pair."""

        groups = (self.pairs, self.tied, self.unresolved)
        return {tid for group in groups for p in group for tid in p}

    def concordance_detail(self, scores: Mapping[str, float]) -> ConcordanceResult:
        """Concordance of `scores` on the judged pairs, charging unscored pairs.

        ``unscored="loss"`` because this number decides acceptance: a
        candidate that declines to score a selection row is a worse ranker,
        and forgiving it would let a candidate improve its exam by dropping
        the rows it gets wrong.
        """

        return concordance_detail(dict(scores), self.pairs, unscored="loss")

    def concordance(self, scores: Mapping[str, float]) -> float:
        return self.concordance_detail(scores).concordance

    def correctness(self, scores: Mapping[str, float]) -> list[bool]:
        """Per-pair right/wrong, aligned with `self.pairs`.

        The paired acceptance test needs a binary vector per candidate. A
        candidate's own tie earns 0.5 in `concordance` but counts as wrong
        here — a pair the candidate did not order is not a pair it got right
        — and so does a pair it left unscored.
        """

        out: list[bool] = []
        for hi, lo in self.pairs:
            if hi not in scores or lo not in scores:
                out.append(False)
                continue
            out.append(_sign(float(scores[hi]) - float(scores[lo])) > 0)
        return out

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pairs"] = [list(p) for p in self.pairs]
        data["tied"] = [list(p) for p in self.tied]
        data["unresolved"] = [list(p) for p in self.unresolved]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelectionSet:
        def _pairs(key: str) -> list[tuple[str, str]]:
            return [(str(a), str(b)) for a, b in data.get(key, [])]

        return cls(
            row_ids=[str(r) for r in data.get("row_ids", [])],
            pairs=_pairs("pairs"),
            tied=_pairs("tied"),
            unresolved=_pairs("unresolved"),
            provenance=dict(data.get("provenance", {})),
        )

    def save(self, path: str | Path) -> Path:
        """Write ``selection_set.json``; a directory path gets the default name.

        Plain JSON with ids as-is. The file lives under ``runs/_selection/``
        (see `selection_set_path`), outside the proposer's workspace, so there
        is nothing to pseudonymize; keeping raw ids is what lets the
        exposure-guard test grep for them.
        """

        target = Path(path)
        if target.is_dir() or target.suffix == "":
            target = target / SELECTION_SET_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> SelectionSet:
        target = Path(path)
        if target.is_dir():
            target = target / SELECTION_SET_FILE
        with target.open("r", encoding="utf-8") as fp:
            return cls.from_dict(json.load(fp))


def _judge_label(judge: object) -> str:
    """Something that identifies which judge produced the verdicts.

    The model name when the client exposes one, otherwise the cache directory
    — a run reading the file later needs to know whether another run's
    selection concordance is comparable to its own.
    """

    client = getattr(judge, "client", None)
    model = getattr(client, "model", None)
    if model:
        return str(model)
    cache_dir = getattr(judge, "cache_dir", None)
    if cache_dir is not None:
        return str(cache_dir)
    return type(judge).__name__


def judge_selection_pairs(
    pairs: list[tuple[str, str]],
    traj_by_id: dict[str, Trajectory],
    judge: _PairJudgeLike,
    *,
    max_workers: int = 8,
    provenance: Mapping[str, Any] | None = None,
) -> SelectionSet:
    """Adjudicate the planned pairs once, in both orders, and freeze the result.

    Same call as the train-side judge in the optimizer, so the verdicts share
    its on-disk cache: a second seed or a re-run on the same rows re-reads
    them instead of paying again. Ties are dropped, not scored at 0.5 — the
    judge gave no verdict, so the pair is not part of the exam.

    `provenance` is the plan's bookkeeping (`SelectionPlan.provenance`); the
    judge label, tie share and timestamp are added here.
    """

    resolution = judge.resolve_pairs_detail(list(pairs), traj_by_id, max_workers=max_workers)
    resolved = [(str(hi), str(lo)) for hi, lo in resolution.resolved]
    tied = [(str(a), str(b)) for a, b in resolution.tied]
    unresolved = [(str(a), str(b)) for a, b in resolution.unresolved]
    logger.info(
        "selection set: %d resolved, %d TIE (%.1f%% of judged), %d unanswered of %d planned pairs",
        len(resolved),
        len(tied),
        100.0 * resolution.tie_share,
        len(unresolved),
        len(pairs),
    )

    record: dict[str, Any] = dict(provenance or {})
    record.setdefault("n_requested", len(pairs))
    record.update(
        {
            "n_planned": len(pairs),
            "n_resolved": len(resolved),
            "n_tied": len(tied),
            "n_unresolved": len(unresolved),
            "tie_share": round(resolution.tie_share, 4),
            "judge_model": _judge_label(judge),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    return SelectionSet(
        row_ids=sorted(traj_by_id),
        pairs=resolved,
        tied=tied,
        unresolved=unresolved,
        provenance=record,
    )


def build_selection_set(
    rows: list[Trajectory],
    judge: _PairJudgeLike,
    *,
    n_pairs: PairCount,
    seed: int,
    initial_scores: Mapping[str, float] | None = None,
    max_workers: int = 8,
) -> SelectionSet:
    """Plan and judge in one step."""

    traj_by_id = {t.trajectory_id: t for t in rows}
    plan = plan_selection_pairs(
        list(traj_by_id), n_pairs=n_pairs, seed=seed, initial_scores=initial_scores
    )
    logger.info(
        "selection set: %d rows, %d pairs planned (%s: %d hard, %d uniform)",
        len(traj_by_id),
        len(plan.pairs),
        "stratified" if plan.stratified else "uniform",
        plan.n_hard,
        plan.n_uniform,
    )
    return judge_selection_pairs(
        plan.pairs,
        traj_by_id,
        judge,
        max_workers=max_workers,
        provenance=plan.provenance(),
    )


def score_on_selection(
    loaded_filter: LoadedFilter, solver: _SolverLike, rows: list[Trajectory]
) -> dict[str, float]:
    """The filter's continuous scores on the selection rows.

    Same path as `ProbeSet.evaluate` and the train solver pass, so a
    candidate's selection number is measured by the code that measures its
    train number. Rows the scorer failed on are left out rather than given a
    placeholder: a missing key is what `SelectionSet.concordance` charges
    for, whereas a fake 0.0 would be a real (and wrong) ordering.
    """

    return scores_from_decisions(solver.run(loaded_filter, rows))


def scores_from_decisions(decisions: Iterable) -> dict[str, float]:
    """The continuous score per row, dropping rows the scorer failed on.

    Split out of `score_on_selection` so a caller that also wants the
    per-criterion cells (`FilterDecision.rubric_scores`, for the criterion
    stability readout) can run the solver once and derive both.
    """

    return {
        d.trajectory_id: float(d.rank_score)
        for d in decisions
        if d.rank_score is not None and not d.scorer_failed
    }


def assert_disjoint(
    selection_ids: Iterable[str], *, train_ids: Iterable[str], probe_ids: Iterable[str]
) -> None:
    """Refuse a selection set that shares a row with train or probe.

    A shared train row is a row the proposer sees the judge's verdict on, so
    the gate would grade on exam questions the candidate was written against.
    A shared probe row leaks gate tuning into the final held-out, which then
    stops being external. Either overlap is a configuration error, not a
    condition to warn about and continue.
    """

    selection = set(selection_ids)
    train_overlap = sorted(selection & set(train_ids))
    probe_overlap = sorted(selection & set(probe_ids))
    if not train_overlap and not probe_overlap:
        return
    parts = []
    if train_overlap:
        parts.append(f"{len(train_overlap)} shared with train: {', '.join(train_overlap[:10])}")
    if probe_overlap:
        parts.append(f"{len(probe_overlap)} shared with probe: {', '.join(probe_overlap[:10])}")
    raise ValueError("selection rows must be disjoint from train and probe — " + "; ".join(parts))


def selection_row_ids_in_text(text: str, ids: Iterable[str]) -> list[str]:
    """Which selection ids occur verbatim in `text`, sorted.

    The exposure guard: run over every saved proposer prompt and evidence
    file, this must return an empty list. Selection rows are never in the
    pseudonymizer's pool, so a leak would carry the raw id — which is what
    this looks for.
    """

    if not text:
        return []
    return sorted(tid for tid in set(ids) if tid and tid in text)
