"""The judged pair set, written where the proposer can read it.

The loop is supervised entirely by pair verdicts, and without this file none
of them would be legible from inside the run directory. `ranking.json` records how
many pairs there were; `critic_outputs/<candidate>.json` holds an
evidence-rich but capped slice of the ones a candidate got backwards; the
verdicts themselves live in the pair-judge cache, which sits outside the
workspace the agentic proposer is allowed to read. So a proposer asked to fix
an ordering would see neither the exam nor its own answers to it.

`ranking_pairs.jsonl` is that exam, one JSON object per line:

    hi, lo            the judge's order, preferred trajectory first. On a TIE
                      row these are the order the pair was sampled in and
                      carry no preference — `tie` says so.
    tie               the judge answered both presentations and they named
                      different winners, so the pair carries a verdict but no
                      order, and concordance excludes it.
    source            "disputed" (the candidates ordered it differently and
                      the pair was bought), "audit" (they agreed and it was
                      bought anyway, to check the consensus), or "inherited"
                      (not bought by this run).
    judged_iteration  which iteration of THIS run paid for it; null when
                      inherited.
    in_pool           whether both trajectories are in this iteration's pool.
                      A pair that is not cannot be scored, so it carries no
                      `by_candidate` entries.
    orders            one entry per cached presentation, naming which
                      trajectory sat in slot A, who won that presentation and
                      whether it was decided at all. Two entries that name
                      different winners are what a `tie` is.
    rationale_ab      the prose from the `ab` presentation (the
                      alphabetically lower id shown as A), and
    rationale_ba      from the `ba` one. Both are the judge's own words; they
                      are what a TIE row is worth reading for.
    by_candidate      {candidate_id: {"ordered_correctly": bool|null,
                      "gap": float|null}} for every candidate scored this
                      iteration, the baseline included. Absent entirely
                      before the candidates have been scored.

`ordered_correctly` is tri-state on purpose, because concordance is:

    True   the candidate scored `hi` strictly above `lo`
    False  it inverted the pair, or it emitted no score for one of the rows —
           `ranking.concordance_detail(unscored="loss")` charges that to the
           candidate rather than to its rivals, so it is a loss here too
    null   the two scores are exactly equal (concordance gives half credit),
           or the row is a TIE and there is no judge order to be right about

so "the pairs this candidate got backwards" is `gap is not None and gap < 0`,
which is exactly `ranking.missed_pairs`.
"""

from __future__ import annotations

import json
from pathlib import Path

from meta_filter.core.schemas import Trajectory
from meta_filter.visibility.pseudonym import Pseudonymizer
from meta_filter.visibility.view_policy import pair_view_signature

FILENAME = "ranking_pairs.jsonl"


def _by_candidate(
    hi: str,
    lo: str,
    tie: bool,
    scores_by_candidate: dict[str, dict[str, float]],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for candidate_id, scores in scores_by_candidate.items():
        if hi not in scores or lo not in scores:
            # The contract violation, recorded as the loop charges it.
            out[candidate_id] = {"ordered_correctly": False, "gap": None}
            continue
        gap = scores[hi] - scores[lo]
        if tie or gap == 0:
            ordered = None
        else:
            ordered = gap > 0
        out[candidate_id] = {"ordered_correctly": ordered, "gap": gap}
    return out


def _orders_and_rationales(
    hi: str, lo: str, traj_by_id: dict[str, Trajectory], judge
) -> tuple[list[dict], str, str]:
    """The cached presentations of one pair, plus the two rationales.

    Empty for a pair whose trajectories are not both in this pool: the cache
    key includes the label view, which is a property of the two trajectories,
    so without them the record cannot even be addressed.
    """

    if judge is None:
        return [], "", ""
    traj_hi, traj_lo = traj_by_id.get(hi), traj_by_id.get(lo)
    if traj_hi is None or traj_lo is None:
        return [], "", ""
    orders = judge.presentations_for(hi, lo, pair_view_signature(traj_hi, traj_lo))
    prose = {entry.get("order"): entry.get("rationale", "") for entry in orders}
    return orders, prose.get("ab", ""), prose.get("ba", "")


def ranking_pair_rows(
    *,
    accumulated_pairs: list[tuple[str, str]],
    tied_pairs: list[tuple[str, str]],
    provenance: dict[frozenset[str], dict],
    traj_by_id: dict[str, Trajectory],
    judge=None,
    scores_by_candidate: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    """One row per judged pair: the accumulated truth set, then this iteration's TIEs.

    TIE rows come last and are kept deliberately. They cost exactly what a
    resolved pair costs — two judge calls — and they are the only record of
    where the judge contradicts itself, which is a fact about the pair rather
    than about any candidate. Dropping them from the dump would leave the
    proposer unable to tell "no pair covers this region" from "every pair
    covering it came back TIE".
    """

    rows: list[dict] = []
    for pair, tie in [(p, False) for p in accumulated_pairs] + [(p, True) for p in tied_pairs]:
        hi, lo = pair
        where = provenance.get(frozenset(pair), {})
        orders, rationale_ab, rationale_ba = _orders_and_rationales(hi, lo, traj_by_id, judge)
        row = {
            "hi": hi,
            "lo": lo,
            "tie": tie,
            "source": where.get("source", "disputed"),
            "judged_iteration": where.get("judged_iteration"),
            "in_pool": hi in traj_by_id and lo in traj_by_id,
            "orders": orders,
            "rationale_ab": rationale_ab,
            "rationale_ba": rationale_ba,
        }
        if scores_by_candidate is not None and row["in_pool"]:
            row["by_candidate"] = _by_candidate(hi, lo, tie, scores_by_candidate)
        elif scores_by_candidate is not None:
            row["by_candidate"] = {}
        rows.append(row)
    return rows


def write_ranking_pairs(
    path: str | Path, rows: list[dict], *, pseudonymizer: Pseudonymizer | None = None
) -> int:
    """Write the rows as JSONL, redacted the way the prompt is, and count them.

    Redaction runs over the serialized line rather than over the id fields:
    the judge's rationale is free prose and has been observed naming the
    trajectories it compares, so redacting only the keys would publish under
    the proposer's nose exactly what the pseudonymizer exists to withhold.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, default=str) for row in rows]
    if pseudonymizer is not None:
        lines = [pseudonymizer.redact(line) for line in lines]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return len(rows)
