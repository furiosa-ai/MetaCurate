"""Build one shared selection set, outside any run, that every run then reads.

    uv run python -m scripts.build_selection_set \\
        --rows data/nemotron_selection220.jsonl \\
        --judge-model openai/gpt-5.6-luna \\
        --judge-cache-dir data/pair_judge_cache \\
        --pairs 4000 \\
        --exclude data/nemotron_train50.jsonl data/nemotron_probe25.jsonl \\
        --out data/nemotron_selection220_judged.json

The selection set is the exam the paired acceptance gate grades candidates
on: a fixed list of pairs over rows disjoint from train and probe, each
judged in both presentation orders. Planning it inside a run makes the exam
depend on that run's first scorer draw (the plan stratifies on the initial
filter's score gaps), so two runs launched side by side could be graded on
different pairs. Building it once here, with no stratification (``--pairs
all`` is every unordered pair; ``--pairs N`` a seed-fixed uniform sample),
and handing every run the same file (``meta_filter.cli run --selection-set``)
keeps their selection concordances comparable.

The rows are stripped at load exactly as a run strips them, and the output is
stamped ``metadata_free: true``, which is what a run checks before it accepts
the set. Judging is the optimizer's own `PairJudge` in both presentation
orders over the same on-disk cache, so a re-run with the same arguments asks
the model nothing and the runs that later read the set share its verdicts.
The OpenRouter key comes from ``OPENROUTER_API_KEY`` or ``~/.env`` and is
never logged. Rows are checked disjoint from every ``--exclude`` file — a
shared train row would grade the gate on a verdict the proposer was shown.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meta_filter.llm.cost_meter import LEDGER, RoleCost, metered
from meta_filter.metadata_free import file_sha256, pseudonymize_id, pseudonymize_selection_set
from meta_filter.selection_set import (
    ALL_PAIRS,
    PairCount,
    SelectionSet,
    _PairJudgeLike,
    judge_selection_pairs,
    plan_selection_pairs,
)
from meta_filter.trajectory import load_trajectories
from meta_filter.view_policy import SCORER_BASELINE_FIELDS, absent_field_signature

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_ROLE = "judge"


def parse_pairs(text: str) -> PairCount:
    """``all`` or a positive integer; anything else is an argument error."""

    value = text.strip().lower()
    if value == ALL_PAIRS:
        return ALL_PAIRS
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--pairs must be 'all' or a positive int, got {text!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"--pairs must be positive, got {n}")
    return n


def trajectory_ids(path: Path) -> list[str]:
    """Every `trajectory_id` in a JSONL pool as a run sees it (pseudonymized)."""

    ids: list[str] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                ids.append(pseudonymize_id(str(json.loads(line)["trajectory_id"])))
    return ids


def assert_excluded(row_ids: Iterable[str], exclude: Sequence[Path]) -> None:
    """Refuse a row that also appears in any excluded pool.

    Both sides are compared in the namespace a run uses, so a row whose id
    names a task instance is matched through its pseudonym.

    Raises:
        ValueError: Names each offending file and the first shared ids.
    """

    rows = set(row_ids)
    problems: list[str] = []
    for path in exclude:
        shared = sorted(rows & set(trajectory_ids(path)))
        if shared:
            problems.append(f"{len(shared)} shared with {path}: {', '.join(shared[:5])}")
    if problems:
        raise ValueError(
            "selection rows must be disjoint from the excluded pools — " + "; ".join(problems)
        )


def assert_uniform_fields(rows: Sequence[Any]) -> None:
    """Every exam row must expose the same LLM-facing fields, or refuse to build.

    An exam whose rows differ in what the scorer can see is not one exam (the
    pair judge renders no metadata field, so the scorer's fields are the ones
    that can differ): a rule reading a field that one side of a pair carries and the
    other lacks fires on one side only, whatever the two trajectories are
    actually worth.

    Checked against the scorer's field set, not the union of keys present, so a
    bookkeeping field one half happens to carry is not mistaken for a
    difference that reaches a model.

    Raises:
        ValueError: The rows do not agree on which scorer fields they carry,
            naming a row from each group.
    """

    for role_fields, what in ((SCORER_BASELINE_FIELDS, "scorer"),):
        groups: dict[str, str] = {}
        for row in rows:
            sig = absent_field_signature(role_fields, row.metadata or {})
            groups.setdefault(sig, row.trajectory_id)
        if len(groups) > 1:
            shown = ", ".join(f"{tid} ({sig or 'all present'})" for sig, tid in groups.items())
            raise ValueError(
                f"the rows do not agree on which {what} fields they carry — {len(groups)} "
                f"groups: {shown}. An exam built from them would grade half its pairs on "
                "information the other half never showed."
            )


def carried_pairs(seed_set: Path, traj_by_id: dict[str, Any]) -> list[tuple[str, str]]:
    """Every unordered pair an earlier set already paid a verdict for.

    Growing the exam must not throw away judging that is already bought:
    carrying the earlier set's pairs into the plan makes them cache hits
    rather than re-buys, and keeps the new exam a strict superset of the old
    one so the two are comparable. Ties come along with the resolved pairs —
    a tie is a paid verdict too.

    Raises:
        ValueError: The seed set names a row the new rows file does not have.
            Silently dropping those pairs would quietly shrink a set the caller
            believes it is extending.
    """

    prior = pseudonymize_selection_set(SelectionSet.load(seed_set))
    missing = sorted(set(prior.row_ids) - set(traj_by_id))
    if missing:
        raise ValueError(
            f"{seed_set} names {len(missing)} row(s) absent from the new rows file "
            f"({', '.join(missing[:3])}...); the new exam must contain the old one"
        )
    return [tuple(pair) for pair in list(prior.pairs) + list(prior.tied)]


def union_pairs(
    planned: list[tuple[str, str]], carried: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """`planned` plus whatever `carried` adds, deduped on the unordered pair."""

    out = list(planned)
    seen = {frozenset(pair) for pair in planned}
    for pair in carried:
        key = frozenset(pair)
        if key not in seen:
            seen.add(key)
            out.append(pair)
    return out


def build(
    rows_path: Path,
    judge: _PairJudgeLike,
    *,
    pairs: PairCount,
    seed: int,
    exclude: Sequence[Path],
    out: Path,
    seed_set: Path | None = None,
    max_workers: int = 8,
    judge_model: str | None = None,
    extra_provenance: dict[str, Any] | None = None,
    pairs_from: Path | None = None,
) -> SelectionSet:
    """Plan, judge and write the set; return it.

    ``pairs="all"`` plans C(n, 2) pairs with no stratification; an int plans
    a seed-fixed uniform sample (`plan_selection_pairs` with no initial
    scores, so no run's scorer draw is involved). An existing `out` whose
    pairs and ties are identical is left in place — its `built_at` then
    still says when the verdicts were first frozen.

    The rows are stripped at load as a run strips them, and the provenance is
    stamped ``metadata_free: true``, which is what a run checks before it
    accepts the set.

    ``pairs_from`` replaces the plan with an existing set's pair list —
    resolved, tied and unanswered alike — so the new file asks exactly the
    questions that set asked and only the verdicts are new.

    Raises:
        ValueError: A row is in an excluded pool, or the judge resolved no pair.
    """

    rows_path = Path(rows_path)
    rows = load_trajectories(rows_path)
    traj_by_id = {t.trajectory_id: t for t in rows}
    if len(traj_by_id) != len(rows):
        raise ValueError(f"{rows_path} has duplicate trajectory ids")
    assert_excluded(traj_by_id, exclude)
    assert_uniform_fields(rows)

    plan = plan_selection_pairs(sorted(traj_by_id), n_pairs=pairs, seed=seed, initial_scores=None)
    if pairs_from is not None:
        # The rows were pseudonymized at load; so are the pairs asked of them.
        source = pseudonymize_selection_set(SelectionSet.load(pairs_from))
        planned = [tuple(p) for p in [*source.pairs, *source.tied, *source.unresolved]]
        outside = sorted({i for p in planned for i in p} - set(traj_by_id))
        if outside:
            raise ValueError(
                f"{pairs_from} names {len(outside)} row(s) absent from {rows_path}: "
                f"{', '.join(outside[:3])}"
            )
        plan = replace(plan, pairs=planned)
    logger.info(
        "%d rows, %d pairs planned (%s)%s",
        len(traj_by_id),
        len(plan.pairs),
        plan.pairs_mode,
        "" if plan.pairs_mode == "all" else f", seed {seed}",
    )
    carried: list[tuple[str, str]] = []
    if seed_set is not None:
        carried = carried_pairs(Path(seed_set), traj_by_id)
        before = len(plan.pairs)
        plan = replace(plan, pairs=union_pairs(plan.pairs, carried))
        logger.info(
            "carried %d already-judged pairs from %s; plan %d -> %d (%d of the carried "
            "were drawn anyway)",
            len(carried),
            seed_set,
            before,
            len(plan.pairs),
            before + len(carried) - len(plan.pairs),
        )
    provenance: dict[str, Any] = {
        **plan.provenance(),
        "rows_file": str(rows_path),
        "rows_sha256": file_sha256(rows_path),
        "n_rows": len(traj_by_id),
        "exclude": [str(p) for p in exclude],
        "builder": "scripts.build_selection_set",
        "seed_set": str(seed_set) if seed_set is not None else None,
        "n_carried": len(carried),
        "pairs_from": str(pairs_from) if pairs_from is not None else None,
        **(extra_provenance or {}),
        "metadata_free": True,
    }
    if plan.pairs_mode == "all":
        provenance["seed"] = None
    selection_set = judge_selection_pairs(
        plan.pairs, traj_by_id, judge, max_workers=max_workers, provenance=provenance
    )
    if judge_model:
        selection_set.provenance["judge_model"] = judge_model
    if not selection_set.pairs:
        raise ValueError(
            f"the judge resolved 0 of {len(plan.pairs)} planned pairs "
            f"({len(selection_set.tied)} TIE, {len(selection_set.unresolved)} unanswered)"
        )

    out = Path(out)
    if out.is_file():
        existing = SelectionSet.load(out)
        if existing.pair_signature() == selection_set.pair_signature():
            logger.info("%s already holds these %d pairs; left unchanged", out, len(existing))
            return existing
        logger.warning(
            "%s exists with different pairs (%d resolved vs %d now); overwriting",
            out,
            len(existing),
            len(selection_set),
        )
    selection_set.save(out)
    logger.info(
        "wrote %s: %d rows, %d resolved, %d TIE, %d unresolved of %d planned",
        out,
        len(selection_set.row_ids),
        len(selection_set),
        len(selection_set.tied),
        len(selection_set.unresolved),
        len(plan.pairs),
    )
    return selection_set


def make_judge(model: str, cache_dir: Path, extra_body: dict | None) -> _PairJudgeLike:
    """The optimizer's `PairJudge` on an OpenAI-compatible client over OpenRouter.

    Same client class, base URL and metering role as `cli run` builds for
    the judge, so the verdicts land in the same cache files a run reads.
    """

    from meta_filter.llm.codex_cli import read_provider_key
    from meta_filter.llm.openai_compatible import OpenAICompatibleClient
    from meta_filter.pair_judge import PairJudge

    client = OpenAICompatibleClient(
        model=model,
        api_key=read_provider_key("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        extra_body=extra_body,
    )
    return PairJudge(client=metered(client, JUDGE_ROLE), cache_dir=Path(cache_dir))


def log_cost(mark: dict[str, RoleCost]) -> RoleCost:
    spent = LEDGER.since(mark).get(JUDGE_ROLE, RoleCost())
    logger.info(
        "judge cost: %d call(s), $%.4f, %d tokens%s",
        spent.calls,
        spent.cost_usd,
        spent.tokens,
        f" ({spent.unpriced_calls} unpriced)" if spent.unpriced_calls else "",
    )
    return spent


def _parse_extra_body(raw: str | None) -> dict | None:
    if not raw:
        return None
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise argparse.ArgumentTypeError("--judge-extra-body must be a JSON object")
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", type=Path, required=True, help="JSONL of the selection rows")
    parser.add_argument("--judge-model", required=True, help="OpenRouter model slug")
    parser.add_argument(
        "--judge-cache-dir",
        type=Path,
        required=True,
        help="pair-judge cache the runs reading this set also use",
    )
    parser.add_argument("--pairs", type=parse_pairs, default=ALL_PAIRS, help="'all' (default) or N")
    parser.add_argument("--seed", type=int, default=4242, help="seed for --pairs N")
    parser.add_argument(
        "--exclude", type=Path, nargs="*", default=[], help="pools the rows must be disjoint from"
    )
    parser.add_argument(
        "--seed-set",
        type=Path,
        default=None,
        help=(
            "an earlier set whose judged pairs are carried into this plan, so "
            "extending an exam re-reads paid verdicts instead of re-buying them "
            "and the new exam is a superset of the old"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--judge-extra-body",
        type=_parse_extra_body,
        default=None,
        help="JSON merged into every judge request (provider pin, reasoning off)",
    )
    parser.add_argument(
        "--pairs-from",
        type=Path,
        default=None,
        help="re-judge exactly the pairs of this existing set instead of planning new ones",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    judge = make_judge(args.judge_model, args.judge_cache_dir, args.judge_extra_body)
    mark = LEDGER.snapshot()
    started = datetime.now(UTC)
    try:
        build(
            args.rows,
            judge,
            pairs=args.pairs,
            seed=args.seed,
            exclude=list(args.exclude),
            out=args.out,
            seed_set=args.seed_set,
            max_workers=args.max_workers,
            judge_model=args.judge_model,
            extra_provenance={
                "judge_cache_dir": str(args.judge_cache_dir),
                "judge_extra_body": args.judge_extra_body,
            },
            pairs_from=args.pairs_from,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        log_cost(mark)
        return 2
    log_cost(mark)
    logger.info("done in %.0fs", (datetime.now(UTC) - started).total_seconds())
    return 0


if __name__ == "__main__":
    sys.exit(main())
