"""Build a training subset from an evolved ranking filter.

A ranking filter's whole point is its ordering, so this scores every eligible
pool row (rules free, rubric via the per-criterion-cached LLM scorer) and then
makes one of these cuts:

    --top N            the N highest-scoring rows, ties broken by a seeded
                       shuffle. A fixed budget, comparable across filters.
    --accepts-only     with --top: rank only the rows the policy ACCEPTS.
    --accept-all       every row the policy ACCEPTs, at whatever size that is.
    --accept-random N  N rows drawn at random from the ACCEPTs: the control
                       for --top at the same budget, with the ordering
                       discarded.

The pool is stripped at load exactly as a loop run strips it, so the scan
scores the same view of every row the loop did, and with the loop's own
scorer settings (`--scorer-engine openai_compatible --scorer-model <the
run's solver model> --scorer-cache-dir <the run's scorer cache>`) it reads
the per-criterion cells the run already paid for. Everything the subset
writes is keyed by the row's id in the pool file.

Rows are copied byte for byte from a rendered superset (`--superset-dir`,
see `scripts.package_subset`); `--eligibility-ledger`, when given, maps
trajectory ids to superset row ids and restricts the scan to the rows it
lists, otherwise the map is the identity over the pool.

Usage:
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \\
    uv run python -m scripts.build_ranking_subset \\
        --filter-dir runs/<run_id>/iterations/<k>/current_filter \\
        --pool data/nemotron_superset.jsonl --superset-dir data/nemotron_superset \\
        --scorer-engine openai_compatible --scorer-model openai/gpt-5.6-luna \\
        --scorer-cache-dir data/scorer_cache \\
        --top 1000 --name nemotron_blank_1k \\
        --readme README_subset.md --note "one line for manifest/recipe" \\
        --out runs/_subsets
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import random
import shutil
from pathlib import Path

from meta_filter.filter_loader import load_filter
from meta_filter.llm import build_client
from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient
from meta_filter.llm.cost_meter import LEDGER, metered
from meta_filter.llm_scorer import LLMRubricScorer
from meta_filter.metadata_free import filter_outcome_reads, pseudonym_map
from meta_filter.solver import Solver
from meta_filter.trajectory import load_trajectories
from scripts.package_subset import index_superset
from scripts.sft_config import SFT_YAML_NAME, write_sft_yaml

logger = logging.getLogger("build_ranking_subset")

# What can answer the rubric: the local Claude Code CLI, or an
# OpenAI-compatible endpoint built exactly as the loop builds its solver.
CLAUDE_CODE = "claude_code"
OPENAI_COMPATIBLE = "openai_compatible"


def build_scorer_client(engine: str, *, model: str, auth: str):
    """The client the rubric scorer calls, for one engine.

    `claude_code` shells out to the local CLI.

    `openai_compatible` is built the way `meta_filter/cli.py` builds the
    loop's solver — `metered(build_client("openai", model=...), "solver")`,
    reading OPENAI_API_KEY / OPENAI_BASE_URL from the environment. Both
    halves matter:

    * the cache key. `LLMRubricScorer._model_name()` reads `client.model` and
      `MeteredClient` proxies it, so a scan run this way files its cells under
      the namespace holding the cells the loop already bought. Name the model
      any other way and the scan re-buys the pool at full price.
    * the spend. The meter is what puts this scan's calls in `LEDGER`, so the
      scan can report what it cost.
    """

    if engine == OPENAI_COMPATIBLE:
        return metered(build_client("openai", model=model), "solver")
    return ClaudeCodeHeadlessClient(model=model, auth=auth)


def superset_provenance(superset_dir: Path) -> tuple[str, str]:
    """(source, recipe `render:` line) for one superset, read off its manifest."""

    manifest_path = superset_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = str(manifest.get("name") or superset_dir.name).split("@")[0]
    source = str(manifest.get("source") or superset_dir)
    return source, f"row-selection from {name} (identical bytes to every other subset)"


def cut_label(args) -> str:
    """How this subset was cut, in the one form the record uses everywhere."""

    if args.accept_all:
        return "accept-all"
    if args.accept_random:
        floor = f"-floor{args.score_floor:g}" if args.score_floor is not None else ""
        return f"accept-random-{args.accept_random}{floor}"
    if args.accepts_only:
        return f"top-{args.top}-of-accepts"
    return f"top-{args.top}"


def cut_sentence(args) -> str:
    if args.accept_all:
        return "every ACCEPT, no top-K truncation"
    if args.accept_random:
        boundary = (
            f"every row scoring >= {args.score_floor:g}"
            if args.score_floor is not None
            else "every ACCEPT"
        )
        return (
            f"{args.accept_random} rows drawn at random from {boundary} "
            f"(seed {args.seed}), the filter's ordering deliberately discarded"
        )
    if args.accepts_only:
        return f"top-{args.top} by score among the rows the policy ACCEPTS"
    return f"top-{args.top} by score, ACCEPT or not"


def choose_rows(
    scored: dict,
    *,
    seed: int,
    top: int,
    accept_all: bool,
    accept_random: int = 0,
    accepts_only: bool = False,
    score_floor: float | None = None,
    whose: object = "the filter",
) -> tuple[list[str], list[str], float, int, int]:
    """Make the cut. Returns (score order, kept, lowest kept score, boundary
    bucket kept, boundary bucket total).

    The shuffle-then-sort is the tie-break: rows sharing a score come out in a
    seeded order, so a top-K cut through the middle of a tied group is
    reproducible instead of dependent on dict insertion order.

    `accept_random` is the control for `--top`: N rows drawn at random from
    everything the policy ACCEPTS. `--top` asks what the filter's ORDERING is
    worth; this asks what its ACCEPT BOUNDARY is worth on its own, at the same
    budget. Both come out of one scan, because both are readings of the same
    scores.
    """

    if accept_random and accept_all:
        raise SystemExit("--accept-random and --accept-all are different cuts; pick one")
    if accepts_only and (accept_all or accept_random):
        raise SystemExit("--accepts-only narrows --top; the other two cuts are already ACCEPT-only")
    if not accept_all and not accept_random and len(scored) < top:
        raise SystemExit(f"only {len(scored)} scored rows for a top-{top} subset")

    # Ordered for every cut: --top slices it, --accept-all filters it, and
    # either way the ledger comes out in score order.
    order = list(scored)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda t: -scored[t].rank_score)

    # `--score-floor` replaces the policy's own ACCEPT with a line the caller
    # supplies. The loop's objective is invariant to the policy's ACCEPT
    # threshold, so the threshold is inherited from the seed rather than
    # tuned, and where to draw the line is a deployment decision.
    def kept_by_boundary(t: str) -> bool:
        if score_floor is not None:
            return scored[t].rank_score >= score_floor
        return str(scored[t].decision) == "ACCEPT"

    if accept_random:
        accepted = [t for t in order if kept_by_boundary(t)]
        if len(accepted) < accept_random:
            raise SystemExit(
                f"{whose} accepted {len(accepted)} of {len(scored)} scored rows, "
                f"fewer than the {accept_random} this cut asks for. The accept "
                "boundary is the result here, so widen the request rather than "
                "the threshold."
            )
        # A fresh stream, so the draw does not inherit the tie-break shuffle's
        # position and quietly favour the top of the order.
        kept = random.Random(seed + 1).sample(accepted, accept_random)
        kept.sort(key=lambda t: -scored[t].rank_score)
        lowest = scored[kept[-1]].rank_score
        return order, kept, lowest, 0, 0

    if not accept_all:
        pool = order
        if accepts_only:
            # `--top` alone reads the ORDERING, which is what the loop
            # optimised, so a top-N cut can contain rows the filter itself
            # would reject. `--accepts-only` makes the threshold a hard
            # filter and the ranking orders what survives it.
            pool = [t for t in order if kept_by_boundary(t)]
            if len(pool) < top:
                raise SystemExit(
                    f"{whose} accepted {len(pool)} of {len(scored)} scored rows, "
                    f"fewer than the top-{top} this cut asks for. Drop "
                    f"--accepts-only to rank across every row, or ask for a "
                    f"smaller budget — do not move the threshold to fit."
                )
        kept = pool[:top]
        lowest = scored[kept[-1]].rank_score
        bucket_total = sum(1 for t in pool if scored[t].rank_score == lowest)
        bucket_kept = sum(1 for t in kept if scored[t].rank_score == lowest)
        return order, kept, lowest, bucket_kept, bucket_total

    kept = [t for t in order if kept_by_boundary(t)]
    if not kept:
        raise SystemExit(
            f"{whose} accepted 0 of {len(scored)} scored rows, so there is no "
            "accept-all subset to build. Check the policy's ACCEPT threshold against "
            "this pool's score distribution before spending the scan again."
        )
    # No boundary bucket: membership is the policy's decision, not a rank, so
    # the tie-break seed neither keeps nor drops any row.
    return order, kept, min(scored[t].rank_score for t in kept), 0, 0


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Separate from `main` so its defaults are readable in a test."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--filter-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True, help="JSONL pool to score")
    parser.add_argument(
        "--eligibility-ledger",
        type=Path,
        default=None,
        help="JSONL of {trajectory_id, row_id}: the rows to scan and their superset "
        "row ids. Default: every pool row, keyed by its own trajectory_id.",
    )
    parser.add_argument("--superset-dir", type=Path, required=True)
    parser.add_argument(
        "--sft-template",
        type=Path,
        default=None,
        help=f"training config to render this subset's identifiers into "
        f"(default: <superset-dir>/{SFT_YAML_NAME}).",
    )
    parser.add_argument(
        "--scorer-cache-dir",
        type=Path,
        default=Path("data/scorer_cache"),
        help="rubric-score cache; point it at the run's own to reuse its cells.",
    )
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument(
        "--accept-all",
        action="store_true",
        help="keep every ACCEPT instead of the top --top by score.",
    )
    parser.add_argument(
        "--score-floor",
        type=float,
        default=None,
        help="use this rank-score line as the accept boundary instead of the "
        "policy's own ACCEPT. The policy's threshold is a seed constant the "
        "loop does not tune, so the line is a deployment choice.",
    )
    parser.add_argument(
        "--accepts-only",
        action="store_true",
        help="rank only the rows the policy ACCEPTS, then take --top of those. "
        "Without it --top reads the ordering alone and can keep a row the "
        "filter would reject.",
    )
    parser.add_argument(
        "--accept-random",
        type=int,
        default=0,
        help="keep N rows drawn at random from everything the policy ACCEPTS. "
        "The control for --top at the same budget: it keeps the accept boundary "
        "and throws the ordering away. One scan serves both.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scorer-model", default="sonnet")
    parser.add_argument("--scorer-auth", default="api_key")
    parser.add_argument(
        "--scorer-engine",
        choices=(CLAUDE_CODE, OPENAI_COMPATIBLE),
        default=CLAUDE_CODE,
        help="what answers the rubric. claude_code (default) shells out to the "
        "local CLI. openai_compatible calls an OpenAI-compatible endpoint "
        "(OPENAI_API_KEY / OPENAI_BASE_URL) built exactly as the loop builds its "
        "solver, so the scan reads the scorer-cache cells the loop already "
        "bought; pass the loop's own --scorer-model with it, and --scorer-auth "
        "is then unused.",
    )
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument(
        "--source-uri",
        default=None,
        help="provenance written to manifest.json and recipe.yaml; "
        "default: read from the superset's own manifest.json",
    )
    parser.add_argument(
        "--render-note",
        default=None,
        help="the recipe's `render:` line; default: derived from --superset-dir",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # `sonnet` is a claude CLI alias; an endpoint wants a slug. Left at the
    # default, --scorer-engine openai_compatible would key every cell under a
    # namespace no run has filled, so the scan would pay for the whole pool
    # again and still not be comparable to the run that produced the filter.
    if args.scorer_engine == OPENAI_COMPATIBLE and args.scorer_model == parser.get_default(
        "scorer_model"
    ):
        raise SystemExit(
            f"--scorer-engine {OPENAI_COMPATIBLE} with --scorer-model "
            f"{args.scorer_model!r}: that is a claude CLI alias, not an endpoint "
            "model id, and scoring under it would namespace this scan's cache away "
            "from the cells the loop bought. Pass the loop's own --scorer-model "
            "(e.g. openai/gpt-5.6-luna)."
        )

    src_manifest, render_note = superset_provenance(args.superset_dir)
    src_manifest = args.source_uri or src_manifest
    render_note = args.render_note or render_note
    logger.info("superset: %s (source %s)", args.superset_dir, src_manifest)

    if (args.accept_all or args.accept_random) and args.top != parser.get_default("top"):
        raise SystemExit("--accept-all and --top are two different cuts; pass one")

    if args.eligibility_ledger is None:
        row_id_of = {
            json.loads(line)["trajectory_id"]: json.loads(line)["trajectory_id"]
            for line in args.pool.open(encoding="utf-8")
            if line.strip()
        }
        logger.info("eligibility: %d rows, identity map from the pool", len(row_id_of))
    else:
        eligible = [json.loads(line) for line in args.eligibility_ledger.open() if line.strip()]
        row_id_of = {r["trajectory_id"]: r["row_id"] for r in eligible}
        logger.info("eligibility: %d rows from %s", len(eligible), args.eligibility_ledger)
    if not row_id_of:
        source = args.pool if args.eligibility_ledger is None else args.eligibility_ledger
        raise SystemExit(f"{source} lists no eligible trajectories")

    pool = load_trajectories(args.pool)
    # The stripped rows carry pseudonymized ids (scored and cached under them,
    # so the scan reuses the loop's cells); everything the subset writes is
    # keyed by the id in the pool file, recomputed from the file itself.
    to_raw = pseudonym_map(args.pool)
    by_id: dict[str, object] = {}
    for t in pool:
        by_id.setdefault(to_raw.get(t.trajectory_id, t.trajectory_id), t)
    trajs = [by_id[tid] for tid in row_id_of if tid in by_id]
    logger.info("matched in pool: %d of %d eligible", len(trajs), len(row_id_of))
    # A join that lands under half is not attrition, it is two different
    # pools. Stop before paying the scorer to rank the wrong rows.
    if len(trajs) * 2 < len(row_id_of):
        raise SystemExit(
            f"only {len(trajs)} of {len(row_id_of)} eligible ids are in {args.pool} — "
            f"{args.eligibility_ledger} and the pool disagree; nothing was scored"
        )

    reads = filter_outcome_reads(args.filter_dir)
    if reads:
        raise SystemExit(
            f"refusing {args.filter_dir}: "
            + "; ".join(f"{name} names {', '.join(v)}" for name, v in reads.items())
        )
    loaded = load_filter(args.filter_dir)
    criteria = list(getattr(loaded.policy_module, "RUBRIC_CRITERIA", []))
    if not criteria:
        raise SystemExit(f"{args.filter_dir} declares no RUBRIC_CRITERIA — not a ranking filter")
    scorer = LLMRubricScorer(
        client=build_scorer_client(
            args.scorer_engine, model=args.scorer_model, auth=args.scorer_auth
        ),
        cache_dir=args.scorer_cache_dir,
        max_workers=args.parallel,
    )
    logger.info(
        "prefetching rubric scores (%s / %s, %d workers)",
        args.scorer_engine,
        args.scorer_model,
        args.parallel,
    )
    scorer.prefetch_parallel(trajs, criteria, loaded.rubric_md)
    if args.scorer_engine == OPENAI_COMPATIBLE:
        # Only the metered engine has anything to report; under claude_code
        # the ledger is empty, which would read as "this scan was free".
        logger.info("scan spend so far: %s", LEDGER.columns())
    solver = Solver(rubric_scorer=scorer, lazy_scoring=False)
    decisions = solver.run(loaded, trajs)

    scored = {
        to_raw.get(d.trajectory_id, d.trajectory_id): d
        for d in decisions
        if d.rank_score is not None
    }
    skipped = len(trajs) - len(scored)
    logger.info("scored %d rows (%d without a rank score)", len(scored), skipped)
    order, top, cut, bucket_kept, bucket_total = choose_rows(
        scored,
        seed=args.seed,
        top=args.top,
        accept_all=args.accept_all,
        accept_random=args.accept_random,
        accepts_only=args.accepts_only,
        score_floor=args.score_floor,
        whose=args.filter_dir,
    )
    levels = len({round(scored[t].rank_score, 4) for t in scored})
    logger.info(
        "%s: %d rows, lowest kept score=%.2f (boundary bucket %d/%d by seed), "
        "%d distinct levels over %d scored",
        cut_label(args),
        len(top),
        cut,
        bucket_kept,
        bucket_total,
        levels,
        len(scored),
    )

    cut_desc = cut_sentence(args)
    dest = args.out / args.name
    dest.mkdir(parents=True, exist_ok=True)
    row_ids = [row_id_of[t] for t in top]
    index = index_superset(args.superset_dir / "train.jsonl")
    missing = [r for r in row_ids if r not in index]
    if missing:
        raise SystemExit(f"{len(missing)} rows missing from superset: {missing[:3]}")

    train = dest / "train.jsonl"
    with train.open("w", encoding="utf-8") as fh:
        for rid in row_ids:
            fh.write(index[rid])
    md5 = hashlib.md5(train.read_bytes()).hexdigest()
    import pandas as pd

    pd.read_json(train, lines=True).to_parquet(dest / "train.parquet", index=False)
    rows = [json.loads(index[rid]) for rid in row_ids]
    # A superset whose rows carry no `instance_id` has no instance count; an
    # all-null set is "not applicable", not one instance.
    instance_ids = {r.get("instance_id") for r in rows}
    n_instances = None if instance_ids <= {None} else len(instance_ids - {None})
    n_instances_yaml = "null" if n_instances is None else str(n_instances)

    selected = set(top)
    with (dest / "filter_decisions.jsonl").open("w", encoding="utf-8") as fh:
        for t in order:
            d = scored[t]
            fh.write(
                json.dumps(
                    {
                        "row_id": row_id_of[t],
                        "trajectory_id": t,
                        "rank_score": d.rank_score,
                        "decision": str(d.decision),
                        "rubric_scores": d.rubric_scores,
                        "fired_rules": d.triggered_rules,
                        "selected": t in selected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "name": args.name.split("@")[0],
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "n_rows": len(row_ids),
        "n_instances": n_instances,
        "md5": md5,
        "note": args.note,
        "source": src_manifest,
        "filter": str(args.filter_dir) + f" (ranking filter, {cut_desc})",
        "cut_score": cut,
        "distinct_levels": levels,
        "boundary_bucket": [bucket_kept, bucket_total],
        "seed": args.seed,
        "row_ids": row_ids,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dest / "recipe.yaml").write_text(
        f"""# Ranking-filter subset
name: {args.name.split("@")[0]}
source: {src_manifest}
render: {render_note}
rule: |
  {args.note}
seed: {args.seed}
n_rows: {len(row_ids)}
n_instances: {n_instances_yaml}
filter: same folder `filter/` — the evolved ranking filter, run as-is; {cut_desc}
ledger: same folder `filter_decisions.jsonl` (per-row score + criterion readings + selected flag)
""",
        encoding="utf-8",
    )
    fdir = dest / "filter"
    fdir.mkdir(exist_ok=True)
    for fname in ("rubric.md", "rules.py", "policy.py"):
        shutil.copyfile(args.filter_dir / fname, fdir / fname)
    shutil.copyfile(args.readme, dest / "README.md")
    # Rendered, not copied, so the training config names this subset (see
    # scripts/sft_config.py).
    template = args.sft_template or (args.superset_dir / SFT_YAML_NAME)
    if not template.is_file():
        raise SystemExit(
            f"no training config at {template}. Every delivered subset ships one — "
            "pass --sft-template pointing at the frozen recipe to render from."
        )
    write_sft_yaml(template, dest, args.name)
    logger.info(
        "packaged -> %s (md5 %s, %s instances)",
        dest,
        md5,
        "n/a" if n_instances is None else n_instances,
    )
    # Machine-readable stats.
    (dest / ".build_stats.json").write_text(
        json.dumps(
            {
                "cut_score": cut,
                "n_rows": len(row_ids),
                "n_scored": len(scored),
                "cut": cut_label(args),
                "n_instances": n_instances,
                "distinct_levels": levels,
                "bucket_kept": bucket_kept,
                "bucket_total": bucket_total,
                "md5": md5,
                "skipped": skipped,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
