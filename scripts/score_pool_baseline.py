"""Score a pool with a baseline grader, one ledger row per trajectory.

Two baselines the method is compared against:

* `grm` — a generated rubric (`--rubric`) applied by a model, which returns a
  `total_score`; a subset is the top rows by that score.
* `llmfilter` — a model asked, with no rubric, whether to keep each
  trajectory; a subset is drawn from the keeps.

The pool is stripped at load exactly as a loop run strips it, so a baseline
sees the same view of every row the method does: the transcript, the
submitted diff and process counts, and no outcome field. The ledger holds one
row per trajectory keyed by its id in the pool file, so every subset size is
a pure re-cut afterwards. Per-row caching makes a run resumable.

Usage:
    uv run python -m scripts.score_pool_baseline --mode grm --rubric <rubric.md> \\
        --pool data/nemotron_superset.jsonl --out data/nemotron_grm_ledger.jsonl \\
        --limit 5                                                   # trial run
    uv run python -m scripts.score_pool_baseline --mode llmfilter \\
        --pool data/nemotron_superset.jsonl --out data/nemotron_llmfilter_ledger.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from meta_filter.concurrency import max_concurrent_calls
from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient
from meta_filter.metadata_free import pseudonym_map
from meta_filter.rendering import render_submission, render_turns
from meta_filter.trajectory import load_trajectories

logger = logging.getLogger("score_pool_baseline")

TOOL_BUDGET = 30_000
TASK_CAP = 8_000
PATCH_CAP = 20_000

LLMFILTER_SYSTEM = (
    "You are filtering SWE agent trajectories to build an SFT training set.\n"
    "You will receive an issue, an agent trajectory, and the final patch. Decide whether\n"
    "this trajectory should be KEPT as training data. Use your own judgment - there is\n"
    "no rubric. Reply ONLY with a JSON object: "
    '{"keep": true|false, "reason": "<one sentence>"}'
)

BODY = """\
=== ISSUE ===
{task}

=== REPO ===
{repo}

=== TRAJECTORY ===
{turns}

=== FINAL PATCH ===
{submission}
"""


def _ids(path: Path) -> set[str]:
    return {json.loads(line)["trajectory_id"] for line in path.open() if line.strip()}


def _render(traj) -> str:
    meta = traj.metadata or {}
    return BODY.format(
        task=(traj.task or "(no task)")[:TASK_CAP],
        repo=str(meta.get("repo") or meta.get("instance_id") or "(unknown)"),
        turns=render_turns(traj, tool_output_budget=TOOL_BUDGET),
        submission=render_submission(traj)[:PATCH_CAP],
    )


def _grm_prompt(traj, rubric: str) -> str:
    return (
        rubric + "\n\n---\n\nScore the trajectory below against the rubric above. "
        "No execution results are available to you.\n\n" + _render(traj)
    )


def _llmfilter_prompt(traj) -> str:
    return LLMFILTER_SYSTEM + "\n\n" + _render(traj)


def _parse_grm(raw: str) -> dict | None:
    m = re.search(r'"total_score"\s*:\s*(\d+)', raw)
    if not m:
        return None
    keep = re.search(r'"decision"\s*:\s*"(keep|drop)"', raw)
    return {
        "total_score": int(m.group(1)),
        "decision": keep.group(1) if keep else None,
    }


def _parse_llmfilter(raw: str) -> dict | None:
    m = re.search(r'"keep"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if not m:
        return None
    return {"keep": m.group(1).lower() == "true"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["grm", "llmfilter"], required=True)
    parser.add_argument("--pool", type=Path, required=True, help="JSONL pool to score")
    parser.add_argument(
        "--eligibility",
        type=Path,
        default=None,
        help="restrict to the trajectory ids in this JSONL (uniform exclusions "
        "across baselines); default: the whole pool",
    )
    parser.add_argument("--rubric", type=Path, default=Path("data/grm_generated_rubric.md"))
    parser.add_argument("--model", default="opus")
    parser.add_argument(
        "--auth", default=None, help="oauth | api_key (default: env META_FILTER_CLAUDE_AUTH)"
    )
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="trial run: first N rows")
    parser.add_argument(
        "--memory-share",
        type=float,
        default=1.0,
        help="fraction of free memory this job may claim; pass 0.5 to each of "
        "two jobs that will overlap (they cannot see each other at startup)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="per-row score cache (default: data/{mode}_rescore_cache)",
    )
    parser.add_argument("--out", type=Path, required=True, help="ledger output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cache_dir = args.cache_dir or Path(f"data/{args.mode}_rescore_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out

    eligible = None
    if args.eligibility is not None:
        if not args.eligibility.exists():
            raise SystemExit(f"--eligibility {args.eligibility} does not exist")
        eligible = _ids(args.eligibility)
        logger.info("eligibility %s restricts to %d ids", args.eligibility, len(eligible))

    pool = load_trajectories(args.pool)
    # The stripped rows carry pseudonymized ids; the ledger, the eligibility
    # filter and the resume check are keyed by the id in the pool file,
    # recomputed from it.
    to_raw = pseudonym_map(args.pool)

    def raw_id(traj) -> str:
        return to_raw.get(traj.trajectory_id, traj.trajectory_id)

    seen: set[str] = set()
    trajs = []
    for t in pool:
        if raw_id(t) in seen:
            continue
        if eligible is not None and raw_id(t) not in eligible:
            continue
        seen.add(raw_id(t))
        trajs.append(t)
    if args.limit:
        trajs = trajs[: args.limit]

    # Zero rows is a failure with a name on it, not an empty ledger.
    if not trajs:
        detail = f"{len(pool)} rows in {args.pool}"
        if eligible is not None:
            detail += f", {len(eligible)} eligible ids in {args.eligibility}, 0 in common"
        raise SystemExit(f"no trajectories to score: {detail}")

    # A ledger already at --out must describe THIS pool before a run's worth
    # of calls writes over it: zero overlap is the wrong-ledger case, never a
    # resume.
    if out_path.exists() and out_path.stat().st_size:
        try:
            head = {
                json.loads(line)["trajectory_id"]
                for line in itertools.islice(out_path.open(), 50)
                if line.strip()
            }
        except (json.JSONDecodeError, KeyError):
            head = set()
        if head and not (head & seen):
            raise SystemExit(
                f"{out_path} already holds a ledger for a DIFFERENT pool "
                f"(e.g. {sorted(head)[0]!r}, none of which are in {args.pool}). "
                "Pass --out for this pool's ledger, or delete that file."
            )

    logger.info("%s: scoring %d trajectories with %s", args.mode, len(trajs), args.model)

    rubric = ""
    rubric_md5 = "none"
    if args.mode == "grm":
        rubric = args.rubric.read_text(encoding="utf-8")
        rubric_md5 = hashlib.md5(rubric.encode("utf-8")).hexdigest()
        logger.info("rubric: %s (%d chars, md5 %s)", args.rubric, len(rubric), rubric_md5[:8])

    # A headless call is its own process, so the only memory lever is how
    # many run at once; the governor sizes the pool against memory actually
    # free.
    ceiling = max_concurrent_calls(memory_share=args.memory_share)
    if ceiling and ceiling < args.parallel:
        logger.warning("governor: parallel %d -> %d (memory)", args.parallel, ceiling)
        args.parallel = ceiling
    logger.info("parallel workers: %d", args.parallel)

    kwargs = {"model": args.model}
    if args.auth:
        kwargs["auth"] = args.auth
    client = ClaudeCodeHeadlessClient(**kwargs)
    safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", s)  # noqa: E731

    # The cache key is {trajectory_id}__{model}; each entry also carries the
    # rubric's md5, and a mismatch is a miss, so a new --rubric never reads a
    # score earned under the old one.
    def score_one(traj) -> tuple[str, dict | None]:
        cache = cache_dir / f"{safe(traj.trajectory_id)}__{args.model}.json"
        if cache.exists():
            try:
                entry = json.loads(cache.read_text())
                if entry.get("rubric_md5") == rubric_md5:
                    return traj.trajectory_id, entry["parsed"]
            except (json.JSONDecodeError, KeyError):
                pass
        prompt = _grm_prompt(traj, rubric) if args.mode == "grm" else _llmfilter_prompt(traj)
        try:
            raw = client.generate(prompt).text
        except Exception as exc:  # noqa: BLE001 — a lost row is a skipped row
            logger.warning("call failed for %s: %s", traj.trajectory_id, type(exc).__name__)
            return traj.trajectory_id, None
        parsed = _parse_grm(raw) if args.mode == "grm" else _parse_llmfilter(raw)
        cache.write_text(
            json.dumps(
                {
                    "trajectory_id": traj.trajectory_id,
                    "parsed": parsed,
                    "rubric_md5": rubric_md5,
                    "raw": raw[:2000],
                }
            )
        )
        return traj.trajectory_id, parsed

    results: dict[str, dict | None] = {}
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool_exec:
        futs = {pool_exec.submit(score_one, t): t for t in trajs}
        for fut in as_completed(futs):
            tid, parsed = fut.result()
            results[tid] = parsed
            done += 1
            failed += parsed is None
            if done % 25 == 0 or done == len(trajs):
                logger.info("progress: %d/%d (unparsed %d)", done, len(trajs), failed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for t in trajs:
            r = results.get(t.trajectory_id)
            fh.write(
                json.dumps(
                    {"trajectory_id": raw_id(t), **(r or {}), "parsed": r is not None},
                    ensure_ascii=False,
                )
                + "\n"
            )
    ok = [r for r in results.values() if r]
    logger.info("wrote %s — %d scored, %d unparsed", out_path, len(ok), failed)
    # The ledger is written first so the failure is inspectable, then the run
    # fails: a full-length ledger of "parsed": false rows is not a deliverable.
    if not ok:
        raise SystemExit(f"{len(trajs)} trajectories scored, 0 parsed verdicts — see {out_path}")
    if args.mode == "grm":
        scores = sorted((r["total_score"] for r in ok), reverse=True)
        logger.info(
            "score spread: max %d, p10 %d, median %d, min %d",
            scores[0],
            scores[len(scores) // 10],
            scores[len(scores) // 2],
            scores[-1],
        )
    if args.mode == "llmfilter":
        keeps = sum(1 for r in ok if r.get("keep"))
        logger.info("keep rate %d/%d = %.1f%%", keeps, len(ok), 100 * keeps / len(ok))


if __name__ == "__main__":
    main()
