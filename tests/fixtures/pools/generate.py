"""Regenerate the synthetic pools the end-to-end tests run on.

    uv run python -m tests.fixtures.pools.generate

Two pool shapes, one per data source the method is run on:

* ``dsv4pro_*`` rows are shaped like mini-SWE-agent rows whose metadata still
  carries the harness grading fields (``resolved``, ``f2p_pass``,
  ``p2p_pass``, ``exit_status``) and the row identifiers (``instance_id``,
  ``host``, ``attempt_idx``). Every one of those must be stripped at load;
  their values carry ``SENTINEL`` so a scan finds any that leaks.
* ``nemotron_*`` rows carry only ``source``, ``dataset``, ``license`` and
  ``submission``, and pass the strip unchanged.

Every transcript is synthetic: short shell sessions over made-up
repositories, varied enough (edits, tests, reverts, disallowed git calls,
debug prints in the diff, empty submissions) that both initial filters' rules
fire on some rows and not on others.

The ``*_selection_judged.json`` files are selection verdict files built by
``scripts.build_selection_set.build`` with the deterministic mock judge, on
the stripped rows. The DSv4Pro one keeps the builder's ``metadata_free``
stamp; the Nemotron one has it removed and is accepted because the rows file
it names carries no outcome field.
"""

from __future__ import annotations

import json
import logging
import random
import tempfile
from pathlib import Path

from meta_filter.gate.pair_judge import PairJudge
from meta_filter.gate.selection_set import SelectionSet
from meta_filter.llm.mock import MockLLMClient

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SENTINEL = "SENTINEL"

SIZES = {"train": 16, "probe": 4, "selection": 10}
SEEDS = {"dsv4pro": 101, "nemotron": 202}

REPOS = ("acme__pager", "globex__ledger", "initech__tps", "umbrella__cache", "hooli__search")
FILES = ("src/pager.py", "src/core/util.py", "lib/ledger.py", "app/cache.py", "pkg/search.py")

SYSTEM_TEXT = (
    "You are a helpful assistant that can interact with a computer shell to solve "
    "programming tasks."
)


def _task(rng: random.Random, repo: str, i: int) -> str:
    issue = rng.choice(
        (
            "an off-by-one error when the last page is empty",
            "a KeyError when the config omits the optional section",
            "a crash when the input list contains None",
            "wrong rounding of negative amounts",
            "a stale cache entry after invalidation",
        )
    )
    return (
        "<pr_description>\nConsider the following PR description:\n"
        f"{repo.replace('__', '/')} #{100 + i}: fix {issue}.\n</pr_description>"
    )


def _session(rng: random.Random, path: str) -> tuple[list[dict], str]:
    """(turns after the task, submitted diff) for one synthetic attempt."""

    steps: list[tuple[str, str]] = []
    explore = rng.randint(1, 8)
    for _ in range(explore):
        steps.append(
            rng.choice(
                (
                    ("ls", "ls -la /testbed"),
                    ("find", f"find /testbed -name '*.py' | grep {path.split('/')[-1][:-3]}"),
                    ("grep", f"grep -rn 'def ' /testbed/{path}"),
                    ("cat", f"cat /testbed/{path}"),
                )
            )
        )
    reproduced = rng.random() < 0.6
    if reproduced:
        steps.append(("python3", "cd /testbed && python3 reproduce.py"))
    n_edits = rng.randint(0, 3)
    for k in range(n_edits):
        steps.append(("sed", f"sed -i 's/limit - {k}/limit - {k + 1}/' /testbed/{path}"))
        if rng.random() < 0.5:
            steps.append(("pytest", "cd /testbed && python -m pytest tests -q"))
    if rng.random() < 0.15:
        steps.append(("git_checkout", f"cd /testbed && git checkout -- {path}"))
    if rng.random() < 0.1:
        steps.append(("git_stash", "cd /testbed && git stash"))
    if rng.random() < 0.5:
        steps.append(("pytest", "cd /testbed && python -m pytest tests -q"))
    steps.append(("git_diff", f"cd /testbed && git diff -- {path} > patch.txt"))
    steps.append(("echo", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"))

    added = [f"+    return limit - {n_edits}"]
    if rng.random() < 0.2:
        added.append('+    print("debug", limit)')
    submission = (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,2 +1,2 @@\n"
        "-    return limit\n" + "\n".join(added) + "\n"
        if n_edits and rng.random() > 0.1
        else ""
    )

    turns: list[dict] = []
    for action, command in steps:
        turns.append({"role": "assistant", "action": action, "content": command})
        failed = action == "pytest" and rng.random() < 0.4
        body = "1 failed, 12 passed" if failed else "ok"
        result = f"<returncode>{1 if failed else 0}</returncode>\n<output>\n{body}\n</output>"
        turns.append({"role": "tool", "action": "tool_result", "content": result})
    return turns, submission


def _row(source: str, rng: random.Random, split: str, i: int) -> dict:
    repo = REPOS[i % len(REPOS)]
    path = FILES[(i + len(split)) % len(FILES)]
    task = _task(rng, repo, i)
    head = [
        {"role": "system", "action": None, "content": SYSTEM_TEXT},
        {"role": "user", "action": None, "content": task},
    ]
    session, submission = _session(rng, path)
    turns = head + session
    if source == "nemotron":
        return {
            "trajectory_id": f"nemotron.{split}-{i:04d}-{rng.getrandbits(32):08x}",
            "task": task,
            "turns": turns,
            "metadata": {
                "source": "nemotron-v3-mini-xml",
                "submission": submission,
                "license": "cc-by-4.0",
                "dataset": "nemotron-mini",
            },
        }
    resolved = bool(submission) and rng.random() < 0.5
    return {
        "trajectory_id": f"{repo}-{split}{i}::att{i % 3}",
        "task": task,
        "turns": turns,
        "metadata": {
            "source": "mini-swe-agent-1.1",
            "instance_id": f"{SENTINEL}-{repo}-{split}{i}",
            "host": f"{SENTINEL}-host",
            "attempt_idx": i % 3,
            "api_calls": len(session) // 2,
            "n_messages": len(turns),
            "submission": submission,
            "submission_bytes": len(submission),
            "exit_status": f"{SENTINEL}-Submitted",
            "resolved": resolved,
            "f2p_pass": resolved,
            "p2p_pass": resolved or rng.random() < 0.5,
        },
    }


def pool_path(source: str, split: str) -> Path:
    return HERE / f"{source}_{split}.jsonl"


def judged_path(source: str) -> Path:
    return HERE / f"{source}_selection_judged.json"


def write_pools() -> None:
    for source, seed in SEEDS.items():
        rng = random.Random(seed)
        for split, n in SIZES.items():
            rows = [_row(source, rng, split, i) for i in range(n)]
            pool_path(source, split).write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
            )


def write_selection_sets() -> None:
    """Build both verdict files, recording repo-relative paths in their provenance."""

    import os

    from scripts.build_selection_set import build

    os.chdir(REPO)

    def rel(path: Path) -> Path:
        return path.relative_to(REPO)

    for source in SEEDS:
        with tempfile.TemporaryDirectory() as cache:
            judge = PairJudge(client=MockLLMClient(), cache_dir=Path(cache))
            build(
                rel(pool_path(source, "selection")),
                judge,
                pairs="all",
                seed=4242,
                exclude=[rel(pool_path(source, "train")), rel(pool_path(source, "probe"))],
                out=rel(judged_path(source)),
                judge_model="mock-1",
            )
        if source == "nemotron":
            # The builder always stamps the set as judged on stripped rows.
            # The Nemotron fixture drops the stamp so the tests exercise the
            # other acceptance route: a rows file that carries no outcome field.
            judged = SelectionSet.load(judged_path(source))
            judged.provenance.pop("metadata_free", None)
            judged.save(judged_path(source))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    write_pools()
    for source in SEEDS:
        judged_path(source).unlink(missing_ok=True)
    write_selection_sets()


if __name__ == "__main__":
    main()
