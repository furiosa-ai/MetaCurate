"""Metadata-free rows: no harness outcome ever reaches a component of the loop.

The loop builds a filter from the transcript and the submitted diff alone.
The DSv4Pro pools carry the harness's verdict on every row — `resolved`,
`f2p_pass`, `p2p_pass`, `exit_status` — and a probe/selection bookkeeping
field derived from it (`probe_stratum`, bucketed by per-instance resolved
counts). The Nemotron pool carries none of it: its rows hold `source`,
`dataset`, `license` and `submission`, and nothing else.

So what a row may carry is defined by the Nemotron row, not by a list of
things to hide.
`strip_metadata` keeps an ALLOW-list (`KEPT_METADATA_FIELDS`) and drops every
other metadata key, so the renderers (`pair_judge._meta_line`,
`view_policy.scorer_metadata_block`, the trajectory critic) can only ever show
allow-listed keys, whichever pool a row came from; a Nemotron row, whose keys
are all on the list, passes through unchanged — byte for byte, so every
Nemotron cache entry stays valid. An allow-list rather
than a deny-list because a deny-list is only as good as the author's memory:
a pool that brings `tests_passed` or `eval_report` would sail through one.

What is kept, and why:

* `submission`, `submission_bytes` — the agent's own patch and its size. What
  the agent shipped is the thing being judged.
* `api_calls`, `n_messages` — process statistics of the agent's own run, not
  anything the harness concluded about it.
* `source`, `dataset`, `license` — provenance; constant within a pool.
* `problem_statement` — task text. No pool carries it natively.

What is dropped beyond the outcome fields, and why: `instance_id`, `host` and
`attempt_idx` identify a row rather than grade it, nothing functional reads
them (the pseudonymizer and every dedup key on `trajectory_id`), and
`instance_id` is exactly the memorisation key
`pseudonym.py` exists to remove. For the same reason an id of the form
`<instance_id>::att<n>` is replaced by a deterministic hash of itself
(`pseudonymize_id`), so instance names do not appear in prompts, cache keys
or run files. This is not a secret: the prefix is a fixed public constant,
and anyone with the source pool can recompute the mapping. Ids that name no
instance are untouched.

Top-level keys other than the four `Trajectory` fields are dropped too:
`Trajectory` allows extras, and the demo validation pool carries a
`gold_decision` label that way.

The strip happens once, at load (`trajectory.load_trajectories`), so nothing
downstream can see a dropped field — there is no second copy to forget.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The metadata a metadata-free row may carry. Nemotron's four keys are all
# here, which is what keeps a Nemotron row byte-identical after the strip.
KEPT_METADATA_FIELDS: frozenset[str] = frozenset(
    {
        "source",
        "dataset",
        "license",
        "submission",
        "submission_bytes",
        "api_calls",
        "n_messages",
        "problem_statement",
    }
)

# The four fields a `Trajectory` has; anything else at the top level is an
# extra (`model_config = extra="allow"`) and is dropped.
TRAJECTORY_FIELDS: tuple[str, ...] = ("trajectory_id", "task", "turns", "metadata")

# Harness outcomes, reference material and anything derived from them. The
# strip does not consult this list (it keeps an allow-list); the precheck
# does, to refuse a filter that tries to READ one of them, and the scans in
# the tests use it to prove nothing rendered them.
OUTCOME_FIELDS: tuple[str, ...] = (
    "resolved",
    "f2p_pass",
    "p2p_pass",
    "exit_status",
    "gold_patch",
    "test_patch",
    "grade_status",
    "regrade_resolved",
    "probe_stratum",
    # SWE-bench's own names for the fail-to-pass / pass-to-pass test lists.
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "fail_to_pass",
    "pass_to_pass",
)

# `resolved` is also an English word ("the issue is resolved", "judge-resolved
# pairs"), so it is only a read when it is shaped like a key: quoted,
# backticked, an attribute, a `metadata.` path or a `<!-- reads: -->` entry.
# Every other outcome field has an underscore and cannot occur in prose.
_UNAMBIGUOUS = tuple(name for name in OUTCOME_FIELDS if name != "resolved")
_UNAMBIGUOUS_RE = re.compile(r"\b(" + "|".join(_UNAMBIGUOUS) + r")\b")
_RESOLVED_KEY_RE = re.compile(
    r"""["'`]resolved["'`]"""
    r"""|\.resolved\b"""
    r"""|\bmetadata\W{0,3}resolved\b"""
    r"""|<!--\s*reads:[^>]*\bresolved\b""",
    re.IGNORECASE,
)

# Where a metadata-free run writes its stripped train rows, inside the run
# directory. `config.yaml` names this file as `train_data`, so the scoring
# and rule tools and a tool-using proposer all read the stripped view.
METADATA_FREE_TRAIN_FILE = "train_pool.metadata_free.jsonl"

# The files of a filter directory the precheck reads, rubric included: in
# this mode a criterion that asks the scorer about an outcome is as much a
# label read as a rule that does, and nothing is grandfathered.
FILTER_FILES: tuple[str, ...] = ("rules.py", "policy.py", "rubric.md")


def strip_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """`metadata` restricted to `KEPT_METADATA_FIELDS`, key order preserved.

    Order matters: `scorer_metadata_block` walks the dict, so the rendered
    bytes follow it.
    """

    return {k: v for k, v in (metadata or {}).items() if k in KEPT_METADATA_FIELDS}


# Trajectory ids of the form `<instance_id>::att<n>` (the mini-SWE-agent
# adapter's) carry the task instance in the id itself, and the id reaches the
# judge's header, the scorer's cache key and every file of the run. Such an id
# is replaced by `traj_` + the first 16 hex digits of sha256(PSEUDONYM_SALT +
# id). This hides the instance name from the LLMs; it is not protection — the
# constant is fixed (so cache entries stay reusable across runs), every pool,
# cache and subset builder derives the same id from the same row, and anyone
# holding the source pool maps it back (`pseudonym_map`). Ids without instance
# information — Nemotron's `nemotron.<uuid>` — are left exactly as they are.
PSEUDONYM_SALT = "meta-filter/metadata-free/v1:"
_INSTANCE_ID_RE = re.compile(r"^[^\s:]+::att\d+$")


def id_carries_instance(trajectory_id: str) -> bool:
    return bool(_INSTANCE_ID_RE.match(trajectory_id or ""))


def pseudonymize_id(trajectory_id: str) -> str:
    """The metadata-free id for `trajectory_id`; unchanged if it names no instance."""

    if not id_carries_instance(trajectory_id):
        return trajectory_id
    digest = hashlib.sha256((PSEUDONYM_SALT + trajectory_id).encode("utf-8")).hexdigest()
    return f"traj_{digest[:16]}"


def pseudonym_map(pool: Path) -> dict[str, str]:
    """`{metadata-free id: original id}` over a raw pool file, recomputed from it."""

    out: dict[str, str] = {}
    with Path(pool).open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                raw = str(json.loads(line)["trajectory_id"])
                out[pseudonymize_id(raw)] = raw
    return out


def pseudonymize_selection_set(selection_set: Any) -> Any:
    """A `SelectionSet` with every id passed through `pseudonymize_id`.

    Idempotent: a pseudonym names no instance, so it maps to itself, and a
    set built in this mode (already pseudonymized) or over Nemotron rows comes
    back equal to what went in.
    """

    from dataclasses import replace

    def pairs(items):
        return [(pseudonymize_id(a), pseudonymize_id(b)) for a, b in items]

    return replace(
        selection_set,
        row_ids=[pseudonymize_id(i) for i in selection_set.row_ids],
        pairs=pairs(selection_set.pairs),
        tied=pairs(selection_set.tied),
        unresolved=pairs(selection_set.unresolved),
    )


def strip_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A raw JSONL row as a metadata-free row: four fields, allowed metadata,
    and an id that names no task instance."""

    out = {k: row[k] for k in TRAJECTORY_FIELDS if k in row}
    if "metadata" in out:
        out["metadata"] = strip_metadata(out["metadata"])
    if "trajectory_id" in out:
        out["trajectory_id"] = pseudonymize_id(str(out["trajectory_id"]))
    return out


def outcome_mentions(text: str) -> list[str]:
    """Outcome fields `text` names in a key-shaped way, sorted, deduplicated."""

    found = set(_UNAMBIGUOUS_RE.findall(text or ""))
    if _RESOLVED_KEY_RE.search(text or ""):
        found.add("resolved")
    return sorted(found, key=OUTCOME_FIELDS.index)


def filter_outcome_reads(filter_dir: Path) -> dict[str, list[str]]:
    """`file -> outcome fields it names`, over rules.py / policy.py / rubric.md.

    A token match: a filter that names a field at all is a filter that can
    read it. It reads the rubric too and has no initial-filter exemption — a
    seed that names a label is a broken seed.
    """

    out: dict[str, list[str]] = {}
    for name in FILTER_FILES:
        path = Path(filter_dir) / name
        if path.is_file():
            hits = outcome_mentions(path.read_text(encoding="utf-8"))
            if hits:
                out[name] = hits
    return out


def rows_with_outcomes(path: Path, fields: Iterable[str] | None = None) -> int:
    """How many rows of a raw JSONL pool carry a non-null outcome field.

    Reads the file as it is on disk — the point is to learn what a judge
    would have seen on these rows BEFORE any strip. `fields` defaults to
    every `OUTCOME_FIELDS` name.
    """

    wanted = tuple(fields) if fields is not None else OUTCOME_FIELDS
    count = 0
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            metadata = (json.loads(line) or {}).get("metadata") or {}
            if any(metadata.get(name) is not None for name in wanted):
                count += 1
    return count


def selection_set_view(provenance: Mapping[str, Any]) -> str | None:
    """Why a prebuilt selection set is NOT accepted as label-free, or None.

    Accepted only when one of two things is positively established:

    * its provenance says it was judged on stripped rows (`metadata_free:
      true`, written by `scripts.build_selection_set` and by a run that
      re-judged), or
    * the rows it was judged on — the `rows_file` its provenance names — can
      be opened, are byte-identical to what was judged (`rows_sha256`), and
      carry no outcome field (`OUTCOME_FIELDS`). That is the Nemotron
      exam: judged on rows that never had a label.

    Anything else is refused, including a set whose rows file is missing from
    this machine or has changed since judging: "could not look" is not "clean".
    Relative paths resolve against the working directory, as the builder
    wrote them.
    """

    if provenance.get("metadata_free") is True:
        return None
    raw = provenance.get("rows_file")
    if not raw:
        return (
            "its provenance names no rows_file and does not say it was judged "
            "metadata-free, so nothing shows its verdicts were formed without labels"
        )
    path = Path(str(raw))
    if not path.is_file():
        return (
            f"the rows it was judged on ({raw}) cannot be opened from here, and its "
            "provenance does not say it was judged metadata-free"
        )
    expected = provenance.get("rows_sha256")
    if not expected:
        return (
            f"its provenance records no rows_sha256, so {raw} cannot be shown to be what was judged"
        )
    actual = _sha256(path)
    if actual != expected:
        return (
            f"{raw} has changed since the set was judged "
            f"(sha256 {actual[:12]} != {str(expected)[:12]})"
        )
    n = rows_with_outcomes(path)
    if n:
        return (
            f"{n} row(s) of {raw} carry an outcome field and the set's "
            "provenance does not say it was judged on stripped rows, so its verdicts "
            "may have been formed with those fields in view"
        )
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return _sha256(path)
