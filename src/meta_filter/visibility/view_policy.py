"""Which metadata fields each LLM-facing role may see, in one place.

Hardcoding the answer to "what does the judge get to read?" separately in
each role — an allow-list in `llm_scorer`, another in `pair_judge`, the whole
dict in `critics/agentic` — leaves nobody knowing about the others, while a
further party, `rubric.md`, issues instructions against yet another list. A
rubric criterion that tells the scorer to read `metadata.gold_patch` while
the scorer's allow-list does not render it makes that criterion a constant:
the scorer answers, correctly, that a field it never received is missing,
and the axis orders nothing while still costing a call per row. When the
policy averages over all criteria, two such constants also compress the
rubric term to two thirds of its designed range, tilting the rule/rubric
balance toward the rules.

## The invariant this module exists to hold

    a role renders a field  <=>  a rubric told it to, and the role is allowed to

So the allow-list is not something a person has to remember to widen. A
criterion declares what it reads, in its own section:

    ## Criterion: Task Relation
    <!-- reads: problem_statement -->

and the field arrives, if `SCORER_RENDERABLE` permits it. Declaring is
*additive* on a per-role baseline: a rubric that declares nothing gets exactly
the baseline fields, byte for byte, which is what keeps every existing filter
and every existing cache entry valid.

`RENDERABLE` is the ceiling, and it is not a formality. The proposer rewrites
`rubric.md` every iteration, so without a ceiling it could declare
`instance_id` and rebuild the memorization key `pseudonym.py` removes, or
declare `resolved` and open an undisclosed label echo. The pair judge's
ceiling equals its baseline: it decides the truth pairs every score is
measured against, and `PAIR_PROMPT` tells it to "judge the process, not just
the outcome".

## Why not simply render everything

`submission` can run to tens of megabytes on some trace corpora and is
already rendered as its own budgeted block. And a field that reaches a judge
without anyone deciding it should is the same failure as a field that does
not reach one -- just harder to notice. The list stays explicit; only the
*authority* over it moves, from memory to the rubric.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping

# --- per-role baselines: exactly what each role renders undeclared ----------

# `llm_scorer.SCORER_PROMPT`'s metadata block: the allow-listed keys a row
# may carry (`metadata_free.KEPT_METADATA_FIELDS`) that are short scalars.
# Order is irrelevant here -- the block is built by walking
# `trajectory.metadata` and filtering on membership, so the rendered order is
# the metadata dict's, not this set's.
SCORER_BASELINE_FIELDS: frozenset[str] = frozenset(
    {"api_calls", "n_messages", "submission_bytes", "source"}
)

# The pair judge renders no metadata field; its metadata line is the fixed
# `pair_judge.PAIR_METADATA_LINE`.
PAIR_JUDGE_FIELDS: tuple[str, ...] = ()

# Cache-key compatibility, and nothing else: the scorer's field-set and
# absent-field signatures are computed over the rendered fields plus these
# five names, which no loaded row carries. Every scorer cache entry already on
# disk was keyed that way; dropping them from the signature input would
# re-key the whole scorer cache. They are never rendered.
_SCORER_KEY_ONLY_FIELDS: frozenset[str] = frozenset(
    {"exit_status", "instance_id", "resolved", "f2p_pass", "p2p_pass"}
)


def scorer_key_fields(fields: Iterable[str]) -> frozenset[str]:
    """The field set a scorer cache-key signature is computed over."""

    return frozenset(fields) | _SCORER_KEY_ONLY_FIELDS


# --- ceilings: how far a rubric may widen its own role ----------------------

# `gold_patch` is the reference fix and `problem_statement` the issue text.
# `test_patch` is deliberately absent: no criterion has ever asked for it, and
# it names the hidden tests that gate the row -- closer to reading the answer
# key than the reference diff is.
SCORER_RENDERABLE: frozenset[str] = SCORER_BASELINE_FIELDS | {"problem_statement"}

# The truth channel cannot be widened. See the module docstring.
PAIR_JUDGE_RENDERABLE: frozenset[str] = frozenset(PAIR_JUDGE_FIELDS)

_ROLES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "scorer": (SCORER_BASELINE_FIELDS, SCORER_RENDERABLE),
    "pair_judge": (frozenset(PAIR_JUDGE_FIELDS), PAIR_JUDGE_RENDERABLE),
}

# Per-field character caps, applied only to fields that appear here. Every
# baseline field is a short scalar and has no entry, so a baseline render is
# byte-identical to what it was before this module existed.
#
# 4000 is not arbitrary: `rendering.DEFAULT_SUBMISSION_BUDGET` caps the
# agent's own diff at the same number. A criterion comparing the two diffs
# has to see them under equal budgets or it is scoring the truncation.
FIELD_CAPS: dict[str, int] = {"problem_statement": 4000}

# Fields rendered as their own raw block instead of inside the metadata JSON,
# with the label to head them with.
#
# `json.dumps` puts a diff on one line with every newline escaped to `\n`,
# while the agent's own diff arrives as raw text under "Final submitted
# patch:". A criterion asked to compare the two would be reading one of them
# in a materially harder format than the other, which defeats the point of
# giving them equal budgets. Long text gets a block; short scalars stay in the
# JSON where they are easier to scan.
RAW_BLOCK_FIELDS: dict[str, str] = {"problem_statement": "Full problem statement"}

_READS_RE = re.compile(r"<!--\s*reads:\s*([^>]*?)\s*-->", re.IGNORECASE)
_FIELD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# A criterion section that references a metadata field without declaring it.
# Used by the dead-axis check, which is why it lives beside the policy.
METADATA_REF_RE = re.compile(r"metadata\.([a-z_][a-z0-9_]*)")


def declared_fields(text: str) -> frozenset[str]:
    """Fields a rubric (or one criterion's section) declares that it reads.

    Reads every `<!-- reads: a, b -->` comment in `text`, so a rubric may
    declare per criterion or once at the top. Unknown names are kept here and
    filtered by `fields_for`, so a typo shows up as "declared but not
    renderable" rather than silently doing nothing.
    """

    found: set[str] = set()
    for block in _READS_RE.findall(text or ""):
        found.update(_FIELD_RE.findall(block))
    return frozenset(found)


def fields_for(role: str, declared: Iterable[str] = ()) -> frozenset[str]:
    """The fields `role` renders, given what the rubric declared.

    `baseline | (declared & renderable)`. The union is what makes every
    existing rubric -- none of which declares anything -- render exactly the
    fields it rendered before.
    """

    try:
        baseline, renderable = _ROLES[role]
    except KeyError:
        raise ValueError(f"unknown role: {role!r} (known: {sorted(_ROLES)})") from None
    return baseline | (frozenset(declared) & renderable)


def unrenderable(role: str, declared: Iterable[str]) -> frozenset[str]:
    """Declared fields this role is not allowed to render, for diagnostics."""

    _, renderable = _ROLES[role]
    return frozenset(declared) - renderable


def field_set_signature(role: str, fields: Iterable[str]) -> str:
    """Cache-key component for a rendered field set. Empty at the baseline.

    Empty-at-baseline is the whole point: it is what keeps the ~28k scorer
    cache entries written before this module readable. Only a set that
    differs from the role's baseline gets a suffix, and then it gets its own
    namespace instead of silently reusing scores taken without those fields.

    Without this component the fix that motivated this module is invisible:
    adding `gold_patch` to the scorer changes the prompt but not the criterion
    name, its section text, the view or the model -- so the key is unchanged,
    the old `0` is served from cache, and nothing says so.
    """

    baseline, _ = _ROLES[role]
    fields = frozenset(fields)
    if role == "scorer":
        baseline, fields = scorer_key_fields(baseline), scorer_key_fields(fields)
    if fields == baseline:
        return ""
    return hashlib.sha1(",".join(sorted(fields)).encode("utf-8")).hexdigest()[:8]


def absent_field_signature(fields: Iterable[str], metadata: Mapping[str, object]) -> str:
    """Cache-key component for declared fields this row does not have.

    `field_set_signature` records which fields a criterion *declares*; it is
    read off the rubric text and never looks at the row. That is the whole key
    as far as metadata goes, so one trajectory id scored under one rubric gets
    one key whether or not the row actually carries what the rubric asked for
    — and `scorer_metadata_block` filters on membership, so a declared field
    the row lacks simply does not render. Two genuinely different prompts,
    one key.

    That collision is reachable the moment the same pool exists in a labelled
    and a stripped form: every stripped row shares its trajectory id with a
    labelled one, and a label-free run would read cells a labelled run wrote
    with `resolved` in the prompt. The label-free result would be a labelled
    one, silently.

    Empty at baseline, for the same reason `field_set_signature` is: a row
    that has everything its rubric declared keeps the key it already has, so
    no cache entry written before this existed becomes unreachable. Only a row
    MISSING something gets its own namespace.

    Args:
        fields: The fields this criterion declared, from the rubric.
        metadata: The row's metadata, the mapping the prompt block walks.

    Returns:
        ``""`` when every declared field is present, else a short digest of
        the absent ones.
    """

    absent = sorted(f for f in frozenset(fields) if f not in metadata)
    if not absent:
        return ""
    return "no" + hashlib.sha1(",".join(absent).encode("utf-8")).hexdigest()[:8]


def _capped(name: str, value: object) -> object:
    cap = FIELD_CAPS.get(name)
    if cap is None or not isinstance(value, str) or len(value) <= cap:
        return value
    return value[:cap] + f"\n... [{len(value) - cap} chars elided]"


def scorer_metadata_block(metadata: Mapping[str, object], fields: Iterable[str]) -> str:
    """The scorer prompt's `Metadata:` block.

    Walks `metadata` and filters on membership -- not the other way round --
    because that is what the inlined version did, and the rendered key order
    is part of the prompt bytes the cache is keyed on.

    `RAW_BLOCK_FIELDS` are excluded here; they go through
    `scorer_reference_blocks` instead.
    """

    allowed = frozenset(fields) - frozenset(RAW_BLOCK_FIELDS)
    return json.dumps(
        {k: _capped(k, v) for k, v in metadata.items() if k in allowed},
        indent=2,
    )


def scorer_reference_blocks(metadata: Mapping[str, object], fields: Iterable[str]) -> str:
    """Raw labelled blocks for the long-text fields a criterion declared.

    Empty string when nothing declared one, and the caller splices it in
    without a separator of its own, so a baseline prompt is byte-identical to
    the prompt that existed before this module.
    """

    allowed = frozenset(fields)
    out = []
    for name, label in RAW_BLOCK_FIELDS.items():
        if name not in allowed:
            continue
        value = metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        out.append(f"\n\n{label}:\n{_capped(name, value)}")
    return "".join(out)


# The fields whose absence the pair-judge and trajectory-critic cache keys
# record (`pair_view_signature`). Rows are stripped at load, so every row lacks
# all three and the component is the same on every key. It is computed from
# the names rather than hardcoded so the key format stays exactly what the
# existing caches were written under.
VIEW_SIGNATURE_FIELDS = frozenset({"resolved", "f2p_pass", "p2p_pass"})


def pair_view_signature(*trajectories: object) -> str:
    """Cache-key component naming which `VIEW_SIGNATURE_FIELDS` the rows lack.

    Empty only when every field is present and non-null on every trajectory
    given; otherwise the first 8 hex digits of the sha1 of the absent names,
    sorted and comma-joined. Computed per pair because the pair judge renders
    each trajectory's metadata line separately.
    """

    absent: set[str] = set()
    for traj in trajectories:
        metadata = getattr(traj, "metadata", None) or {}
        get = metadata.get if hasattr(metadata, "get") else (lambda _k: None)
        absent.update(name for name in VIEW_SIGNATURE_FIELDS if get(name) is None)
    if not absent:
        return ""
    return hashlib.sha1(",".join(sorted(absent)).encode("utf-8")).hexdigest()[:8]
