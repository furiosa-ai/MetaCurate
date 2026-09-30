"""A row missing a field its rubric declared must not reuse the labelled cell.

The scorer's cache key is built from the trajectory *id* plus the fields the
criterion DECLARES, read off the rubric text. It never looks at the row. But
`view_policy.scorer_metadata_block` filters on membership, so a declared field
the row does not carry simply does not render — two different prompts, one key.

That is reachable the moment the same pool exists labelled and stripped:
every stripped row shares its trajectory id with a labelled one. Without the
guard below, a label-free run reads cells a labelled run wrote with
`resolved` in the prompt and reports a label-free result that is nothing of
the kind.

`pair_judge` already had this right — its cache path carries a `label_view`
component "which grading fields the judge could actually see", empty for the
all-present case. This is the same idea, finally applied to the scorer.
"""

from __future__ import annotations

from meta_filter.llm_scorer import _criterion_cache_key
from meta_filter.trajectory import Trajectory
from meta_filter.view_policy import absent_field_signature, field_set_signature

DECLARED = ("resolved", "f2p_pass", "exit_status")
LABELLED = {"resolved": True, "f2p_pass": True, "exit_status": "submitted"}
STRIPPED = {"exit_status": "submitted"}


def _traj(metadata: dict) -> Trajectory:
    return Trajectory(trajectory_id="t1", task="t", turns=[], metadata=metadata)


def _sig(fields, metadata) -> str:
    return field_set_signature("scorer", fields) + absent_field_signature(fields, metadata)


def test_a_row_with_every_declared_field_keeps_the_key_it_already_had():
    """The guard is empty at baseline, so no cell written before it existed
    becomes unreachable. This is the same discipline `field_set_signature`
    keeps, and for the same reason: the cache holds tens of thousands of
    entries that cost real money."""

    assert absent_field_signature(DECLARED, LABELLED) == ""
    assert _sig(DECLARED, LABELLED) == field_set_signature("scorer", DECLARED)


def test_the_stripped_row_gets_its_own_namespace():
    labelled = _criterion_cache_key(
        "section", "Outcome", _traj(LABELLED), "m", _sig(DECLARED, LABELLED), "hdr"
    )
    stripped = _criterion_cache_key(
        "section", "Outcome", _traj(STRIPPED), "m", _sig(DECLARED, STRIPPED), "hdr"
    )
    assert labelled != stripped, "the label-free run would read the labelled cell"


def test_two_rows_missing_the_same_fields_still_share_a_key():
    """The guard separates by WHAT is absent, not by which row. Two stripped
    rows of the same pool must still share cells with each other, or a
    label-free run pays for every cell twice over."""

    a = _sig(DECLARED, STRIPPED)
    b = _sig(DECLARED, {"exit_status": "failed"})
    assert a == b


def test_a_criterion_declaring_only_present_fields_is_unaffected():
    """Most criteria read no metadata at all. They must not be dragged into a
    new namespace because some other criterion's field is missing."""

    assert absent_field_signature((), STRIPPED) == ""
    assert absent_field_signature(("exit_status",), STRIPPED) == ""


def test_missing_different_fields_gives_different_namespaces():
    only_resolved_gone = _sig(DECLARED, {"f2p_pass": True, "exit_status": "x"})
    both_gone = _sig(DECLARED, {"exit_status": "x"})
    assert only_resolved_gone != both_gone
