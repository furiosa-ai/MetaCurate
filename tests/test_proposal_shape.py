"""Whether a candidate edited the baseline or replaced it.

The proposer is told to copy `current_filter/` and edit the copy. When it does
not — candidates carrying 50-78% new lines — nothing accumulates between
generations and the scorer cache, keyed on the rubric text, misses on every
trajectory of every candidate.
"""

from __future__ import annotations

import pytest

from meta_filter.proposal_shape import REWRITE_THRESHOLD, measure, report

BASELINE_RUBRIC = "\n".join(f"line {i}" for i in range(40))
BASELINE_RULES = "\n".join(f"def rule_{i}(t): return None" for i in range(20))
BASELINE_POLICY = "THRESHOLD = 0.5\n"


def _filter(root, rubric=BASELINE_RUBRIC, rules=BASELINE_RULES, policy=BASELINE_POLICY):
    root.mkdir(parents=True, exist_ok=True)
    (root / "rubric.md").write_text(rubric, encoding="utf-8")
    (root / "rules.py").write_text(rules, encoding="utf-8")
    (root / "policy.py").write_text(policy, encoding="utf-8")
    return root


def test_a_targeted_edit_reads_as_an_edit(tmp_path):
    """One new rule plus the policy branch that reads it — the shape a single
    hypothesis is supposed to have."""

    base = _filter(tmp_path / "base")
    cand = _filter(
        tmp_path / "cand",
        rules=BASELINE_RULES + "\ndef rule_new(t): return {'verdict': 'REJECT'}",
        policy=BASELINE_POLICY + "USE_NEW_RULE = True\n",
    )

    shape = measure(base, cand)
    assert not shape.is_rewrite
    assert shape.novelty < 0.1
    assert set(shape.files_touched) == {"rules.py", "policy.py"}


def test_a_rewrite_reads_as_a_rewrite(tmp_path):
    """Most lines replaced across all three files while claiming to be an
    improvement on the baseline."""

    base = _filter(tmp_path / "base")
    cand = _filter(
        tmp_path / "cand",
        rubric="\n".join(f"different {i}" for i in range(40)),
        rules="\n".join(f"def other_{i}(t): return None" for i in range(20)),
        policy="THRESHOLD = 0.9\n",
    )

    shape = measure(base, cand)
    assert shape.is_rewrite
    assert shape.novelty >= REWRITE_THRESHOLD
    assert len(shape.files_touched) == 3


def test_an_untouched_copy_has_no_novelty(tmp_path):
    base = _filter(tmp_path / "base")
    cand = _filter(tmp_path / "cand")

    shape = measure(base, cand)
    assert shape.novelty == 0.0
    assert shape.files_touched == ()


def test_a_missing_baseline_does_not_crash_the_iteration(tmp_path):
    """Shape is diagnostic. A run must not die because a directory moved."""

    cand = _filter(tmp_path / "cand")
    shape = measure(tmp_path / "absent", cand)
    assert shape.novelty == 1.0  # everything is new when there is no baseline


def test_rewrites_are_reported_not_rejected(tmp_path, caplog):
    """A rewrite can still be the right answer — the proposer holds the
    evidence. The loop's job is to make the drift visible, not to veto it."""

    base = _filter(tmp_path / "base")
    edited = _filter(tmp_path / "edited", rules=BASELINE_RULES + "\ndef rule_new(t): return None")
    rewritten = _filter(
        tmp_path / "rewritten",
        rubric="\n".join(f"x {i}" for i in range(40)),
        rules="\n".join(f"def z_{i}(t): return None" for i in range(20)),
    )

    shapes = report(base, [edited, rewritten])

    assert len(shapes) == 2  # nothing was dropped
    assert [s.is_rewrite for s in shapes] == [False, True]
    assert "rewrote rather than edited" in caplog.text


# --- kept lines --------------------------------------------------------------
#
# Novelty alone reads the diff from one side only. A candidate that deletes
# most of a file and adds a handful of lines has a *low* addition count and is
# still a replacement: a candidate that shrinks the rubric rather than growing
# it would read as an edit.


def test_a_deletion_rewrite_reads_as_a_rewrite(tmp_path):
    """Two thirds of the rubric deleted, nothing added: novelty says edit."""

    base = _filter(tmp_path / "base")
    cand = _filter(
        tmp_path / "cand",
        rubric="\n".join(f"line {i}" for i in range(12)),
    )

    shape = measure(base, cand)
    assert shape.novelty < REWRITE_THRESHOLD  # the old check would pass it
    assert shape.min_kept_share == pytest.approx(12 / 40)
    assert shape.is_rewrite


def test_keeping_most_of_every_file_stays_an_edit(tmp_path):
    """The boundary: half the rubric's lines gone is still an edit, one more
    is not. A rewrite check that fires on ordinary pruning is noise."""

    base = _filter(tmp_path / "base")
    kept_half = _filter(tmp_path / "half", rubric="\n".join(f"line {i}" for i in range(20)))
    kept_less = _filter(tmp_path / "less", rubric="\n".join(f"line {i}" for i in range(19)))

    assert not measure(base, kept_half).is_rewrite
    assert measure(base, kept_less).is_rewrite


def test_a_deleted_file_is_a_rewrite_of_that_file(tmp_path):
    base = _filter(tmp_path / "base")
    cand = _filter(tmp_path / "cand")
    (cand / "rules.py").unlink()

    shape = measure(base, cand)
    assert shape.min_kept_share == 0.0
    assert shape.is_rewrite


def test_reindenting_keeps_the_lines(tmp_path):
    """Lines are compared as a set of stripped bodies, so indenting a block
    under a new heading is not mistaken for deleting it. (Novelty still counts
    the moved lines as added — that half of the measurement is unchanged.)"""

    base = _filter(tmp_path / "base")
    indented = "\n".join((f"    line {i}" if i < 10 else f"line {i}") for i in range(40))
    cand = _filter(tmp_path / "cand", rubric=indented)

    shape = measure(base, cand)
    assert shape.min_kept_share == 1.0
    assert not shape.is_rewrite


def test_kept_share_is_reported_per_file(tmp_path):
    base = _filter(tmp_path / "base")
    cand = _filter(tmp_path / "cand", rubric="\n".join(f"line {i}" for i in range(10)))

    shape = measure(base, cand)
    shares = dict(shape.kept_shares)
    assert shares["rubric.md"] == pytest.approx(0.25)
    assert shares["rules.py"] == 1.0
    assert shares["policy.py"] == 1.0
    assert "25% of rubric.md" in shape.describe()


def test_an_absent_baseline_has_nothing_to_keep(tmp_path):
    """No parent lines means no kept-lines opinion — novelty already calls it."""

    cand = _filter(tmp_path / "cand")
    shape = measure(tmp_path / "absent", cand)
    assert shape.kept_shares == ()
    assert shape.min_kept_share == 1.0
    assert shape.is_rewrite  # from novelty, as before
