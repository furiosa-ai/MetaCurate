"""The cache-key components every LLM cache in the loop is filed under.

Changing any of these re-keys a cache, so every verdict and score already on
disk stops being found. That may be the right call one day; it must never be
an accident.
"""

from __future__ import annotations

from meta_filter.critics.trajectory_critic import TrajectoryCritic
from meta_filter.pair_judge import PAIR_TOOL_BUDGET, PairJudge
from meta_filter.rendering import DEFAULT_TRAJECTORY_TOOL_BUDGET, view_signature
from meta_filter.schemas import Trajectory
from meta_filter.view_policy import VIEW_SIGNATURE_FIELDS, pair_view_signature

# sha1("f2p_pass,p2p_pass,resolved")[:8]: every stripped row lacks all three.
STRIPPED_VIEW = "6fd64b73"


def _traj(tid: str = "t", **metadata) -> Trajectory:
    return Trajectory(trajectory_id=tid, metadata=metadata)


def test_view_signatures_are_pinned():
    assert view_signature(PAIR_TOOL_BUDGET) == "859ab0b4"
    assert view_signature(DEFAULT_TRAJECTORY_TOOL_BUDGET) == "12289822"


def test_a_stripped_pair_has_the_one_stripped_view():
    a = _traj("a", source="x", submission="diff")
    b = _traj("b")
    assert pair_view_signature(a, b) == STRIPPED_VIEW
    assert pair_view_signature(a) == STRIPPED_VIEW


def test_the_view_depends_on_which_fields_are_absent():
    full = {name: True for name in VIEW_SIGNATURE_FIELDS}
    assert pair_view_signature(_traj(**full)) == ""
    one_missing = dict(full, resolved=None)
    assert pair_view_signature(_traj(**one_missing)) not in ("", STRIPPED_VIEW)


def test_the_pair_cache_path_carries_the_view(tmp_path):
    judge = PairJudge(client=object(), cache_dir=tmp_path)
    plain = judge._cache_path("a", "b")
    assert "__lv_" not in plain.name
    keyed = judge._cache_path("a", "b", STRIPPED_VIEW)
    assert f"__lv_{STRIPPED_VIEW}" in keyed.name and keyed != plain


def test_the_trajectory_critic_keys_the_view(tmp_path):
    class _C:
        model = "opus"

    critic = TrajectoryCritic(client=_C(), cache_dir=tmp_path)
    critic.register_label_views([_traj("t1")])
    assert f"__lv_{STRIPPED_VIEW}" in critic.cache_path_for("t1").name


def test_an_id_only_reader_finds_the_one_verdict_on_file(tmp_path):
    """An unregistered read accepts the single verdict present for an id and,
    faced with several, reports absence rather than picking one."""

    class _C:
        model = "opus"

    writer = TrajectoryCritic(client=_C(), cache_dir=tmp_path)
    writer.register_label_views([_traj("t1")])
    written = writer.cache_path_for("t1")
    written.write_text("{}", encoding="utf-8")

    reader = TrajectoryCritic(client=_C(), cache_dir=tmp_path)
    assert reader.cache_path_for("t1") == written

    other = tmp_path / written.name.replace("__lv_", "__lv_ffffffff__x_")
    other.write_text("{}", encoding="utf-8")
    ambiguous = TrajectoryCritic(client=_C(), cache_dir=tmp_path)
    assert "__lv_" not in ambiguous.cache_path_for("t1").name
