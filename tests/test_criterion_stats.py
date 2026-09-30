"""criterion_stats: deadness, redundancy, penalty and rendering."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from meta_filter.criterion_stats import (
    criterion_stats,
    redundancy_penalty,
    render_criterion_health,
)


def _decisions(columns: dict[str, list[float]]):
    n = len(next(iter(columns.values())))
    return [SimpleNamespace(rubric_scores={c: v[i] for c, v in columns.items()}) for i in range(n)]


def _spread(seed: int, n: int = 40) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in rng.integers(-2, 3, size=n)]


def test_dead_axis_and_duplicate_pair_are_reported() -> None:
    live = _spread(0)
    stats = criterion_stats(
        _decisions(
            {
                "Live A": live,
                "Copy of A": list(live),  # exact duplicate: |rho| = 1
                "Other": _spread(1),
                "Dead": [0.0] * 39 + [1.0],  # 97.5% on one anchor
            }
        )
    )
    assert stats is not None
    assert stats["per_criterion"]["Dead"]["dead"] is True
    assert stats["per_criterion"]["Live A"]["dead"] is False
    assert stats["max_abs_corr"] > 0.99
    assert set(stats["max_pair"]) == {"Live A", "Copy of A"}
    # 3 live axes, one of them a copy: well under nominal
    assert stats["n_live"] == 4 or stats["n_live"] == 3
    assert stats["effective_rank"] < stats["n_live"] - 0.5


def test_independent_axes_keep_high_effective_rank_and_no_penalty() -> None:
    stats = criterion_stats(_decisions({f"C{i}": _spread(i + 10) for i in range(4)}))
    assert stats is not None
    assert stats["effective_rank"] > 3.0
    assert redundancy_penalty(stats, weight=0.5, threshold=0.7) == 0.0


def test_penalty_ramps_to_weight_at_perfect_duplication() -> None:
    live = _spread(2)
    stats = criterion_stats(_decisions({"A": live, "B": list(live), "C": _spread(3)}))
    pen = redundancy_penalty(stats, weight=0.5, threshold=0.7)
    assert abs(pen - 0.5) < 0.02  # |rho| = 1 -> full weight
    assert redundancy_penalty(stats, weight=0.0, threshold=0.7) == 0.0
    assert redundancy_penalty(None, weight=0.5, threshold=0.7) == 0.0


def test_render_marks_dead_and_redundant() -> None:
    live = _spread(4)
    stats = criterion_stats(
        _decisions(
            {
                "A": live,
                "B": list(live),
                "Dead": [0.0] * 40,
            }
        )
    )
    text = render_criterion_health(stats, threshold=0.7)
    assert "DEAD AXIS" in text
    assert "re-measure" in text
    assert "effective rank" in text


def test_too_few_criteria_returns_none() -> None:
    assert criterion_stats(_decisions({"Only": _spread(5)})) is None
    assert render_criterion_health(None, threshold=0.7) == ""


def test_the_render_reports_the_share_at_every_score_level() -> None:
    """Modal share alone cannot distinguish two very different failures.

    "80% at +1" is a widen-the-anchors problem when the other 20% is spread
    over four levels, and a this-axis-has-two-states problem when it all sits
    on one neighbour. The critics asked about criterion quality carry weight
    0, so this deterministic block is what the
    proposer reads instead — it has to carry the distinction.
    """

    stats = criterion_stats(
        _decisions(
            {
                "Split": [1.0] * 30 + [-2.0] * 10,
                "Spread": [1.0] * 30 + [-2.0, -1.0, 0.0, 2.0] * 2 + [3.0] * 2,
            }
        )
    )
    text = render_criterion_health(stats, threshold=0.7)

    assert "score levels:" in text
    assert "+1:75%" in text
    assert "-2:25%" in text
    # And the redundancy readout the penalty is computed from is present.
    assert "|rho|=" in text
