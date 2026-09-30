"""No-loss simplification, new rubric defects, frozen files.

Filters are built in `tmp_path`; the simplification test reads nothing but
their file sizes, and the rubric-defect tests synthesise the baseline's
`criterion_stats` — the numbers the criterion-health block would have shown
the proposer.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meta_filter import constraints as cx
from meta_filter.criterion_stats import criterion_stats
from meta_filter.precheck import _constraint_line, run_precheck

# --- fixtures -------------------------------------------------------------------

CRITERIA = ("Alpha", "Beta", "Gamma", "Dead", "Twin")


def _criterion(name: str, body: str) -> str:
    return f"## Criterion: {name}\n\n{body}\n\n+2: yes\n-2: no\n"


def _rubric(names=CRITERIA, header: str = "# Rubric\n\nScale +2 / -2.\n\n", bodies=None) -> str:
    bodies = bodies or {}
    return header + "\n".join(
        _criterion(n, bodies.get(n, f"Does {n.lower()} hold?")) for n in names
    )


RULES = "def rule_one(trajectory):\n    return None\n"
# Same LOC as RULES, so only the freeze (not the rules_loc soft cap) can fire.
RULES_CHANGED = "def rule_one(t):\n    return None\n"
RULES_TWO = RULES + "\n\ndef rule_two(trajectory):\n    return None\n"


def _policy(names=CRITERIA, extra: str = "") -> str:
    items = "".join(f'    "{n}",\n' for n in names)
    return (
        f"RUBRIC_CRITERIA = [\n{items}]\n\nWEIGHT = 1.0\n{extra}\n\n"
        "def decide(trajectory, rubric_scores, triggered_rules):\n"
        "    values = [float(v) for v in rubric_scores.values()]\n"
        "    mean = sum(values) / len(values) if values else 0.0\n"
        "    score = 50 + WEIGHT * mean\n"
        "    return {'decision': 'ACCEPT' if score >= 50 else 'REJECT', 'score': score}\n"
    )


def make_filter(root: Path, name: str, *, rubric: str, rules: str = RULES, policy: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "rubric.md").write_text(rubric, encoding="utf-8")
    (d / "rules.py").write_text(rules, encoding="utf-8")
    (d / "policy.py").write_text(policy, encoding="utf-8")
    return d


def _spread(seed: int, n: int = 50) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in rng.integers(-2, 3, size=n)]


def _decisions(columns: dict[str, list[float]]):
    n = len(next(iter(columns.values())))
    return [SimpleNamespace(rubric_scores={c: v[i] for c, v in columns.items()}) for i in range(n)]


@pytest.fixture
def baseline_stats() -> dict:
    """Dead is dead (100 % at 0), Twin duplicates Alpha (|rho| = 1), the rest live."""

    alpha = _spread(1)
    return criterion_stats(
        _decisions(
            {
                "Alpha": alpha,
                "Beta": _spread(2),
                "Gamma": _spread(3),
                "Dead": [0.0] * 50,
                "Twin": list(alpha),
            }
        )
    )


@pytest.fixture
def baseline(tmp_path) -> Path:
    return make_filter(tmp_path, "baseline", rubric=_rubric(), rules=RULES_TWO, policy=_policy())


def _check(candidate: Path, baseline: Path, **kwargs) -> cx.ConstraintReport:
    return cx.check_candidate(candidate, baseline_dir=baseline, initial_dir=baseline, **kwargs)


# --- is_simplification ------------------------------------------------------------


def test_dropping_a_criterion_is_a_simplification(tmp_path, baseline):
    names = ("Alpha", "Beta", "Gamma", "Twin")
    cand = make_filter(
        tmp_path, "drop_dead", rubric=_rubric(names), rules=RULES_TWO, policy=_policy(names)
    )
    report = _check(cand, baseline)
    assert report.simplification is True
    assert report.deleted == ["Dead"]
    assert report.status == "ok"
    assert "SIMPLIFICATION" in report.summary()


def test_pruning_a_rule_is_a_simplification_even_though_the_rubric_is_untouched(tmp_path, baseline):
    """Cutting rule functions is a simplification too, not only rubric
    edits."""

    cand = make_filter(tmp_path, "prune", rubric=_rubric(), rules=RULES, policy=_policy())
    report = _check(cand, baseline)
    assert report.simplification is True
    assert report.deleted == ["rule_two"]


def test_removing_a_live_axis_is_a_simplification_too(tmp_path, baseline):
    """No rubric statistics are consulted: the axis need not be dead or
    redundant, because `b >= c` on the hidden pairs is what decides whether
    the removal cost anything."""

    names = ("Alpha", "Gamma", "Dead", "Twin")  # Beta is live and independent
    cand = make_filter(
        tmp_path, "drop_beta", rubric=_rubric(names), rules=RULES_TWO, policy=_policy(names)
    )
    assert _check(cand, baseline).simplification is True


def test_rewriting_a_kept_criterion_while_removing_another_still_simplifies(tmp_path, baseline):
    """The kept sections need not be byte-identical: every component still
    falls, so this is a simplification."""

    names = ("Alpha", "Beta", "Gamma", "Twin")
    cand = make_filter(
        tmp_path,
        "drop_and_sharpen",
        rubric=_rubric(names, bodies={"Beta": "Sharper."}),
        rules=RULES_TWO,
        policy=_policy(names),
    )
    assert _check(cand, baseline).simplification is True


def test_a_component_that_grows_disqualifies_a_smaller_total(tmp_path, baseline):
    """A deleted rule is worth 1.0 + 0.05 per line; 200 more rubric characters
    are worth 0.4. The total falls and the candidate is still not a
    simplification — that trade is a confounded edit, not a smaller filter."""

    longer = _rubric(bodies={"Beta": "Does beta hold? " + "x" * 200})
    cand = make_filter(tmp_path, "trade", rubric=longer, rules=RULES, policy=_policy())
    report = _check(cand, baseline)
    assert report.candidate["rubric_chars"] > report.baseline["rubric_chars"]
    assert cx.complexity_score(report.candidate) < cx.complexity_score(report.baseline)
    assert report.simplification is False


def test_an_identical_filter_is_not_a_simplification(tmp_path, baseline):
    cand = make_filter(tmp_path, "same", rubric=_rubric(), rules=RULES_TWO, policy=_policy())
    assert _check(cand, baseline).simplification is False


def test_a_growing_candidate_is_not_a_simplification(tmp_path, baseline):
    names = (*CRITERIA, "Extra")
    cand = make_filter(
        tmp_path, "grow", rubric=_rubric(names), rules=RULES_TWO, policy=_policy(names)
    )
    assert _check(cand, baseline).simplification is False


def test_the_report_round_trips_the_simplification_flag(tmp_path, baseline):
    cand = make_filter(tmp_path, "prune", rubric=_rubric(), rules=RULES, policy=_policy())
    payload = _check(cand, baseline).to_dict()
    assert payload["simplification"] is True
    assert payload["new_dead_axes"] == [] and payload["new_redundant_pairs"] == []
    assert payload["frozen_changed"] == []
    assert "cleanup" not in payload


# --- new rubric defects ------------------------------------------------------


def test_a_new_dead_axis_needs_strong_evidence(baseline_stats):
    cand = criterion_stats(
        _decisions({"Alpha": _spread(1), "Beta": _spread(2), "Fresh": [1.0] * 48 + [0.0, 2.0]})
    )
    report = cx.ConstraintReport(
        status="ok",
        reasons=[],
        deletion=False,
        deleted=[],
        added=["Fresh"],
        mechanism_count=1,
        candidate={},
        baseline={},
        initial={},
    )
    cx.assess_rubric_health(report, cand, baseline_stats)
    assert report.status == "needs_strong_evidence"
    assert report.new_dead_axes == ["Fresh"]
    assert report.reasons == ["new dead axis Fresh (96% at +1 over 50 rows)"]
    # Idempotent: a second measurement adds no duplicate reason.
    cx.assess_rubric_health(report, cand, baseline_stats)
    assert len(report.reasons) == 1


def test_an_inherited_dead_axis_fires_nothing(baseline_stats):
    cand = criterion_stats(
        _decisions({"Alpha": _spread(1), "Beta": _spread(2), "Dead": [0.0] * 50})
    )
    new_dead, new_pairs = cx.rubric_defects(cand, baseline_stats)
    assert new_dead == [] and new_pairs == []


def test_a_new_redundant_pair_needs_strong_evidence_and_an_inherited_one_does_not(baseline_stats):
    beta = _spread(2)
    alpha = _spread(1)
    cand = criterion_stats(
        _decisions({"Alpha": alpha, "Twin": list(alpha), "Beta": beta, "BetaCopy": list(beta)})
    )
    new_dead, new_pairs = cx.rubric_defects(cand, baseline_stats)
    assert new_dead == []
    assert new_pairs == [["Beta", "BetaCopy"]]
    report = cx.ConstraintReport(
        status="ok",
        reasons=[],
        deletion=False,
        deleted=[],
        added=["BetaCopy"],
        mechanism_count=1,
        candidate={},
        baseline={},
        initial={},
    )
    cx.assess_rubric_health(report, cand, baseline_stats)
    assert report.status == "needs_strong_evidence"
    assert report.reasons == ["new |rho| >= 0.70 pair Beta vs BetaCopy (1.00)"]


def test_a_dead_axis_over_too_few_rows_is_not_a_defect_yet(baseline_stats):
    cand = criterion_stats(_decisions({"Alpha": _spread(1, 30), "Fresh": [1.0] * 30}))
    assert cx.rubric_defects(cand, baseline_stats) == ([], [])


def test_an_invalid_report_is_not_downgraded_by_a_defect(baseline_stats):
    cand = criterion_stats(_decisions({"Alpha": _spread(1), "Fresh": [1.0] * 50}))
    report = cx.ConstraintReport(
        status="invalid",
        reasons=["x"],
        deletion=False,
        deleted=[],
        added=[],
        mechanism_count=1,
        candidate={},
        baseline={},
        initial={},
    )
    cx.assess_rubric_health(report, cand, baseline_stats)
    assert report.status == "invalid"
    assert report.new_dead_axes == ["Fresh"]


# --- frozen files -------------------------------------------------------------


def test_parse_freeze_accepts_the_three_names_and_refuses_others():
    assert cx.parse_freeze("rules,policy") == ("rules", "policy")
    assert cx.parse_freeze(" Rules , rules ") == ("rules",)
    assert cx.parse_freeze(None) == ()
    assert cx.parse_freeze("") == ()
    with pytest.raises(ValueError, match="unknown --freeze target 'tests'"):
        cx.parse_freeze("rules,tests")


def test_a_changed_frozen_file_is_invalid_and_named(tmp_path, baseline):
    cand = make_filter(
        tmp_path, "rules_edit", rubric=_rubric(), rules=RULES_CHANGED, policy=_policy()
    )
    report = _check(cand, baseline, freeze=("rules", "policy"))
    assert report.status == "invalid"
    assert report.reasons == ["frozen file changed: rules.py"]
    assert report.frozen_changed == ["rules.py"]


def test_a_rubric_only_edit_passes_the_freeze(tmp_path, baseline):
    cand = make_filter(
        tmp_path,
        "rubric_edit",
        rubric=_rubric(bodies={"Beta": "Sharper Beta."}),
        rules=RULES_TWO,
        policy=_policy(),
    )
    report = _check(cand, baseline, freeze=("rules", "policy"))
    assert report.status == "ok" and report.frozen_changed == []


def test_a_simplification_may_still_only_touch_unfrozen_files(tmp_path, baseline):
    """The freeze is checked first and wins: a rubric-only run whose proposer
    may rewrite policy.py whenever the filter shrinks is not a rubric-only
    run."""

    names = ("Alpha", "Beta", "Gamma", "Twin")
    cand = make_filter(
        tmp_path, "prune_all", rubric=_rubric(names), rules=RULES, policy=_policy(names)
    )
    report = _check(cand, baseline, freeze=("rules", "policy"))
    assert report.simplification is True
    assert report.status == "invalid"
    assert report.reasons == [
        "frozen file changed: rules.py",
        "frozen file changed: policy.py",
    ]

    # The same shrinking done only in rubric.md is fine.
    rubric_only = make_filter(
        tmp_path,
        "shorter_rubric",
        rubric=_rubric(bodies={"Beta": "B."}),
        rules=RULES_TWO,
        policy=_policy(),
    )
    report = _check(rubric_only, baseline, freeze=("rules", "policy"))
    assert report.simplification is True and report.status == "ok"


# --- through run_precheck ---------------------------------------------------------


class _ConstantScorer:
    """Every criterion scores off its name: the dead one is 0 on every row."""

    def score(self, trajectory, criteria, rubric_md=""):
        from meta_filter.solver import ScorerCost

        scores = {}
        for c in criteria:
            if c == "Dead":
                scores[c] = 0.0
            else:
                scores[c] = float(trajectory.metadata[c.lower()])
        return scores, [], ScorerCost()


def _rows(n: int = 24):
    from meta_filter.schemas import Trajectory

    rng = np.random.default_rng(7)
    rows = {}
    for i in range(n):
        tid = f"t{i:02d}"
        rows[tid] = Trajectory(
            trajectory_id=tid,
            task="t",
            metadata={
                "alpha": float(rng.integers(-2, 3)),
                "beta": float(rng.integers(-2, 3)),
                "gamma": float(rng.integers(-2, 3)),
                "twin": float(rng.integers(-2, 3)),
            },
        )
    return rows


def test_run_precheck_reports_simplification_and_frozen_files(tmp_path, baseline, baseline_stats):
    from meta_filter.filter_loader import load_filter

    names = ("Alpha", "Beta", "Gamma", "Twin")
    smaller = make_filter(
        tmp_path / "cands",
        "c_smaller",
        rubric=_rubric(names),
        rules=RULES_TWO,
        policy=_policy(names),
    )
    frozen = make_filter(
        tmp_path / "cands", "c_rules", rubric=_rubric(), rules=RULES_CHANGED, policy=_policy()
    )
    rows = _rows()
    ids = sorted(rows)
    pairs = [(ids[i], ids[i + 1]) for i in range(0, len(ids) - 1, 2)]
    results = run_precheck(
        [smaller, frozen],
        parent_filter=load_filter(baseline),
        trajectories_by_id=rows,
        truth_pairs=pairs,
        scorer=_ConstantScorer(),
        initial_dir=baseline,
        baseline_stats=baseline_stats,
        freeze=("rules",),
        cost_ledger=None,
    )
    assert results["c_smaller"].constraint.simplification is True
    assert results["c_smaller"].criterion_stats is not None
    assert results["c_rules"].constraint.status == "invalid"
    assert "frozen file changed: rules.py" in results["c_rules"].constraint.reasons
    line = _constraint_line(results["c_smaller"].constraint)
    assert line.startswith("  SIMPLIFICATION")
    assert "b >= c" in line
    payload = (smaller / "precheck.json").read_text(encoding="utf-8")
    assert '"simplification": true' in payload
    assert '"criterion_stats"' in payload
