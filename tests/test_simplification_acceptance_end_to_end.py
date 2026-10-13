"""A no-loss simplification is accepted through the whole loop.

Built on `tests.selection_gate_fixture` with one more criterion, `Dead`, that
the scorer reads off a task-text number no row carries — so it is 0 on every
row and removing it changes no ordering anywhere. Treated as a no-op, such a
candidate would be held at precheck and, had it been scored, refused by the
gate at `b = c = 0`, and dead axes would survive whole runs. It is a
simplification: scored, accepted on `b >= c`, and recorded as
`gate_decision = accepted_simplification`.

`b = c = 0` is also confirmed without a second draw — the
ordering is identical on every pair, so there is nothing to re-draw. The case
where the re-draw is still taken and can still refuse a simplification is
`b > c`; see `tests/test_identical_ordering_simplification.py` and
`tests/test_selector_simplification.py`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from meta_filter.core.solver import ScorerCost, Solver
from meta_filter.loop.optimizer import NOOP_REASON
from tests.selection_gate_fixture import RUBRIC, RULES, build_optimizer, prepare, task_number
from tests.test_candidate_gating import INVERTED_NOISY_POLICY, RichProposer

DEAD_RUBRIC = RUBRIC + "\n## Criterion: Dead\nReads a field this pool does not carry.\n"

# The baseline scores Noisy and Dead; Dead is a constant so the ordering is Noisy's.
BASELINE_WITH_DEAD = """RUBRIC_CRITERIA = ['Noisy', 'Dead']


def decide(trajectory, scores, triggered):
    return {
        'decision': 'REVIEW',
        'score': scores['Noisy'] + scores['Dead'],
        'confidence': 0.5,
    }
"""

# The simplification: Dead gone from the rubric and from RUBRIC_CRITERIA, its
# term gone from the sum. Every component falls or holds, the total falls.
SMALLER_POLICY = """RUBRIC_CRITERIA = ['Noisy']


def decide(trajectory, scores, triggered):
    return {
        'decision': 'REVIEW',
        'score': scores['Noisy'],
        'confidence': 0.5,
    }
"""

# Both criteria kept, every score doubled: the same ordering on every pair, so
# a no-op — and no smaller than the baseline, so it gets no exemption.
SCALED_POLICY = BASELINE_WITH_DEAD.replace(
    "scores['Noisy'] + scores['Dead']", "2 * scores['Noisy'] + 2 * scores['Dead']"
)


class DeadAwareScorer:
    """Scores every criterion off the row's task text; `dead=` is absent from
    every row unless a test writes it, so Dead is 0 everywhere by default."""

    FIELDS = {"Truth": "truth", "Noisy": "noisy", "TrainBoost": "boost", "Dead": "dead"}

    def score(self, trajectory, criteria, rubric_md=""):
        scores = {c: task_number(trajectory.task, self.FIELDS[c]) for c in criteria}
        return scores, [], ScorerCost()


def _build(run, script, *, run_id: str, iterations: int = 2):
    optimizer = build_optimizer(run, script=script, iterations=iterations, run_id=run_id)
    optimizer.proposer = RichProposer(script, run.judge)
    optimizer.config.candidates_per_iteration = max(len(step) for step in script.values())
    optimizer.solver = Solver(rubric_scorer=DeadAwareScorer())
    return optimizer


def _prepare_with_dead(tmp_path: Path):
    run = prepare(tmp_path)
    (run.initial / "rubric.md").write_text(DEAD_RUBRIC, encoding="utf-8")
    (run.initial / "policy.py").write_text(BASELINE_WITH_DEAD, encoding="utf-8")
    return run


def _summary(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(run_dir: Path, k: int, name: str):
    return json.loads((run_dir / "iterations" / f"{k:03d}" / name).read_text("utf-8"))


def _diag(run_dir: Path, k: int, candidate: str) -> dict:
    """One candidate's per-candidate block in `ranking.json`, wherever the
    file keeps them (the top level also carries run-wide fields)."""

    ranking = _json(run_dir, k, "ranking.json")
    for value in ranking.values():
        if isinstance(value, dict) and candidate in value and isinstance(value[candidate], dict):
            return value[candidate]
    raise KeyError(candidate)


@pytest.fixture(scope="module")
def simplification_run(tmp_path_factory) -> Path:
    run = _prepare_with_dead(tmp_path_factory.mktemp("simplification"))
    script = {
        # Iteration 0: an inverted candidate disputes every pair (so iteration 1
        # has judged pairs and a precheck) and loses; the baseline's full pass
        # measures Dead at 100 % of rows on one level.
        0: [("inverted", INVERTED_NOISY_POLICY, DEAD_RUBRIC, RULES)],
        1: [
            ("drop_dead", SMALLER_POLICY, RUBRIC, RULES),
            ("scaled", SCALED_POLICY, DEAD_RUBRIC, RULES),
        ],
    }
    return _build(run, script, run_id="simplification").run()


def test_the_health_block_called_the_axis_dead_before_the_removal(simplification_run):
    baseline = _diag(simplification_run, 0, "iter000_baseline")
    assert baseline["criterion_stats"]["per_criterion"]["Dead"]["dead"] is True
    assert baseline["criterion_stats"]["per_criterion"]["Dead"]["modal_share"] == 1.0


def test_the_simplification_is_scored_not_held_as_a_no_op(simplification_run):
    precheck = _json(simplification_run, 1, "precheck.json")
    assert "iter001_00_drop_dead" in precheck["scored"]
    report = precheck["constraints"]["iter001_00_drop_dead"]
    assert report["simplification"] is True
    assert report["deleted"] == ["Dead"]
    # The scaled sibling orders every judged pair as the baseline does and is
    # no smaller, so it is the no-op the collapse holds — exactly the fate the
    # simplification escapes.
    scaled = precheck["constraints"]["iter001_01_scaled"]
    assert scaled["simplification"] is False and scaled["deletion"] is False
    assert precheck["gated_out"]["iter001_01_scaled"]["reason"] == NOOP_REASON


def test_the_simplification_is_accepted_at_b_equals_c_equals_zero_and_confirmed(
    simplification_run,
):
    selection = _json(simplification_run, 1, "selection.json")
    assert selection["selected_candidate"] == "iter001_00_drop_dead"
    gate = selection["gate"]
    assert gate["decision"] == "accepted_simplification"
    assert (gate["b"], gate["c"]) == (0, 0)
    assert gate["confirmation"]["confirmed"] is True
    # Identical ordering on every pair, so no second draw was taken.
    assert gate["confirmation"]["b2"] is None
    assert gate["confirmation"]["reason"].startswith("ordering identical on all 60")
    record = selection["per_candidate"]["iter001_00_drop_dead"]
    assert record["simplification"] is True and record["eligible"] is True
    assert list(selection["per_candidate"]) == ["iter001_00_drop_dead"]


def test_the_summary_and_the_next_filter_record_the_simplification(simplification_run):
    rows = _summary(simplification_run)
    assert [r["gate_decision"] for r in rows] == ["kept", "accepted_simplification"]
    assert rows[1]["gate_confirmed"] == "True"
    assert rows[1]["constraint_status"] == "ok"
    diag = _diag(simplification_run, 1, "iter001_00_drop_dead")
    assert diag["simplification"] is True
    assert diag["rubric_defects"] == {"new_dead_axes": [], "new_redundant_pairs": []}


def test_a_simplification_that_loses_a_hidden_pair_is_not_accepted(tmp_path):
    """The `b >= c` refusal itself: the hidden rows carry `dead`, so removing
    the axis reorders them even though it is dead on every train row."""

    run = _prepare_with_dead(tmp_path)
    # Give the SELECTION rows (only) a `dead` value, so the axis is dead on
    # train — which is all the loop can see — and live on the hidden pairs.
    lines = run.selection.read_text(encoding="utf-8").splitlines()
    out = []
    for i, line in enumerate(lines):
        row = json.loads(line)
        row["task"] += f" dead={40.0 if i % 2 else 0.0}"
        out.append(json.dumps(row))
    run.selection.write_text("\n".join(out) + "\n", encoding="utf-8")
    script = {
        0: [("inverted", INVERTED_NOISY_POLICY, DEAD_RUBRIC, RULES)],
        1: [("drop_dead", SMALLER_POLICY, RUBRIC, RULES)],
    }
    run_dir = _build(run, script, run_id="simplification_loses").run()
    precheck = _json(run_dir, 1, "precheck.json")
    assert precheck["constraints"]["iter001_00_drop_dead"]["simplification"] is True
    selection = _json(run_dir, 1, "selection.json")
    record = selection["per_candidate"]["iter001_00_drop_dead"]
    assert record["simplification"] is True
    if record["b"] < record["c"]:
        assert record["eligible"] is False
        assert record["blockers"] == [
            f"simplification lost pairs: b={record['b']} < c={record['c']}"
        ]
        assert selection["gate"]["decision"] == "kept"
    else:
        assert selection["gate"]["decision"] == "accepted_simplification"
