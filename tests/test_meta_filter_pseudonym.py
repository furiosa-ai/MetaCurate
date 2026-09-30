"""Trajectory ids must not reach a surface the agentic proposer can read."""

from __future__ import annotations

import json

import pytest

from meta_filter.experience_store import ExperienceStore
from meta_filter.pseudonym import Pseudonymizer
from meta_filter.schemas import CounterExample, Decision, FilterDecision, Trajectory

REAL_ID = "pandas-dev__pandas-48540::att0"
INSTANCE_ID = "pandas-dev__pandas-48540"


@pytest.fixture
def pseudo() -> Pseudonymizer:
    return Pseudonymizer.for_pool([REAL_ID], salt="fixed-salt-for-tests")


def test_pseudonym_is_stable_and_opaque(pseudo):
    first = pseudo.redact(REAL_ID)
    assert first == pseudo.redact(REAL_ID)
    assert first.startswith("traj_")
    assert "pandas" not in first


def test_bare_instance_id_maps_to_the_same_pseudonym(pseudo):
    """`metadata.instance_id` drops the `::att0` suffix but is the same key."""

    assert pseudo.redact(INSTANCE_ID) == pseudo.redact(REAL_ID)


def test_full_id_is_replaced_before_the_bare_instance(pseudo):
    """Longest-first, or the `::att0` tail would be left dangling."""

    assert "::att0" not in pseudo.redact(f"decided on {REAL_ID} today")


def test_redacts_ids_embedded_in_prose(pseudo):
    """Models name trajectories in rationales; field-level rewriting misses those."""

    text = f"The judge said {INSTANCE_ID} was fine, but {REAL_ID} regressed."
    out = pseudo.redact(text)
    assert "pandas-dev" not in out
    assert out.count(pseudo.redact(REAL_ID)) == 2


def test_salt_changes_the_pseudonym():
    """An unsalted digest could be recomputed at runtime from the real id."""

    a = Pseudonymizer.for_pool([REAL_ID], salt="one").redact(REAL_ID)
    b = Pseudonymizer.for_pool([REAL_ID], salt="two").redact(REAL_ID)
    assert a != b


def test_saved_map_reverses_the_pseudonym(tmp_path, pseudo):
    out = tmp_path / "_idmap" / "run.json"
    pseudo.save(out)
    payload = json.loads(out.read_text())
    assert payload["pseudonym_to_id"][pseudo.redact(REAL_ID)] == REAL_ID
    assert payload["salt"] == "fixed-salt-for-tests"


def _counterexample() -> CounterExample:
    return CounterExample(
        trajectory_id=REAL_ID,
        source_iteration=0,
        filter_version="abc123",
        critic_decision=Decision.ACCEPT,
        predicted_decision=Decision.REJECT,
        failure_type="false_positive",
        why_filter_failed=f"rule fired on {INSTANCE_ID}",
        raw_trajectory=Trajectory(
            trajectory_id=REAL_ID,
            task="fix the bug",
            metadata={"instance_id": INSTANCE_ID},
        ),
        solver_decision=FilterDecision(trajectory_id=REAL_ID, decision=Decision.REJECT),
    )


def test_store_writes_no_real_id_anywhere(tmp_path, pseudo):
    """The whole point: nothing the proposer can open names a trajectory.

    Covers the nested surfaces a field-level rewrite would miss —
    `raw_trajectory.trajectory_id`, `metadata.instance_id`, and the id quoted
    inside `why_filter_failed`.
    """

    store = ExperienceStore(run_dir=tmp_path / "run", pseudonymizer=pseudo)
    ce = _counterexample()
    store.save_critic_output(0, "cand", [ce], [], [])
    accepted = FilterDecision(trajectory_id=REAL_ID, decision=Decision.ACCEPT)
    store.save_solver_output(0, "cand", [accepted])
    store.save_report(0, f"# report\n\n{REAL_ID} was the worst case.\n")
    store.save_proposer_io(0, f"prompt mentioning {INSTANCE_ID}", "{}")

    for path in (tmp_path / "run").rglob("*"):
        if path.is_file():
            assert "pandas-dev" not in path.read_text(), f"real id leaked into {path}"


def test_solver_and_critic_outputs_stay_joinable(tmp_path, pseudo):
    """Mixed spellings would let the proposer rebuild the mapping by joining."""

    store = ExperienceStore(run_dir=tmp_path / "run", pseudonymizer=pseudo)
    store.save_critic_output(0, "cand", [_counterexample()], [])
    accepted = FilterDecision(trajectory_id=REAL_ID, decision=Decision.ACCEPT)
    store.save_solver_output(0, "cand", [accepted])

    it = tmp_path / "run" / "iterations" / "000"
    critic = json.loads((it / "critic_outputs" / "cand.json").read_text())
    solver = json.loads((it / "solver_outputs" / "cand.json").read_text())
    assert critic["false_negatives"][0]["trajectory_id"] == solver[0]["trajectory_id"]


def test_store_without_pseudonymizer_is_unchanged(tmp_path):
    """Static runs keep real ids — they are the debuggable default."""

    store = ExperienceStore(run_dir=tmp_path / "run")
    store.save_report(0, REAL_ID)
    assert REAL_ID in (tmp_path / "run" / "iterations" / "000" / "report.md").read_text()
