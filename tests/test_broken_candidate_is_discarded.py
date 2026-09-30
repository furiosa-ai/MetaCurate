"""A candidate whose own code raises must not end the run.

Candidates are code an LLM wrote seconds earlier, and some of them are simply
wrong. A policy helper that raises (outside the rule loop, where the solver
already turns a raising rule into a REVIEW verdict) must cost the candidate,
not the run: it is logged, discarded, and its
siblings and the baseline carry on.
"""

from __future__ import annotations

import json

from tests.selection_gate_fixture import build_optimizer, prepare

BROKEN_POLICY = """
RUBRIC_CRITERIA = ["Noisy"]


def _substance(value):
    analysis = {"n_files": value}
    # The key does not exist: every row raises here.
    return analysis["changed"] / 40.0


def decide(trajectory, scores, triggered):
    return {"decision": "ACCEPT", "score": _substance(scores["Noisy"]), "confidence": 0.5}
"""


def test_a_raising_candidate_does_not_kill_the_run(tmp_path, caplog):
    run = prepare(tmp_path)
    script = {0: [("broken", BROKEN_POLICY)], 1: [("broken", BROKEN_POLICY)]}
    optimizer = build_optimizer(run, script=script, iterations=2, run_id="broken")

    run_dir = optimizer.run()

    assert optimizer.proposer.script and len(list(run_dir.glob("iterations/*"))) == 2
    assert any("candidate code raised" in record.getMessage() for record in caplog.records), (
        "the discard has to be logged loudly, not swallowed"
    )
    for k in range(2):
        selection = json.loads(
            (run_dir / "iterations" / f"{k:03d}" / "selection.json").read_text("utf-8")
        )
        assert selection["selected_candidate"] == f"iter{k:03d}_baseline"
