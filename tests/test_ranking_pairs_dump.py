"""The judged pair set has to be readable from inside the run directory.

The loop is supervised entirely by pair verdicts, and without this file none
of them would be legible where the proposer works. `ranking.json` says how
many pairs there were. `critic_outputs/<candidate>.json` holds an
evidence-rich slice capped at `MAX_PAIR_MISORDER_RECORDS`. The verdicts
themselves live in the pair-judge cache, outside the workspace the agentic
proposer is allowed to read. So a proposer told to fix an ordering would see
neither the exam nor its own answers to it, only a count and a sample.

`iterations/<NNN>/ranking_pairs.jsonl` is the whole exam. The prompt's pair
budget and the `critic_outputs/` cap are unchanged — this is disk, for a
proposer that goes looking.
"""

from __future__ import annotations

import json
from pathlib import Path

from meta_filter.optimizer import MAX_PAIR_MISORDER_RECORDS
from meta_filter.pair_dump import ranking_pair_rows, write_ranking_pairs
from meta_filter.pseudonym import Pseudonymizer
from meta_filter.schemas import Trajectory, Turn
from tests.test_critic_evidence_is_judge_derived import _ranking_run


def _rows(run_dir: Path, iteration: int = 0) -> list[dict]:
    path = run_dir / "iterations" / f"{iteration:03d}" / "ranking_pairs.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _ranking_json(run_dir: Path, iteration: int = 0) -> dict:
    path = run_dir / "iterations" / f"{iteration:03d}" / "ranking.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _misorders_from_critic_outputs(run_dir: Path, iteration: int = 0) -> dict[str, set]:
    """Per candidate, the pairs `critic_outputs/` says it got backwards.

    Each false-negative record is the over-ranked row, and its
    `pair_misorder_vs:` subtype names the row the judge preferred — so the
    pair is (judge's pick, over-ranked row).
    """

    out: dict[str, set] = {}
    directory = run_dir / "iterations" / f"{iteration:03d}" / "critic_outputs"
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[path.stem] = {
            (record["failure_subtype"].split(":", 1)[1], record["trajectory_id"])
            for record in payload["false_negatives"]
        }
    return out


# --- over a real ranking run -------------------------------------------------


def test_the_dump_holds_every_judged_pair_and_every_candidate(tmp_path):
    run_dir = _ranking_run(tmp_path)

    rows = _rows(run_dir)
    ranking = _ranking_json(run_dir)
    scored_candidates = set(ranking["candidates"])

    assert scored_candidates, "the run scored no candidate at all"
    # The accumulated truth set, plus this iteration's TIEs — which carry a
    # verdict, cost two judge calls, and appear nowhere else.
    assert len(rows) == ranking["eval_set"]["accumulated_pairs"] + ranking["tied_pairs"]
    assert sum(1 for row in rows if row["tie"]) == ranking["tied_pairs"]

    for row in rows:
        assert row["hi"] != row["lo"]
        assert row["source"] in ("disputed", "audit", "inherited")
        assert row["judged_iteration"] == 0
        assert set(row["by_candidate"]) == scored_candidates, row
        assert row["orders"], "a judged pair with no cached presentation"
        for presentation in row["orders"]:
            assert presentation["shown_as_a"] in (row["hi"], row["lo"])
            assert presentation["decided"] is True


def test_the_dump_agrees_with_what_critic_outputs_recorded(tmp_path):
    """Same pairs, same verdicts — one capped and rich, one complete and flat."""

    run_dir = _ranking_run(tmp_path)
    rows = _rows(run_dir)
    from_critic = _misorders_from_critic_outputs(run_dir)

    assert from_critic, "no critic_outputs to compare against"
    for candidate, expected in from_critic.items():
        # Only meaningful below the cap; above it critic_outputs holds a
        # subset by construction.
        assert len(expected) < MAX_PAIR_MISORDER_RECORDS
        from_dump = {
            (row["hi"], row["lo"])
            for row in rows
            if not row["tie"]
            and (entry := row["by_candidate"].get(candidate))
            and entry["gap"] is not None
            and entry["gap"] < 0
        }
        assert from_dump == expected, candidate


def test_a_tie_row_keeps_the_judges_own_words_and_asserts_no_order(tmp_path):
    """A TIE is the only record of where the judge contradicts itself."""

    run_dir = _ranking_run(tmp_path)
    ties = [row for row in _rows(run_dir) if row["tie"]]

    assert ties, "this fixture is supposed to produce TIE pairs"
    for row in ties:
        # Both presentations answered, and they named different winners.
        assert len({p["winner"] for p in row["orders"]}) == 2
        assert row["rationale_ab"] and row["rationale_ba"]
        # No judge order, so no candidate can be right or wrong about it.
        for entry in row["by_candidate"].values():
            assert entry["ordered_correctly"] is None


def test_the_prompt_pair_cap_is_untouched(tmp_path):
    """The dump is disk. The prompt still shows a handful of pairs.

    The two channels are deliberately different sizes: the prompt pays tokens
    for full transcript evidence on a few pairs, the file pays nothing for one
    flat line on all of them.
    """

    from meta_filter.proposer import _format_pair_misorders
    from tests.proposer_prompt_fixture import a_pair

    run_dir = _ranking_run(tmp_path)
    assert len(_rows(run_dir)) > 5

    over, under = [], []
    for i in range(12):
        high, low = a_pair(over=f"over_{i}", under=f"under_{i}", gap=1.0 + i, judge="why")
        over.append(high)
        under.append(low)
    rendered = _format_pair_misorders(over, under, iteration=1, limit=5)

    assert rendered.count("### Misordered pair ") == 5


def test_the_file_is_written_once_before_scoring_and_once_after(tmp_path, monkeypatch):
    """The first write is what an iteration that dies in the solver leaves behind.

    Candidate scores exist before the judge is called, so filling
    `by_candidate` in one pass would be possible — and would mean that a run
    killed between the judging round and the end of scoring records nothing
    about the verdicts it just paid for.
    """

    import meta_filter.pair_dump as pair_dump

    writes: list[tuple[int, bool]] = []
    original = pair_dump.write_ranking_pairs

    def spy(path, rows, *, pseudonymizer=None):
        writes.append((len(rows), any("by_candidate" in row for row in rows)))
        return original(path, rows, pseudonymizer=pseudonymizer)

    monkeypatch.setattr(pair_dump, "write_ranking_pairs", spy)
    _ranking_run(tmp_path)

    assert len(writes) == 2
    (first_rows, first_scored), (second_rows, second_scored) = writes
    assert first_rows == second_rows > 0
    assert first_scored is False
    assert second_scored is True


# --- the row builder, on its own ---------------------------------------------


def _traj(tid: str) -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task=f"task {tid}",
        turns=[Turn(role="agent", action="edit", content="patch")],
    )


def test_rows_carry_no_candidate_column_before_the_candidates_are_scored():
    """The first write lands before the solver pass, so an iteration that dies
    in scoring still leaves a legible record of the verdicts it paid for."""

    rows = ranking_pair_rows(
        accumulated_pairs=[("hi", "lo")],
        tied_pairs=[],
        provenance={frozenset({"hi", "lo"}): {"source": "audit", "judged_iteration": 2}},
        traj_by_id={"hi": _traj("hi"), "lo": _traj("lo")},
    )

    assert "by_candidate" not in rows[0]
    assert rows[0]["source"] == "audit"
    assert rows[0]["judged_iteration"] == 2


def test_an_unscored_row_costs_its_pair_here_the_way_it_does_in_scoring():
    """`concordance_detail(unscored="loss")` charges the candidate, not its rivals."""

    rows = ranking_pair_rows(
        accumulated_pairs=[("hi", "lo")],
        tied_pairs=[],
        provenance={},
        traj_by_id={"hi": _traj("hi"), "lo": _traj("lo")},
        scores_by_candidate={
            "right": {"hi": 9.0, "lo": 1.0},
            "wrong": {"hi": 1.0, "lo": 9.0},
            "flat": {"hi": 5.0, "lo": 5.0},
            "silent": {"hi": 5.0},
        },
    )
    by_candidate = rows[0]["by_candidate"]

    assert by_candidate["right"] == {"ordered_correctly": True, "gap": 8.0}
    assert by_candidate["wrong"] == {"ordered_correctly": False, "gap": -8.0}
    # Equal scores are half credit in concordance — neither ordered nor
    # misordered, so neither True nor False.
    assert by_candidate["flat"] == {"ordered_correctly": None, "gap": 0.0}
    assert by_candidate["silent"] == {"ordered_correctly": False, "gap": None}


def test_an_out_of_pool_pair_is_kept_and_marked_unscoreable():
    rows = ranking_pair_rows(
        accumulated_pairs=[("hi", "gone")],
        tied_pairs=[],
        provenance={frozenset({"hi", "gone"}): {"source": "inherited", "judged_iteration": None}},
        traj_by_id={"hi": _traj("hi")},
        scores_by_candidate={"c": {"hi": 1.0}},
    )

    assert rows[0]["in_pool"] is False
    assert rows[0]["by_candidate"] == {}
    assert rows[0]["source"] == "inherited"
    assert rows[0]["judged_iteration"] is None


def test_ids_are_redacted_everywhere_including_the_judges_prose(tmp_path):
    """The rationale is free prose and has named the trajectories it compares."""

    class _Judge:
        @staticmethod
        def presentations_for(id_a, id_b, label_view=""):
            return [
                {
                    "order": "ab",
                    "shown_as_a": id_a,
                    "shown_as_b": id_b,
                    "letter": "A",
                    "winner": id_a,
                    "decided": True,
                    "rationale": f"{id_a} verified its fix, {id_b} did not",
                }
            ]

    pseudonymizer = Pseudonymizer.for_pool(["hi", "lo"], salt="fixed")
    rows = ranking_pair_rows(
        accumulated_pairs=[("hi", "lo")],
        tied_pairs=[],
        provenance={},
        traj_by_id={"hi": _traj("hi"), "lo": _traj("lo")},
        judge=_Judge(),
    )
    path = tmp_path / "ranking_pairs.jsonl"
    write_ranking_pairs(path, rows, pseudonymizer=pseudonymizer)
    text = path.read_text(encoding="utf-8")

    assert '"hi"' not in text and '"lo"' not in text
    assert "verified its fix" in text
    assert pseudonymizer.mapping["hi"] in text
    # And the un-redacted write keeps the real ids, so the redaction above is
    # the pseudonymizer's doing rather than the writer dropping the field.
    plain = tmp_path / "plain.jsonl"
    write_ranking_pairs(plain, rows)
    assert '"hi": "hi"' in plain.read_text(encoding="utf-8")
