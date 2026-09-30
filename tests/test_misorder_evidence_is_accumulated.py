"""The misordered-pair evidence is drawn from the accumulated exam.

The loop buys judged pairs every iteration and accumulates them, so a run
ends on several hundred resolved pairs. The proposer's richest evidence is
the misordered-pair block — and built from the CURRENT iteration's purchase
alone, it would name only pairs judged at the immediately preceding
iteration, never repeat one, and leave most of what the run paid for
unnameable.

These tests pin the selection rule (`optimizer._select_misordered_pairs`):
the accumulated exam restricted to the pool, a minority of slots held for the
freshest buy, the rest spent on the pairs the champion still gets wrong by the
widest margin, consensus-audit pairs last, repeats kept and labelled rather
than suppressed.

They also pin the second half: `--pair-evidence` carries the audit stratum
the blocks rank last, rather than drawing from the same mis-ordered list the
blocks render, of which it would be a strict subset.
"""

from __future__ import annotations

import json

from meta_filter.optimizer import (
    MAX_PAIR_MISORDER_RECORDS,
    _audit_blind_spot_pairs,
    _pair_misorder_examples,
    _select_misordered_pairs,
)
from meta_filter.proposer import (
    MISORDER_BLOCK_BUDGET,
    PROMPT_MISORDER_PAIRS,
    _format_pair_misorders,
)
from meta_filter.schemas import Decision, FilterDecision, Trajectory, Turn
from tests.selection_gate_fixture import (
    BASELINE_POLICY,
    build_optimizer,
    prepare,
)


# A pool of ids the judge orders `hi<N>` over `lo<N>`, which the filter below
# inverts by `gap` — one knob per pair, so a test says only what it means.
def _pair(n: int) -> tuple[str, str]:
    return (f"hi{n}", f"lo{n}")


def _scores(gaps: dict[int, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for n, gap in gaps.items():
        out[f"hi{n}"] = 0.0
        out[f"lo{n}"] = gap
    return out


def _provenance(spec: dict[int, tuple[str, int | None]]) -> dict[frozenset[str], dict]:
    return {
        frozenset(_pair(n)): {"source": source, "judged_iteration": bought}
        for n, (source, bought) in spec.items()
    }


# --- the selection rule -------------------------------------------------------------


def test_the_block_draws_from_the_accumulated_set_not_the_fresh_buy():
    """The defect itself: pairs bought three iterations ago must be eligible."""

    pairs = [_pair(n) for n in range(4)]
    picked = _select_misordered_pairs(
        scores=_scores({0: 9.0, 1: 8.0, 2: 7.0, 3: 1.0}),
        resolved_pairs=pairs,
        provenance=_provenance(
            {0: ("disputed", 0), 1: ("disputed", 1), 2: ("disputed", 2), 3: ("disputed", 5)}
        ),
        iteration=5,
        limit=PROMPT_MISORDER_PAIRS,
    )
    named = {frozenset((hi, lo)) for hi, lo, _gap, _note in picked}
    # Under the old rule only pair 3 — the fresh buy — could ever appear.
    assert named == {frozenset(_pair(n)) for n in range(4)}
    ages = {
        int(note.split("bought at iter ")[1].split(";")[0])
        for *_x, note in picked
        if "bought at iter " in note
    }
    assert ages == {0, 1, 2}


def test_the_fresh_minority_is_present_but_stays_a_minority():
    """Two of the five rendered slots go to this iteration's purchase.

    Not all five (that is the defect) and not none: the freshest pairs are the
    only ones aimed at what the last proposal actually changed.
    """

    standing = {n: ("disputed", n % 9) for n in range(20)}
    fresh = {n: ("disputed", 99) for n in range(20, 40)}
    # The fresh pairs are the SMALLEST gaps, so nothing but the reservation
    # can put them in the top five.
    gaps = {n: 100.0 - n for n in range(20)} | {n: 1.0 for n in range(20, 40)}
    picked = _select_misordered_pairs(
        scores=_scores(gaps),
        resolved_pairs=[_pair(n) for n in range(40)],
        provenance=_provenance(standing | fresh),
        iteration=99,
        limit=PROMPT_MISORDER_PAIRS,
    )
    is_fresh = ["bought this iteration" in note for *_x, note in picked]
    assert sum(is_fresh) == 2 and len(picked) == 5
    # Spread, not appended: the head of the block carries both streams.
    assert is_fresh == [False, False, True, False, True]
    # The full record cap keeps the same ratio.
    full = _select_misordered_pairs(
        scores=_scores(gaps),
        resolved_pairs=[_pair(n) for n in range(40)],
        provenance=_provenance(standing | fresh),
        iteration=99,
        limit=MAX_PAIR_MISORDER_RECORDS,
    )
    assert sum("bought this iteration" in note for *_x, note in full) == 8


def test_a_stream_that_runs_dry_gives_its_slots_to_the_other():
    """Iteration 0 has no standing failures; nothing may be left unsaid."""

    picked = _select_misordered_pairs(
        scores=_scores({0: 3.0, 1: 2.0, 2: 1.0}),
        resolved_pairs=[_pair(n) for n in range(3)],
        provenance=_provenance({n: ("disputed", 0) for n in range(3)}),
        iteration=0,
        limit=PROMPT_MISORDER_PAIRS,
    )
    assert [w for w, *_x in picked] == ["hi0", "hi1", "hi2"]


def test_audit_pairs_are_deprioritised_however_wrong_they_are():
    """Consensus-audit pairs rank candidates at tau 0.184; disputed at 0.50.

    So a hugely mis-ordered audit pair is still worse RANKING evidence than a
    narrowly mis-ordered disputed one. They are not dropped — `--pair-evidence`
    carries them — but they go to the back of this block.
    """

    picked = _select_misordered_pairs(
        scores=_scores({0: 100.0, 1: 90.0, 2: 1.0, 3: 0.5}),
        resolved_pairs=[_pair(n) for n in range(4)],
        provenance=_provenance(
            {
                0: ("audit", 2),
                1: ("audit", 2),
                2: ("disputed", 2),
                3: ("disputed", 2),
            }
        ),
        iteration=3,
        limit=PROMPT_MISORDER_PAIRS,
    )
    assert [w for w, *_x in picked] == ["hi2", "hi3", "hi0", "hi1"]
    assert "consensus-audit stratum" in picked[2][3]


def test_a_pair_still_wrong_reappears_and_says_it_has_been_shown():
    """Repeats are evidence, not noise — but the proposer must be told."""

    shown = {frozenset(_pair(0)): [2, 4]}
    picked = _select_misordered_pairs(
        scores=_scores({0: 9.0, 1: 1.0}),
        resolved_pairs=[_pair(0), _pair(1)],
        provenance=_provenance({0: ("disputed", 1), 1: ("disputed", 5)}),
        iteration=5,
        shown_before=shown,
        limit=PROMPT_MISORDER_PAIRS,
    )
    repeated = next(note for w, _l, _g, note in picked if w == "hi0")
    assert "already shown at iter 2, 4 and STILL mis-ordered" in repeated
    first_time = next(note for w, _l, _g, note in picked if w == "hi1")
    assert "already shown" not in first_time


def test_without_provenance_the_order_is_the_pure_gap_order():
    """With no provenance the order is the inversion-gap order."""

    picked = _select_misordered_pairs(
        scores=_scores({0: 1.0, 1: 5.0, 2: 3.0}),
        resolved_pairs=[_pair(n) for n in range(3)],
        limit=PROMPT_MISORDER_PAIRS,
    )
    assert [w for w, *_x in picked] == ["hi1", "hi2", "hi0"]
    assert {note for *_x, note in picked} == {""}


# --- the records and the rendered block ---------------------------------------------


def _traj(tid: str) -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task=f"task {tid}",
        turns=[Turn(role="assistant", action="edit", content="x" * 400) for _ in range(4)],
    )


def _decision(tid: str) -> FilterDecision:
    return FilterDecision(
        trajectory_id=tid,
        decision=Decision.REVIEW,
        confidence=0.5,
        rubric_scores={"Grounding": 1.0, "Verification": 2.0},
    )


def _records(*, shown, provenance, iteration, n=6):
    ids = [tid for n_ in range(n) for tid in _pair(n_)]
    return _pair_misorder_examples(
        scores=_scores({n_: 10.0 - n_ for n_ in range(n)}),
        resolved_pairs=[_pair(n_) for n_ in range(n)],
        traj_by_id={tid: _traj(tid) for tid in ids},
        decisions_by_id={tid: _decision(tid) for tid in ids},
        iteration=iteration,
        filter_version="v1",
        provenance=provenance,
        shown_before=shown,
    )


def test_the_history_line_reaches_the_prompt_and_the_saved_record():
    fns, fps = _records(
        shown={frozenset(_pair(0)): [3]},
        provenance=_provenance({n: ("disputed", 1) for n in range(6)}),
        iteration=4,
    )
    assert "already shown at iter 3" in fns[0].pair_history
    # It survives the critic_outputs/ JSON round-trip like the breakdown does.
    assert json.loads(fns[0].model_dump_json())["pair_history"] == fns[0].pair_history
    block = _format_pair_misorders(fns, fps, iteration=4)
    header = next(i for i, line in enumerate(block.splitlines()) if line.startswith("### "))
    assert block.splitlines()[header + 1].startswith("- pair history: bought at iter 1")
    assert "STILL mis-ordered" in block.splitlines()[header + 1]


def test_the_block_respects_its_character_budget():
    fns, fps = _records(
        shown={},
        provenance=_provenance({n: ("disputed", 1) for n in range(6)}),
        iteration=4,
    )
    full = _format_pair_misorders(fns, fps, iteration=4, limit=5)
    assert len(full) <= MISORDER_BLOCK_BUDGET
    one_pair = _format_pair_misorders(fns, fps, iteration=4, limit=1)
    trimmed = _format_pair_misorders(fns, fps, iteration=4, limit=5, budget=len(one_pair) + 10)
    assert len(trimmed) <= len(one_pair) + 10
    # A budget below even one pair still renders that pair: truncated evidence
    # beats none.
    assert _format_pair_misorders(fns, fps, iteration=4, limit=5, budget=1) == one_pair


# --- the audit channel (`--pair-evidence`) ------------------------------------------


def test_pair_evidence_carries_only_audit_pairs_and_only_unshown_ones():
    provenance = _provenance(
        {0: ("audit", 1), 1: ("audit", 2), 2: ("disputed", 2), 3: ("audit", 3)}
    )
    pool, missed = _audit_blind_spot_pairs(
        scores=_scores({0: 5.0, 1: 4.0, 2: 9.0, 3: -1.0}),
        pairs=[_pair(n) for n in range(4)],
        provenance=provenance,
        exclude={frozenset(_pair(0))},
    )
    # Three audit pairs applied; pair 3 is ordered correctly, pair 0 is
    # already in the misordered-pair blocks, so one line is left — and it is
    # not the most confidently wrong pair overall, which is disputed.
    assert len(pool) == 3
    assert [(hi, lo, when) for hi, lo, _gap, when in missed] == [("hi1", "lo1", 2)]


def test_the_audit_channel_is_empty_when_nothing_was_audited():
    pool, missed = _audit_blind_spot_pairs(
        scores=_scores({0: 5.0}),
        pairs=[_pair(0)],
        provenance=_provenance({0: ("disputed", 1)}),
    )
    assert pool == [] and missed == []


# --- end to end ---------------------------------------------------------------------

# Worse than the baseline on every pair, so the baseline stays champion and
# keeps inverting the same pairs iteration after iteration — which is the
# situation the accumulated exam exists for.
REVERSED_POLICY = """RUBRIC_CRITERIA = ['Noisy']


def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': -scores['Noisy'], 'confidence': 0.5}
"""


def test_a_mock_run_names_pairs_bought_before_the_previous_iteration(tmp_path):
    run = prepare(tmp_path)
    script = {k: [("reversed", REVERSED_POLICY), ("clone", BASELINE_POLICY)] for k in range(8)}
    opt = build_optimizer(
        run,
        script=script,
        iterations=8,
        run_id="accumulated_evidence",
    )
    opt.run()

    named_by_iteration: dict[int, list[frozenset[str]]] = {}
    histories: list[str] = []
    for prompt in opt.proposer.prompts:
        payload = json.loads(prompt)
        if "recent_false_negatives" not in payload:
            continue
        pairs: list[frozenset[str]] = []
        for blob in payload["recent_false_negatives"][:PROMPT_MISORDER_PAIRS]:
            partner = blob.split("failure_subtype='pair_misorder_vs:")[1].split("'")[0]
            tid = blob.split("trajectory_id='")[1].split("'")[0]
            pairs.append(frozenset((tid, partner)))
            histories.append(blob.split("pair_history='")[1].split("'")[0])
        named_by_iteration[payload["iteration"]] = pairs

    bought_at = {key: record["judged_iteration"] for key, record in opt._pair_provenance.items()}
    # The measured defect: under the old rule every named pair was judged at
    # exactly k-1. Here some prompt names a pair bought at least two
    # iterations earlier.
    older = [
        (k, bought_at[pair])
        for k, pairs in named_by_iteration.items()
        for pair in pairs
        if bought_at.get(pair) is not None and bought_at[pair] <= k - 2
    ]
    assert older, f"no prompt reached past k-1: {named_by_iteration} / {bought_at}"

    # ...and the same pair is shown again while it is still wrong, labelled.
    seen: dict[frozenset[str], int] = {}
    for pairs in named_by_iteration.values():
        for pair in pairs:
            seen[pair] = seen.get(pair, 0) + 1
    assert max(seen.values()) > 1, f"no pair was ever shown twice: {seen}"
    assert any("STILL mis-ordered" in note for note in histories)

    # The age distribution, which is the thing the defect flattened: the last
    # prompt of the run carries pairs from more than one buying iteration.
    last = max(named_by_iteration)
    ages = {bought_at[pair] for pair in named_by_iteration[last]}
    assert len(ages) > 1, f"prompt {last} carried one age only: {ages}"
