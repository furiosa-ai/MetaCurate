"""`critic_outputs/` has to hold what a second opinion said.

`BoundaryCritic` writes `critic_decision=d.decision,
predicted_decision=d.decision` — the filter's own verdict, copied into both
fields of a record filed under "critic". Written into `critic_outputs/`
beside the real disagreements, those self-copies would be most of the
directory's contents, because the judge speaks about PAIRS and produces no
per-row verdicts at all — while the proposer's workspace note points it at
that directory as "where the judge disagreed, and its stated rationale". A
proposer investigating its own history would find a pile of records
asserting that a judge had confirmed the filter, when no judge had looked.

So the judge's pair verdicts become the FN/FP evidence, boundary flags live
in `boundary_flags/` under their own name, and no record in `critic_outputs/`
can be the filter agreeing with itself.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from meta_filter.llm.base import LLMResponse
from meta_filter.optimizer import Optimizer, OptimizerConfig, _pair_misorder_examples
from meta_filter.proposer import Proposer, ProposerResult
from meta_filter.schemas import Decision, FilterDecision, Trajectory, Turn
from meta_filter.selector import Selector
from meta_filter.solver import HeuristicRubricScorer, Solver
from meta_filter.validator import Validator
from tests.demo_pool import selection_rows

ROOT = Path(__file__).resolve().parents[1]
INITIAL_RANKING = ROOT / "filters" / "initial_ranking_nometa"


# --- the builder, on its own ------------------------------------------------


def _traj(tid: str) -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task=f"task {tid}",
        turns=[Turn(role="agent", action="edit", content="patch")],
    )


def test_a_misordered_pair_becomes_two_records_pointing_opposite_ways():
    """One inversion is two mistakes, and the proposer needs both halves."""

    traj_by_id = {tid: _traj(tid) for tid in ("hi", "lo")}
    decisions = {
        tid: FilterDecision(trajectory_id=tid, decision=Decision.ACCEPT, confidence=0.4)
        for tid in traj_by_id
    }
    # The judge ranked "hi" above "lo"; the filter scored them the other way.
    fn, fp = _pair_misorder_examples(
        scores={"hi": 1.0, "lo": 9.0},
        resolved_pairs=[("hi", "lo")],
        traj_by_id=traj_by_id,
        decisions_by_id=decisions,
        iteration=3,
        filter_version="v1",
        rationale_for=lambda a, b: "it ran the tests",
    )

    assert [c.trajectory_id for c in fn] == ["lo"], "the over-ranked row"
    assert [c.trajectory_id for c in fp] == ["hi"], "the under-ranked row"
    for record in fn + fp:
        assert record.critic_decision != record.predicted_decision
        assert record.failure_subtype.startswith("pair_misorder_vs:")
        assert "it ran the tests" in record.description


def test_a_pair_the_filter_orders_correctly_produces_nothing():
    traj_by_id = {tid: _traj(tid) for tid in ("hi", "lo")}
    fn, fp = _pair_misorder_examples(
        scores={"hi": 9.0, "lo": 1.0},
        resolved_pairs=[("hi", "lo")],
        traj_by_id=traj_by_id,
        decisions_by_id={},
        iteration=0,
        filter_version="v1",
    )

    assert (fn, fp) == ([], [])


# --- and end to end, over a real run tree -----------------------------------


class _PairClient:
    """A judge with a consistent, order-independent preference."""

    model = "stub"

    def generate(self, prompt, **_kw):
        first = prompt.split("=== TRAJECTORY A (", 1)[1].split(")", 1)[0]
        letter = "A" if "good" in first else "B"
        return LLMResponse(
            text=f'{{"winner": "{letter}", "rationale": "it verified the fix"}}',
            prompt_tokens=0,
            completion_tokens=0,
        )


class _Backend:
    """A pair-judging backend that rules on no individual trajectory."""

    client = _PairClient()

    def critic_verdict(self, trajectory):
        return None


class _InvertingProposer(Proposer):
    """Emits one candidate that ranks the pool backwards.

    A candidate that disagrees with the baseline is what creates disputed
    pairs, and disputed pairs are what the judge is asked about — without one
    the truth channel is empty and there is no evidence to be truthful about.
    """

    def __init__(self) -> None:
        self.client = _PairClient()

    def propose(self, *, candidates_root, iteration, **_kw):
        cand = Path(candidates_root) / f"iter{iteration:03d}_00_inverted"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(INITIAL_RANKING, cand, ignore=shutil.ignore_patterns("__pycache__"))
        policy = cand / "policy.py"
        source = policy.read_text(encoding="utf-8")
        assert "score = BASE_OFFSET + mean * BASE_SPAN" in source
        policy.write_text(
            source.replace(
                "score = BASE_OFFSET + mean * BASE_SPAN", "score = BASE_OFFSET - mean * BASE_SPAN"
            ),
            encoding="utf-8",
        )
        return ProposerResult(
            prompt="",
            response=LLMResponse(text="{}", prompt_tokens=0, completion_tokens=0),
            candidate_dirs=[cand],
        )


class _BrokenJudgeBackend:
    """A judge that answers only on the hidden selection rows, so no train
    pair is ever resolved."""

    class _Client(_PairClient):
        def generate(self, prompt, **_kw):
            if "=== TRAJECTORY A (sel_" not in prompt:
                raise RuntimeError("judge is down")
            return super().generate(prompt, **_kw)

    client = _Client()

    def critic_verdict(self, trajectory):
        return None


def _ranking_run(tmp_path: Path, *, backend=None, proposer=None, iterations=1) -> Path:
    optimizer = Optimizer(
        initial_filter_dir=INITIAL_RANKING,
        train_data=ROOT / "data" / "demo_train.jsonl",
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=proposer or _InvertingProposer(),
        selector=Selector(),
        solver=Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=Validator(loss_weights={}, critic_backend=backend or _Backend()),
        config=OptimizerConfig(
            iterations=iterations,
            candidates_per_iteration=1,
            pair_budget=8,
            max_pairs=66,
            pair_judge_cache_dir=tmp_path / "pair_cache",
            selection_data=selection_rows(tmp_path),
        ),
        run_id="evidence-test",
    )
    return optimizer.run()


def _critic_records(run_dir: Path) -> list[dict]:
    out: list[dict] = []
    for path in (run_dir / "iterations" / "000" / "critic_outputs").glob("*.json"):
        for group in json.loads(path.read_text(encoding="utf-8")).values():
            out.extend(group)
    return out


def test_no_record_in_critic_outputs_is_the_filter_agreeing_with_itself(tmp_path):
    run_dir = _ranking_run(tmp_path)
    records = _critic_records(run_dir)

    assert records, "the run produced no judge evidence at all"
    for record in records:
        assert record["critic_decision"] != record["predicted_decision"], record["trajectory_id"]
        assert record["failure_type"] != "boundary"


def test_the_evidence_names_the_pair_the_judge_ordered(tmp_path):
    run_dir = _ranking_run(tmp_path)

    for record in _critic_records(run_dir):
        assert record["failure_subtype"].startswith("pair_misorder_vs:")


def test_boundary_flags_are_kept_but_filed_under_their_own_name(tmp_path):
    """Not deleted — they are real diagnostics and memory is built from them."""

    run_dir = _ranking_run(tmp_path)
    it = run_dir / "iterations" / "000"

    flags = list((it / "boundary_flags").glob("*.json"))
    assert flags, "boundary flags must still be written somewhere"
    payload = json.loads(flags[0].read_text(encoding="utf-8"))
    assert payload["source"] == "boundary_heuristic"
    assert "no second opinion was consulted" in payload["note"]

    for path in (it / "critic_outputs").glob("*.json"):
        assert "boundary" not in json.loads(path.read_text(encoding="utf-8"))


# --- an iteration with no verdicts says so, rather than substituting one -----


def test_an_unresolved_iteration_writes_no_judge_evidence(tmp_path):
    """The fallback path is the same risk one channel over.

    With the judge down, nothing is resolved. Falling through to the
    validator's records — which are derived from CANDIDATE CONSENSUS, the
    committee's own opinion — would file them under `critic_outputs/` and
    the FN/FP prompt headings, where the proposer reads them as a second
    opinion. An empty directory is the honest artifact.
    """

    run_dir = _ranking_run(tmp_path, backend=_BrokenJudgeBackend())
    it = run_dir / "iterations" / "000"

    for path in (it / "critic_outputs").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == {
            "false_negatives": [],
            "false_positives": [],
            "pair_misorders": [],
        }, path.name

    # Not vacuous: the run did produce self-graded records, they are simply
    # filed where they belong instead of being promoted to judge evidence.
    flags = [
        json.loads(p.read_text(encoding="utf-8")) for p in (it / "boundary_flags").glob("*.json")
    ]
    assert any(f["boundary"] for f in flags)


# --- the telemetry names the floor and the penalty it actually applies -------


class _CapturingProposer(_InvertingProposer):
    """Records the telemetry block each proposer call was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.ranking_blocks: list[str] = []

    def propose(self, **kwargs):
        self.ranking_blocks.append((kwargs.get("loop_telemetry") or {}).get("ranking_block") or "")
        return super().propose(**kwargs)


def test_the_proposer_is_told_the_floor_and_what_crossing_it_costs(tmp_path):
    """ "Zeroed" was half the story, and the half that matters was missing.

    A degenerate candidate has its concordance forced to 0 AND a further 1.0
    taken off its score, which puts it below every non-degenerate rival no
    matter how the secondary terms fall. Told only that it is "zeroed", a
    proposer can reasonably read the guard as a cap worth trading against —
    and it was never told the number it had to clear at all.
    """

    proposer = _CapturingProposer()
    _ranking_run(tmp_path, proposer=proposer, iterations=2)

    block = proposer.ranking_blocks[-1]
    assert "degeneracy floor 3" in block, block
    assert "concordance is forced to 0.000" in block
    assert "1.000 is subtracted from its score" in block


# --- and the prompt renders those records as pairs, not as verdicts ----------


def test_the_pair_records_reach_the_proposer_as_one_pair_block(tmp_path):
    """The records were right; the prompt then split them back into halves.

    `_pair_misorder_examples` writes both directions of an inversion so the
    proposer can see which row was pushed up and which was pushed down. The
    prompt rendered those two records under "Recent false negatives" and
    "Recent false positives" — ten headings for five pairs, each row's partner
    reachable only through a `subtype=pair_misorder_vs:` string — and a
    proposer reading it reported that no adjudicated mistakes existed.
    """

    from meta_filter.filter_loader import load_filter, materialize_candidate
    from meta_filter.proposer import Proposer
    from meta_filter.schemas import FilterMetrics

    traj_by_id = {tid: _traj(tid) for tid in ("hi", "lo")}
    decisions = {
        tid: FilterDecision(trajectory_id=tid, decision=Decision.ACCEPT, confidence=0.4)
        for tid in traj_by_id
    }
    fn, fp = _pair_misorder_examples(
        scores={"hi": 1.0, "lo": 9.0},
        resolved_pairs=[("hi", "lo")],
        traj_by_id=traj_by_id,
        decisions_by_id=decisions,
        iteration=3,
        filter_version="v1",
        rationale_for=lambda a, b: "it ran the tests",
    )

    filter_dir = tmp_path / "f"
    materialize_candidate(
        filter_dir,
        rubric_md="# r\n",
        rules_py="def rule_x(t):\n    return None\n",
        policy_py="def decide(t, s, r):\n    return {'decision': 'REVIEW', 'score': 1.0}\n",
    )
    proposer = Proposer(n_candidates=1)
    prompt = proposer._build_prompt(
        loaded_filter=load_filter(filter_dir),
        iteration=3,
        current_metrics=FilterMetrics(
            total=2,
            critic_accept_total=0,
            critic_reject_total=0,
            accepted=0,
            rejected=0,
            reviewed=2,
            agree_accept=0,
            agree_reject=0,
            filter_accept_critic_reject=0,
            filter_reject_critic_accept=0,
            critic_reject_recall=0.0,
            critic_accept_retention=0.0,
            disputed_accept_rate=0.0,
            disputed_reject_rate=0.0,
            review_rate=1.0,
        ),
        recent_false_negatives=fn,
        recent_false_positives=fp,
    )

    pairs = prompt.split("### Misordered pairs", 1)[1]
    assert "the judge preferred hi over lo" in pairs
    assert "this filter scored lo above hi" in pairs
    assert 'the judge\'s rationale: "it ran the tests"' in pairs
    # One block for the inversion, not one per row.
    assert pairs.count("### Misordered pair") == 1
    assert "false negative" not in prompt.lower()
