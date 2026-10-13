"""The proposer's criterion-scoring tool (`meta_filter.score_tool`).

The proposer only builds what it can verify offline, and the rubric needs the
LLM scorer, so without a tool a proposer rarely edits it. The tool lets the
agent score a drafted criterion on a few TRAIN rows under the run's own
scorer. What has to hold, and is pinned here:

* the cells it buys are the cells the full pass reads — same prompt, same
  cache key — so a criterion scored here is free at precheck;
* it answers to the names the proposer actually has. Everything the agent
  reads names trajectories by pseudonym, so `--rows traj_...` has to work
  and the report has to come back in the same spelling — otherwise every
  targeted call is refused, and the agent reads the refusal as a policy
  boundary and gives up;
* a row outside the train pool is refused, whatever it is, so the hidden
  selection rows and the probe are unreachable through it;
* every call is charged to a per-leg dollar ledger the optimizer resets at
  each proposer leg and folds into `cost_proposer`; a call past the cap is
  refused, and every output ends with the budget line;
* each harness gets exactly the sandbox hole the tool needs and no other.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from meta_filter.core.filter_loader import materialize_candidate
from meta_filter.core.llm_scorer import LLMRubricScorer
from meta_filter.llm.agent_guard import ENV_ALLOW, bash_violations, build_guard_settings
from meta_filter.llm.base import LLMResponse
from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient
from meta_filter.llm.codex_cli import CodexCliClient
from meta_filter.llm.mock import MockLLMClient
from meta_filter.score_tool import (
    LEDGER_FILENAME,
    STAGING_DIRNAME,
    ScoreBudgetLedger,
    ScoreRequest,
    ScoreToolAccess,
    access_for,
    main,
    merge_staged_cells,
    score_criterion,
    staging_dir_for,
)
from meta_filter.visibility.pseudonym import Pseudonymizer
from meta_filter.visibility.trajectory import load_trajectories
from tests.selection_gate_fixture import (
    RULES,
    TRUTH_POLICY,
    ScriptedProposer,
    build_optimizer,
    prepare,
    write_pool,
)

RUBRIC = """# Rubric

Scale +2 / -2. Doubt is 0.

## Criterion: Truth
+2 when the row is what the judge would rank first.

## Criterion: Noisy
+2 when the row looks good at a glance.
"""

# The candidate adds one criterion and declares it, as the prompt tells the
# agent to do before scoring.
CANDIDATE_RUBRIC = (
    RUBRIC
    + """
## Criterion: Verification
+2 when the agent ran the tests it changed; -2 when it never ran anything.
"""
)
CANDIDATE_POLICY = """RUBRIC_CRITERIA = ['Truth', 'Noisy', 'Verification']


def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': scores['Truth'] + scores['Verification']}
"""
BASELINE_POLICY = """RUBRIC_CRITERIA = ['Truth', 'Noisy']


def decide(trajectory, scores, triggered):
    return {'decision': 'REVIEW', 'score': scores['Truth']}
"""


@dataclass
class ToolRun:
    run_dir: Path
    cache_dir: Path
    train_ids: list[str]
    selection_ids: list[str]
    candidate: Path
    baseline: Path


def _write_run(tmp_path: Path, *, budget: float = 1.5, provider: str = "mock") -> ToolRun:
    """A run directory shaped like the optimizer leaves it, with no optimizer."""

    root = tmp_path / "repo"
    run_dir = root / "runs" / "run_x"
    cache_dir = root / "data" / "scorer_cache"
    (root / "data").mkdir(parents=True)
    train = root / "data" / "train.jsonl"
    selection = root / "data" / "selection.jsonl"
    train_ids = write_pool(train, "train", 6, train=True)
    selection_ids = write_pool(selection, "sel", 4, train=False)
    candidates = run_dir / "iterations" / "001" / "candidates"
    baseline = candidates / "iter001_baseline"
    materialize_candidate(baseline, rubric_md=RUBRIC, rules_py=RULES, policy_py=BASELINE_POLICY)
    candidate = candidates / "iter001_00_verification"
    materialize_candidate(
        candidate, rubric_md=CANDIDATE_RUBRIC, rules_py=RULES, policy_py=CANDIDATE_POLICY
    )
    # Two judged pairs among train rows, one TIE: the tool must count the
    # two and skip the tie.
    rows = [
        {"hi": train_ids[3], "lo": train_ids[0], "tie": False},
        {"hi": train_ids[5], "lo": train_ids[2], "tie": False},
        {"hi": train_ids[1], "lo": train_ids[4], "tie": True},
    ]
    (run_dir / "iterations" / "000").mkdir(parents=True)
    (run_dir / "iterations" / "000" / "ranking_pairs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "train_data": "data/train.jsonl",
                "selection_data": "data/selection.jsonl",
                "solver_backend": "llm",
                "solver_provider": provider,
                "solver_model": "mock-1" if provider == "mock" else "openai/gpt-5.6-luna",
                "solver_base_url": None,
                "solver_extra_body": None,
                "solver_cache_dir": "data/scorer_cache",
                "proposer_score_budget": budget,
            }
        ),
        encoding="utf-8",
    )
    return ToolRun(run_dir, cache_dir, train_ids, selection_ids, candidate, baseline)


def _request(run: ToolRun, **overrides) -> ScoreRequest:
    kwargs = dict(
        run_dir=run.run_dir,
        rubric=run.candidate / "rubric.md",
        criterion="Verification",
        rows=None,
        sample=4,
        repo_root=run.run_dir.parents[1],
    )
    kwargs.update(overrides)
    return ScoreRequest(**kwargs)


def _write_idmap(run: ToolRun, *, extra_ids: tuple[str, ...] = ()) -> Pseudonymizer:
    """The map the optimizer writes beside the runs, for this run.

    Built and saved through the real `Pseudonymizer` rather than as a
    hand-written dict, so these tests read the file a run actually leaves
    behind — including the detail that the reverse direction resolves to the
    full `…::att0` spelling and not the bare instance id.

    `extra_ids` stands for the rows that are in the map but not in train
    (a run's validation pool): the case where a pseudonym resolves to a real
    row the tool must still refuse.
    """

    pseudo = Pseudonymizer.for_pool([*run.train_ids, *extra_ids], salt="fixed")
    pseudo.save(run.run_dir.parent / "_idmap" / f"{run.run_dir.name}.json")
    return pseudo


def _cells(cache_dir: Path) -> list[Path]:
    return sorted((cache_dir / "per_criterion").glob("*.json")) if cache_dir.exists() else []


def _staged(run: ToolRun) -> list[Path]:
    return _cells(staging_dir_for(run.run_dir))


# --- scoring ------------------------------------------------------------------


def test_scores_train_rows_writes_cells_and_ends_with_the_budget_line(tmp_path):
    run = _write_run(tmp_path)
    report = score_criterion(_request(run))
    out = report.render()

    assert report.exit_code == 0
    assert out.rstrip().splitlines()[-1].startswith("budget: spent $0.00 of $1.50 this leg")
    assert "remaining $1.50" in out
    rows = report.payload["rows"]
    assert len(rows) == 4 and all(r["score"] is not None for r in rows)
    assert all(r["evidence"] == ["mock evidence for Verification"] for r in rows)
    # New criterion: the parent rubric has no "Verification", so no old score.
    assert report.payload["baseline_available"] is False
    assert "new criterion" in out
    # Four cells were STAGED inside the run directory, one per row; the run
    # cache itself was not written — the optimizer merges after the leg.
    assert len(_staged(run)) == 4
    for cell in _staged(run):
        assert json.loads(cell.read_text())["criterion"] == "Verification"
    assert _cells(run.cache_dir) == []
    # The JSON block is machine-readable and sits before the budget line.
    text, _, tail = out.partition("--- json ---\n")
    body, _, budget = tail.rpartition("\n")
    assert json.loads(body)["criterion"] == "Verification"
    assert budget.startswith("budget:")


def test_the_cells_it_buys_are_the_cells_the_full_pass_reads(tmp_path):
    """The whole point: same section split, same key, so a later `score()`
    over the candidate's full criteria list is a cache hit for that cell."""

    run = _write_run(tmp_path)
    ids = run.train_ids[:3]
    score_criterion(_request(run, rows=ids, sample=None))
    counts = merge_staged_cells(staging_dir_for(run.run_dir), run.cache_dir, run.train_ids)
    assert counts == {
        "merged": 3,
        "rejected_by_id": 0,
        "rejected_existing": 0,
        "rejected_unreadable": 0,
    }

    client = MockLLMClient()
    scorer = LLMRubricScorer(client=client, cache_dir=run.cache_dir)
    train = {
        t.trajectory_id: t for t in load_trajectories(run.run_dir.parents[1] / "data/train.jsonl")
    }
    criteria = ["Truth", "Noisy", "Verification"]
    for tid in ids:
        cell = scorer.score_one(
            train[tid], "Verification", criteria, CANDIDATE_RUBRIC, cache_only=True
        )
        assert cell.score is not None and cell.cached, tid
    # And the full-pass call pays for the OTHER two criteria only.
    scores, _, cost = scorer.score(train[ids[0]], criteria, CANDIDATE_RUBRIC)
    assert set(scores) == set(criteria)
    assert cost.llm_calls == 2


def test_the_old_score_and_the_judged_pairs_are_reported_for_an_existing_criterion(tmp_path):
    run = _write_run(tmp_path)
    # Warm the baseline's cells for "Truth" the way the iteration's train
    # pass would have, then ask about a candidate that rewrote "Truth".
    train = {
        t.trajectory_id: t for t in load_trajectories(run.run_dir.parents[1] / "data/train.jsonl")
    }
    scorer = LLMRubricScorer(client=MockLLMClient(), cache_dir=run.cache_dir)
    for traj in train.values():
        scorer.score(traj, ["Truth", "Noisy"], RUBRIC)
    rewritten = run.candidate / "rubric.md"
    rewritten.write_text(
        CANDIDATE_RUBRIC.replace(
            "+2 when the row is what the judge would rank first.",
            "+2 when the row is what a careful reviewer would rank first.",
        ),
        encoding="utf-8",
    )

    report = score_criterion(_request(run, criterion="Truth", rows=run.train_ids, sample=None))

    assert report.payload["baseline_available"] is True
    assert all(r["baseline_score"] is not None for r in report.payload["rows"])
    pairs = report.payload["judged_pairs"]
    assert pairs["n_judged_among_rows"] == 2, "the TIE pair is skipped"
    assert pairs["old_criterion_pairs"] == 2
    assert "judged pairs among these rows: 2" in report.text
    assert "| old" in report.text


def test_a_criterion_no_heading_names_is_refused(tmp_path):
    run = _write_run(tmp_path)
    report = score_criterion(_request(run, criterion="Nonexistent"))
    assert report.exit_code == 2
    assert "no line of rubric.md names the criterion" in report.text
    assert _cells(run.cache_dir) == []


# --- pseudonyms: the only names the proposer has ---------------------------------


def test_a_pseudonym_in_rows_resolves_to_its_train_row_and_scores(tmp_path):
    """`--rows` must not match pool ids only.

    The proposer is shown `traj_...` and nothing else, so matching pool ids
    alone would refuse every targeted call it makes, though every one of
    those ids resolves to a genuine train row.
    """

    run = _write_run(tmp_path)
    pseudo = _write_idmap(run)
    asked = [pseudo.mapping[t] for t in run.train_ids[:3]]

    report = score_criterion(_request(run, rows=asked, sample=None))

    assert report.exit_code == 0, report.text
    rows = report.payload["rows"]
    assert [r["trajectory_id"] for r in rows] == asked, "reported back in the agent's spelling"
    assert all(r["score"] is not None for r in rows)
    # Resolution is only for picking rows: the cells are keyed by the real
    # trajectory, which is what makes the merge into the run cache accept
    # them and the full pass read them.
    counts = merge_staged_cells(staging_dir_for(run.run_dir), run.cache_dir, run.train_ids)
    assert counts["merged"] == 3
    # And nothing the agent reads carries a pool id, which is the whole
    # reason the map lives outside the run directory.
    out = report.render()
    assert not any(tid in out for tid in run.train_ids)


def test_pseudonyms_and_pool_ids_mix_and_a_row_named_twice_is_scored_once(tmp_path):
    run = _write_run(tmp_path)
    pseudo = _write_idmap(run)
    first, second = run.train_ids[0], run.train_ids[1]

    report = score_criterion(
        _request(run, rows=[pseudo.mapping[first], second, first], sample=None)
    )

    assert report.exit_code == 0, report.text
    assert [r["trajectory_id"] for r in report.payload["rows"]] == [
        pseudo.mapping[first],
        pseudo.mapping[second],
    ]


def test_sampled_rows_come_back_as_pseudonyms_the_next_call_can_name(tmp_path):
    """Printing pool ids from `--sample` would hand over the join the map
    withholds — and leave the agent with no id it could pass back to
    `--rows`."""

    run = _write_run(tmp_path)
    _write_idmap(run)

    sampled = score_criterion(_request(run, sample=3))
    shown = [r["trajectory_id"] for r in sampled.payload["rows"]]
    assert all(name.startswith("traj_") for name in shown)

    again = score_criterion(_request(run, rows=shown, sample=None))
    assert again.exit_code == 0, again.text
    assert [r["trajectory_id"] for r in again.payload["rows"]] == shown
    assert all(r["cached"] for r in again.payload["rows"]), "the same rows, not a fresh sample"


def test_the_judged_pair_block_joins_a_pseudonymized_pairs_file(tmp_path):
    """`ranking_pairs.jsonl` inside a run is written redacted, so a block
    keyed on pool ids joined with nothing: every pseudonymized run read
    "judged pairs among these rows: none yet", whatever it asked for."""

    run = _write_run(tmp_path)
    pseudo = _write_idmap(run)
    pairs = run.run_dir / "iterations" / "000" / "ranking_pairs.jsonl"
    pairs.write_text(pseudo.redact(pairs.read_text(encoding="utf-8")), encoding="utf-8")

    report = score_criterion(
        _request(run, rows=[pseudo.mapping[t] for t in run.train_ids], sample=None)
    )

    assert report.payload["judged_pairs"]["n_judged_among_rows"] == 2, "the TIE pair is skipped"
    assert "judged pairs among these rows: 2" in report.text


# --- exposure guard -------------------------------------------------------------


def test_a_selection_row_id_is_refused_without_scoring_anything(tmp_path):
    run = _write_run(tmp_path)
    ids = [run.train_ids[0], run.selection_ids[0]]
    report = score_criterion(_request(run, rows=ids, sample=None))
    out = report.render()

    assert report.exit_code == 2
    assert "not in this run's train pool" in out
    assert run.selection_ids[0] in out, "the offending id is echoed so the agent can drop it"
    assert "selection" not in out.replace(run.selection_ids[0], ""), "and nothing says why"
    assert out.rstrip().splitlines()[-1].startswith("budget: spent $0.00 of $1.50")
    assert _cells(run.cache_dir) == []
    assert not (run.run_dir / LEDGER_FILENAME).exists(), "a refusal charges nothing"


def test_a_pseudonym_for_a_row_outside_the_train_pool_is_still_refused(tmp_path):
    """Resolution moved the lookup, not the boundary. A pseudonym is a name,
    not a permit: it buys the rows the agent may already have, nothing else."""

    run = _write_run(tmp_path)
    pseudo = _write_idmap(run, extra_ids=tuple(run.selection_ids))
    held_out = pseudo.mapping[run.selection_ids[0]]

    report = score_criterion(
        _request(run, rows=[pseudo.mapping[run.train_ids[0]], held_out], sample=None)
    )
    out = report.render()

    assert report.exit_code == 2
    assert "not in this run's train pool" in out and "unrecognised" not in out
    assert held_out in out, "the offending id is echoed so the agent can drop it"
    assert run.selection_ids[0] not in out, "and never under its real name"
    assert "selection" not in out.replace(held_out, ""), "nothing says which pool it is in"
    assert _staged(run) == [] and not (run.run_dir / LEDGER_FILENAME).exists()


def test_an_id_the_run_never_heard_of_is_refused_in_different_words(tmp_path):
    """The two refusals have to be told apart or the agent stops trying.

    Under one sentence for both, a mistyped pseudonym read as "only train
    rows can be scored here" — an authoritative boundary — and the proposer
    concluded the rows it cared about were held out. Distinguishing them
    says nothing about which pool anything is in; it says only that this
    string names no row, which the agent can fix by retyping it.
    """

    run = _write_run(tmp_path)
    pseudo = _write_idmap(run, extra_ids=tuple(run.selection_ids))
    # `traj_` plus eight hex digits, so a trailing `z` cannot collide.
    typo = pseudo.mapping[run.train_ids[0]][:-1] + "z"

    unknown = score_criterion(_request(run, rows=[typo], sample=None))
    held_out = score_criterion(
        _request(run, rows=[pseudo.mapping[run.selection_ids[0]]], sample=None)
    )

    assert unknown.exit_code == 2 and held_out.exit_code == 2
    assert "unrecognised id" in unknown.text and typo in unknown.text
    assert "train pool" not in unknown.text, "an unknown name is not told where rows live"
    assert "only train rows can be scored here" in held_out.text
    assert "unrecognised" not in held_out.text
    # Without a map there is no vocabulary to judge a name against, so the
    # single sentence stands — which is what every non-agentic run reads.
    plain = _write_run(tmp_path / "plain")
    mapless = score_criterion(_request(plain, rows=["nothing-like-a-row"], sample=None))
    assert "not in this run's train pool" in mapless.text


def test_more_than_twenty_rows_is_refused(tmp_path):
    run = _write_run(tmp_path)
    report = score_criterion(_request(run, rows=[f"x{i}" for i in range(21)], sample=None))
    assert report.exit_code == 2 and "at most 20 per call" in report.text


def test_the_cli_refuses_a_selection_id_with_exit_two(tmp_path, capsys):
    run = _write_run(tmp_path)
    code = main(
        [
            "--run-dir",
            str(run.run_dir),
            "--rubric",
            str(run.candidate / "rubric.md"),
            "--criterion",
            "Verification",
            "--rows",
            run.selection_ids[1],
            "--repo-root",
            str(run.run_dir.parents[1]),
        ]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "refused" in out and out.rstrip().splitlines()[-1].startswith("budget:")


# --- budget ---------------------------------------------------------------------


@dataclass
class PricedClient:
    """Answers like the mock, but every call reports a cost."""

    cost_usd: float
    model: str = "openai/gpt-5.6-luna"
    mock: MockLLMClient = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.mock = MockLLMClient(model=self.model)

    def generate(self, prompt, **kw):
        inner = self.mock.generate(prompt, **kw)
        return LLMResponse(
            text=inner.text,
            prompt_tokens=inner.prompt_tokens,
            completion_tokens=inner.completion_tokens,
            model=self.model,
            cost_usd=self.cost_usd,
            cost_source="openrouter",
        )


def test_the_ledger_accumulates_and_refuses_past_the_cap(tmp_path):
    run = _write_run(tmp_path, budget=1.5, provider="openai")
    factory = lambda settings: PricedClient(cost_usd=0.15)  # noqa: E731
    seen: list[str] = []
    for i in range(3):
        rows = [run.train_ids[i]]
        report = score_criterion(
            _request(run, rows=rows, sample=None, criterion="Verification"), client_factory=factory
        )
        assert report.exit_code == 0, report.text
        seen.append(report.budget_line)
    assert seen[0].startswith("budget: spent $0.15 of $1.50 this leg, remaining $1.35")
    assert seen[2].startswith("budget: spent $0.45 of $1.50 this leg, remaining $1.05")
    ledger = ScoreBudgetLedger.load(run.run_dir)
    assert ledger.calls == 3 and abs(ledger.spent_usd - 0.45) < 1e-9
    assert [e["rows"] for e in ledger.entries] == [1, 1, 1]

    # Spend the rest, then the next call is refused and charges nothing.
    ledger.spent_usd = 1.5
    ledger.save()
    report = score_criterion(
        _request(run, rows=[run.train_ids[4]], sample=None), client_factory=factory
    )
    assert report.exit_code == 2
    assert "scoring budget is spent" in report.text
    assert report.budget_line.startswith("budget: spent $1.50 of $1.50 this leg, remaining $0.00")
    assert ScoreBudgetLedger.load(run.run_dir).calls == 3


def test_a_call_whose_estimate_would_pass_the_cap_is_refused_before_scoring(tmp_path):
    """Priced model, near-zero cap: the list-price estimate alone refuses."""

    run = _write_run(tmp_path, budget=0.000001, provider="openai")
    report = score_criterion(
        _request(run, rows=run.train_ids[:3], sample=None),
        client_factory=lambda settings: PricedClient(cost_usd=0.0),
    )
    assert report.exit_code == 2
    assert "estimated at" in report.text and "ask for fewer rows" in report.text
    assert _cells(run.cache_dir) == []


def test_the_optimizer_resets_the_ledger_per_leg_and_bills_the_proposer(tmp_path):
    """The tool runs in the agent's subprocess; the optimizer opens a leg,
    reads what the leg spent and adds it to `cost_proposer`."""

    class ChargingProposer(ScriptedProposer):
        """Stands in for an agent that called the tool once per leg."""

        def __init__(self, script, client, run_dir: Path) -> None:
            super().__init__(script, client)
            self.run_dir = run_dir
            self.legs_seen: list[tuple[int | None, str, float]] = []

        def _charge(self, cost: float) -> None:
            ledger = ScoreBudgetLedger.load(self.run_dir)
            self.legs_seen.append((ledger.iteration, ledger.leg, ledger.spent_usd))
            ledger.charge({"cost_usd": cost, "llm_calls": 2, "llm_tokens": 500, "rows": 10})
            ledger.save()

        def propose(self, **kwargs):
            self._charge(0.25)
            return super().propose(**kwargs)

        def revise(self, **kwargs):
            self._charge(0.10)
            return super().revise(**kwargs)

    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run,
        script={0: [("truth", TRUTH_POLICY)], 1: [("truth2", TRUTH_POLICY)]},
        iterations=2,
        run_id="ledger",
    )
    optimizer.proposer = ChargingProposer(
        optimizer.proposer.script, run.judge, optimizer.store.run_dir
    )
    optimizer.score_tool_enabled = True
    optimizer.config.proposer_score_budget = 2.0
    run_dir = optimizer.run()

    # Iteration 0 has no judged pairs, so no revision leg: one charge.
    # Iteration 1 prechecks and revises: two charges, two fresh legs.
    record0 = json.loads((run_dir / "iterations/000/record.json").read_text())
    record1 = json.loads((run_dir / "iterations/001/record.json").read_text())
    assert abs(record0["cost_proposer"] - 0.25) < 1e-9
    assert abs(record1["cost_proposer"] - 0.35) < 1e-9
    assert record0["unpriced_calls"] == 0
    # Every leg started from zero with the configured cap.
    assert optimizer.proposer.legs_seen == [
        (0, "propose", 0.0),
        (1, "propose", 0.0),
        (1, "revise", 0.0),
    ]
    ledger = ScoreBudgetLedger.load(run_dir)
    assert ledger.cap_usd == 2.0
    assert [(h["iteration"], h["leg"], h["spent_usd"]) for h in ledger.history] == [
        (0, "propose", 0.25),
        (1, "propose", 0.25),
    ]
    assert (ledger.iteration, ledger.leg, ledger.spent_usd) == (1, "revise", 0.10)


def test_without_the_tool_the_optimizer_writes_no_ledger(tmp_path):
    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="off"
    )
    assert optimizer.score_tool_enabled is False, "static proposer, deterministic scorer"
    assert optimizer.proposer.score_tool is None
    run_dir = optimizer.run()
    assert not (run_dir / LEDGER_FILENAME).exists()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert cfg["proposer_score_budget"] is None
    # The fixture's MetaScorer is not an LLM scorer: nothing for the tool to rebuild.
    assert cfg["solver_backend"] == "heuristic" and cfg["solver_provider"] is None


# --- what the optimizer records for the tool to rebuild the scorer ----------------


def test_config_yaml_records_the_solver_the_tool_needs(tmp_path):
    from meta_filter.core.solver import Solver

    run = prepare(tmp_path)
    optimizer = build_optimizer(
        run, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="cfg"
    )
    optimizer.solver = Solver(
        rubric_scorer=LLMRubricScorer(
            client=MockLLMClient(model="mock-1"), cache_dir=tmp_path / "cache"
        ),
        lazy_scoring=False,
    )
    run_dir = optimizer.run()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert cfg["solver_backend"] == "llm"
    assert cfg["solver_provider"] == "mock"
    assert cfg["solver_model"] == "mock-1"
    assert cfg["solver_cache_dir"] == str(tmp_path / "cache")


# --- the harness holes --------------------------------------------------------------


def test_access_for_spells_the_absolute_command():
    access = access_for(Path("runs/run_x"), budget_usd=1.5)
    assert access.command.startswith(f"{sys.executable} -m meta_filter.score_tool --run-dir /")
    assert access.command.endswith(str(Path("runs/run_x").resolve()))
    assert access.executable == sys.executable


def test_codex_config_opens_network_and_the_cache_only_when_asked(tmp_path):
    base = CodexCliClient(model="openai/gpt-5.6-luna", allowed_tools=("Bash",))
    plain = base.render_config()
    assert "[sandbox_workspace_write]" not in plain, "no sandbox section by default"

    armed = CodexCliClient(
        model="openai/gpt-5.6-luna",
        allowed_tools=("Bash",),
        network_access=True,
        writable_roots=(tmp_path / "cache",),
    ).render_config()
    section = armed[armed.index("[sandbox_workspace_write]") :]
    assert "network_access = true" in section
    assert f'writable_roots = ["{(tmp_path / "cache").resolve()}"]' in section
    assert armed.index("[sandbox_workspace_write]") < armed.index("[model_providers.")
    # Everything else is the plain config.
    assert armed.replace(section[: section.index("[model_providers.")], "") == plain

    static = CodexCliClient(
        model="openai/gpt-5.6-luna", network_access=True, writable_roots=(tmp_path,)
    ).render_config()
    assert "[sandbox_workspace_write]" not in static, "a read-only sandbox has no such holes"


def test_the_claude_guard_allows_the_tool_interpreter_and_nothing_else(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    other = tmp_path / "venv" / "bin" / "pip"
    other.write_text("")
    command = (
        f"{venv_python} -m meta_filter.score_tool --run-dir {root} "
        "--rubric iterations/000/x/rubric.md"
    )

    assert bash_violations(command, root, root) == [str(venv_python)], "blocked without the allow"
    assert bash_violations(command, root, root, allow=(str(venv_python),)) == []
    assert bash_violations(f"{other} install x", root, root, allow=(str(venv_python),)) == [
        str(other)
    ]

    settings = json.loads(build_guard_settings(root, root / ".audit", allow=(str(venv_python),)))
    assert settings["env"][ENV_ALLOW] == str(venv_python.resolve())
    assert ENV_ALLOW not in json.loads(build_guard_settings(root, root / ".audit"))["env"]


def test_with_tools_sets_exactly_the_hole_each_harness_has(tmp_path):
    from meta_filter.llm.cost_meter import metered, unwrap
    from meta_filter.propose.agentic_proposer import _with_tools

    access = ScoreToolAccess(
        command="/v/bin/python -m meta_filter.score_tool --run-dir /r",
        budget_usd=1.5,
        executable="/v/bin/python",
    )
    claude = _with_tools(
        metered(ClaudeCodeHeadlessClient(scratch_dir=tmp_path / "c"), "proposer"),
        ("Bash",),
        score_tool=access,
    )
    inner = unwrap(claude)
    assert inner.guard_allow == ("/v/bin/python",)
    assert not hasattr(inner, "network_access")

    codex = _with_tools(
        CodexCliClient(model="openai/gpt-5.6-luna", scratch_dir=tmp_path / "d"),
        ("Bash",),
        score_tool=access,
    )
    assert codex.network_access is True
    # The run cache is NOT opened to writes: the tool stages inside the run
    # directory and the optimizer merges. Only the network hole is rendered.
    assert codex.writable_roots == ()
    config = codex.render_config()
    assert "network_access = true" in config
    assert "writable_roots" not in config

    untouched = _with_tools(
        CodexCliClient(model="openai/gpt-5.6-luna", scratch_dir=tmp_path / "e"), ("Bash",)
    )
    assert untouched.network_access is False and untouched.writable_roots == ()


def test_the_optimizer_arms_the_tool_for_an_agentic_proposer_over_an_llm_scorer(tmp_path):
    """The one configuration where the tool can work is the one it is
    advertised in: a tool-using proposer and an LLM scorer."""

    from meta_filter.core.solver import Solver
    from meta_filter.gate.selector import Selector
    from meta_filter.gate.validator import Validator
    from meta_filter.loop.optimizer import Optimizer, OptimizerConfig
    from meta_filter.propose.agentic_proposer import AgenticProposer
    from tests.selection_gate_fixture import Backend

    run = prepare(tmp_path)
    cache = tmp_path / "scorer_cache"
    optimizer = Optimizer(
        initial_filter_dir=run.initial,
        train_data=run.train,
        validation_data=None,
        runs_dir=run.runs_dir,
        memory_dir=run.root / "memory",
        proposer=AgenticProposer(
            client=ClaudeCodeHeadlessClient(scratch_dir=tmp_path / "scratch"), n_candidates=3
        ),
        selector=Selector(),
        solver=Solver(rubric_scorer=LLMRubricScorer(client=MockLLMClient(), cache_dir=cache)),
        validator=Validator(loss_weights={}, critic_backend=Backend(run.judge)),
        config=OptimizerConfig(
            iterations=1,
            selection_data=run.selection,
            pair_judge_cache_dir=run.root / "pair_cache",
            proposer_score_budget=0.75,
        ),
        run_id="armed",
    )
    assert optimizer.score_tool_enabled is True
    access = optimizer.proposer.score_tool
    assert access is not None and access is optimizer.score_tool
    assert access.command.endswith(str((run.runs_dir / "armed").resolve()))
    assert access.budget_usd == 0.75
    assert access.executable == sys.executable


# --- staging and the merge ---------------------------------------------------------


def _fake_cell(cache_root: Path, trajectory_id: str, criterion: str, score: float) -> Path:
    """A per-criterion cell file named as the scorer names it."""

    from meta_filter.core.llm_scorer import sanitize_trajectory_id

    target = cache_root / "per_criterion"
    target.mkdir(parents=True, exist_ok=True)
    stem = f"{sanitize_trajectory_id(trajectory_id)}__crit_deadbeef_0123456789__view_x"
    path = target / f"{stem}.json"
    path.write_text(
        json.dumps(
            {
                "trajectory_id": trajectory_id,
                "model": "mock-1",
                "criterion": criterion,
                "section_sha1": "x",
                "score": score,
                "evidence": [f"[{criterion}] forged"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_the_merge_takes_train_cells_only_never_overwrites_and_clears_the_staging(tmp_path):
    run = _write_run(tmp_path)
    staging = staging_dir_for(run.run_dir)
    # What a leg could leave behind: a real train cell, a forged selection
    # cell, a train cell whose key the cache already holds, a file whose
    # JSON id disagrees with its filename, and garbage.
    _fake_cell(staging, run.train_ids[0], "Verification", 1.0)
    _fake_cell(staging, run.selection_ids[0], "Verification", 2.0)
    existing = _fake_cell(run.cache_dir, run.train_ids[1], "Verification", -1.0)
    _fake_cell(staging, run.train_ids[1], "Verification", 2.0)
    mismatched = _fake_cell(staging, run.train_ids[2], "Verification", 0.0)
    mismatched.write_text(
        mismatched.read_text().replace(run.train_ids[2], run.selection_ids[1]), encoding="utf-8"
    )
    garbage = _fake_cell(staging, run.train_ids[3], "Verification", 0.0)
    garbage.write_text("{", encoding="utf-8")

    counts = merge_staged_cells(staging, run.cache_dir, run.train_ids)

    assert counts == {
        "merged": 1,
        "rejected_by_id": 2,
        "rejected_existing": 1,
        "rejected_unreadable": 1,
    }
    names = {c.name for c in _cells(run.cache_dir)}
    assert any(n.startswith(run.train_ids[0]) for n in names)
    assert not any(run.selection_ids[0] in n or run.selection_ids[1] in n for n in names)
    assert json.loads(existing.read_text())["score"] == -1.0, "the canonical cell stands"
    assert not staging.exists(), "the staging dir is cleared, rejects included"
    # Nothing staged, nothing merged, and an absent staging dir is fine.
    assert merge_staged_cells(staging, run.cache_dir, run.train_ids)["merged"] == 0


def test_the_tool_reads_the_run_cache_read_only_and_survives_an_unreadable_one(tmp_path):
    """Existing cells are served from the run cache without writing it; a
    cache the sandbox hides is a miss, not an error."""

    import os

    run = _write_run(tmp_path)
    train = {
        t.trajectory_id: t for t in load_trajectories(run.run_dir.parents[1] / "data/train.jsonl")
    }
    scorer = LLMRubricScorer(client=MockLLMClient(), cache_dir=run.cache_dir)
    for traj in train.values():
        scorer.score(traj, ["Truth", "Noisy"], RUBRIC)
    before = {c.name: c.stat().st_mtime_ns for c in _cells(run.cache_dir)}

    report = score_criterion(_request(run, criterion="Noisy", rows=run.train_ids[:3], sample=None))
    assert all(r["cached"] for r in report.payload["rows"]), "served from the run cache"
    assert _staged(run) == [], "nothing new, nothing staged"
    assert {c.name: c.stat().st_mtime_ns for c in _cells(run.cache_dir)} == before

    if os.geteuid() == 0:
        return  # root reads through chmod; the fallback cannot be provoked
    os.chmod(run.cache_dir, 0)
    try:
        report = score_criterion(
            _request(run, criterion="Noisy", rows=run.train_ids[:2], sample=None)
        )
    finally:
        os.chmod(run.cache_dir, 0o755)
    assert report.exit_code == 0
    assert not any(r["cached"] for r in report.payload["rows"]), "a hidden cache is a miss"
    assert len(_staged(run)) == 2


def test_the_optimizer_merges_the_staged_cells_after_each_leg(tmp_path, caplog):
    """The fold after a proposer leg is the one path from the workspace into
    the run cache: train cells merged, a forged selection cell dropped and
    logged, the staging dir gone before the solver pass."""

    from tests.test_promotion_rescore_is_canonical import _llm_optimizer

    optimizer, run, _client = _llm_optimizer(
        tmp_path, script={0: [("truth", TRUTH_POLICY)]}, iterations=1, run_id="merge"
    )
    cache_dir = Path(optimizer.solver.rubric_scorer.cache_dir)
    staging = staging_dir_for(optimizer.store.run_dir)

    class StagingProposer(ScriptedProposer):
        def propose(self, **kwargs):
            _fake_cell(staging, run.train_ids[0], "Truth", 1.0)
            _fake_cell(staging, run.selection_ids[0], "Truth", 2.0)
            return super().propose(**kwargs)

    optimizer.proposer = StagingProposer(optimizer.proposer.script, run.judge)
    optimizer.score_tool_enabled = True
    optimizer.run()

    assert not staging.exists()
    names = [c.name for c in _cells(cache_dir)]
    assert any("deadbeef" in n and n.startswith(run.train_ids[0]) for n in names)
    assert not any("deadbeef" in n and run.selection_ids[0] in n for n in names)
    assert "merged 1 staged scorer cell(s)" in caplog.text
    assert "1 not a train row" in caplog.text


def test_an_armed_codex_client_lists_no_writable_root(tmp_path):
    from meta_filter.propose.agentic_proposer import _with_tools

    access = access_for(tmp_path / "run", budget_usd=1.5)
    codex = _with_tools(
        CodexCliClient(model="openai/gpt-5.6-luna", scratch_dir=tmp_path / "ws"),
        ("Bash",),
        score_tool=access,
    )
    config = codex.render_config()
    assert "[sandbox_workspace_write]" in config and "network_access = true" in config
    assert "writable_roots" not in config
    assert STAGING_DIRNAME not in config
