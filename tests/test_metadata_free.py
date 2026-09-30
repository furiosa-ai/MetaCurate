"""No harness outcome reaches any LLM, cache or prompt.

Covers, in order: the strip at load; that a stripped DSv4Pro row renders to
the judge and the scorer exactly as a Nemotron row does; the strict precheck;
cache-key separation between a row's raw and stripped view; the refusal of a
selection set that does not show it was judged on stripped rows, and the
re-judge path; the four `configs/metadata_free/` cells; the launcher's dry
run; and two mock end-to-end runs (the CLI with the static proposer, and the
optimizer with a scripted tool-using proposer) whose every LLM prompt is
scanned.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from meta_filter.llm.base import LLMResponse
from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient
from meta_filter.llm.mock import MockLLMClient
from meta_filter.llm_scorer import _criterion_cache_key
from meta_filter.metadata_free import (
    KEPT_METADATA_FIELDS,
    METADATA_FREE_TRAIN_FILE,
    filter_outcome_reads,
    outcome_mentions,
    selection_set_view,
    strip_row,
)
from meta_filter.pair_judge import PAIR_METADATA_LINE, PAIR_PROMPT, PairJudge, _meta_line
from meta_filter.schemas import Trajectory
from meta_filter.trajectory import load_trajectories
from meta_filter.view_policy import (
    SCORER_BASELINE_FIELDS,
    absent_field_signature,
    field_set_signature,
    pair_view_signature,
    scorer_key_fields,
    scorer_metadata_block,
)

REPO = Path(__file__).resolve().parents[1]
CELLS = ("dsv4pro_blank", "dsv4pro_seeded", "nemotron_blank", "nemotron_seeded")
BLANK_SEED = "filters/initial_blank_wired_v2"
NONBLANK_SEED = "filters/initial_ranking_nometa"

# The pair judge's fixed metadata line (`pair_judge.PAIR_METADATA_LINE`). The
# scans below remove it before looking for outcome names.
NULL_JUDGE_LINE = PAIR_METADATA_LINE
SENTINEL = "SENTINEL"


def dsv4pro_row(tid: str = "owner__repo-1::att0", i: int = 1) -> dict:
    """A DSv4Pro row as the pool files carry it: every field, labels included.

    Values an LLM must never see carry `SENTINEL`, so a scan finds them even
    where a key name would be reworded.
    """

    return {
        "trajectory_id": tid,
        "task": f"Fix the off-by-one in pager {i}",
        "turns": [
            {"role": "assistant", "action": "grep", "content": f"grep -rn page{i} src"},
            {"role": "assistant", "action": "edit", "content": f"sed -i 's/a/b{i}/' src/p.py"},
            {"role": "assistant", "action": "test", "content": "pytest tests -q"},
        ][: 2 + i % 2],
        "metadata": {
            "source": "mini-swe-agent-1.1",
            "instance_id": f"{SENTINEL}INSTANCE-{i}",
            "host": f"{SENTINEL}HOST",
            "attempt_idx": i % 3,
            "api_calls": 10 + i,
            "n_messages": 20 + 2 * i,
            "submission": f"diff --git a/src/p{i}.py b/src/p{i}.py\n+x = {i}\n",
            "submission_bytes": 40 + i,
            "exit_status": f"{SENTINEL}EXIT",
            "resolved": i % 2 == 0,
            "f2p_pass": i % 2 == 0,
            "p2p_pass": i % 3 != 0,
            "probe_stratum": f"{SENTINEL}STRATUM",
            "gold_patch": f"{SENTINEL}GOLD",
            "test_patch": f"{SENTINEL}TESTPATCH",
        },
        # `Trajectory` allows extras; the demo pools carry a label this way.
        "gold_decision": f"{SENTINEL}DECISION",
    }


def nemotron_row(tid: str = "nemotron.abc", i: int = 1) -> dict:
    """A Nemotron row exactly as `nemotron_train50.jsonl` carries it."""

    return {
        "trajectory_id": tid,
        "task": f"Fix the off-by-one in pager {i}",
        "turns": [{"role": "assistant", "action": "edit", "content": "sed -i s/a/b/ x.py"}],
        "metadata": {
            "source": "nemotron-v3-mini-xml",
            "submission": "diff --git a/x.py b/x.py\n+y\n",
            "license": "cc-by-4.0",
            "dataset": "nemotron-mini",
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def leaks(text: str) -> list[str]:
    """Outcome fields or sentinel values in `text`, once the null judge line is removed."""

    cleaned = text.replace(NULL_JUDGE_LINE, "")
    found = outcome_mentions(cleaned)
    if SENTINEL in cleaned:
        found.append(SENTINEL)
    return found


# --- the strip -----------------------------------------------------------------


def test_a_stripped_dsv4pro_row_keeps_only_the_nemotron_shape():
    row = strip_row(dsv4pro_row())

    assert set(row) == {"trajectory_id", "task", "turns", "metadata"}
    assert list(row["metadata"]) == [
        "source",
        "api_calls",
        "n_messages",
        "submission",
        "submission_bytes",
    ], "key order is preserved: the scorer's metadata block walks the dict"
    assert set(row["metadata"]) <= KEPT_METADATA_FIELDS
    assert not leaks(json.dumps(row))


def test_a_nemotron_row_passes_the_strip_byte_for_byte():
    row = nemotron_row()
    assert json.dumps(strip_row(row)) == json.dumps(row)


def test_load_trajectories_strips_every_row(tmp_path):
    pool = write_jsonl(tmp_path / "pool.jsonl", [dsv4pro_row(f"r{i}", i) for i in range(4)])

    for traj in load_trajectories(pool):
        assert set(traj.metadata) <= KEPT_METADATA_FIELDS
        assert not hasattr(traj, "gold_decision")
        assert not leaks(traj.model_dump_json())


def test_the_nemotron_pool_loads_unchanged(tmp_path):
    rows = [nemotron_row(f"n{i}", i) for i in range(3)]
    pool = write_jsonl(tmp_path / "nemo.jsonl", rows)
    assert [t.model_dump_json() for t in load_trajectories(pool)] == [
        Trajectory.model_validate(r).model_dump_json() for r in rows
    ]


def test_the_probe_set_is_stripped_too(tmp_path):
    from meta_filter.probe import ProbeSet

    pool = write_jsonl(tmp_path / "probe.jsonl", [dsv4pro_row("p0", 0)])
    probe = ProbeSet.from_path(pool)
    assert "probe_stratum" not in probe.trajectories[0].metadata


# --- rendering equivalence -------------------------------------------------------


def _traj(row: dict) -> Trajectory:
    return Trajectory.model_validate(row)


def _nemotron_shaped_twin(row: dict) -> dict:
    """A row the Nemotron adapter could write, carrying the DSv4Pro row's own
    content — the "modulo the row's own content" of the equivalence."""

    stripped = strip_row(row)
    return {**stripped, "metadata": dict(stripped["metadata"])}


def test_the_judge_metadata_line_of_a_stripped_row_is_the_nemotron_line():
    stripped = _traj(strip_row(dsv4pro_row()))
    nemotron = _traj(nemotron_row())

    assert _meta_line(stripped) == _meta_line(nemotron) == NULL_JUDGE_LINE
    assert NULL_JUDGE_LINE == (
        '{"resolved": null, "f2p_pass": null, "p2p_pass": null, "exit_status": null}'
    ), "the line is part of every pair-judge cache key's prompt; it must not move"


def test_the_scorer_metadata_block_matches_a_nemotron_shaped_row():
    stripped = _traj(strip_row(dsv4pro_row()))
    twin = _traj(_nemotron_shaped_twin(dsv4pro_row()))

    block = scorer_metadata_block(stripped.metadata, SCORER_BASELINE_FIELDS)
    assert block == scorer_metadata_block(twin.metadata, SCORER_BASELINE_FIELDS)
    assert not leaks(block)
    # The same renderer on a real Nemotron row: only its provenance key.
    assert scorer_metadata_block(nemotron_row()["metadata"], SCORER_BASELINE_FIELDS) == (
        json.dumps({"source": "nemotron-v3-mini-xml"}, indent=2)
    )


def test_the_whole_pair_prompt_of_stripped_rows_carries_nothing_but_the_null_line():
    a = _traj(strip_row(dsv4pro_row("a", 1)))
    b = _traj(strip_row(dsv4pro_row("b", 2)))
    seen: list[str] = []

    @dataclass
    class _Client:
        model: str = "rec"

        def generate(self, prompt, **_kw):
            seen.append(prompt)
            return LLMResponse(text='{"winner": "A", "rationale": "x"}', prompt_tokens=0)

    import tempfile

    with tempfile.TemporaryDirectory() as cache:
        PairJudge(client=_Client(), cache_dir=Path(cache)).resolve_pairs_detail(
            [("a", "b")], {"a": a, "b": b}
        )
    assert len(seen) == 2 and all(p.count(NULL_JUDGE_LINE) == 2 for p in seen)
    assert all(not leaks(p) for p in seen)
    assert PAIR_PROMPT.count("{meta_a}") == 1


# --- the strict precheck ---------------------------------------------------------


def _filter(root: Path, *, rules: str = "", rubric: str = "", policy: str = "") -> Path:
    from meta_filter.filter_loader import materialize_candidate

    materialize_candidate(
        root,
        rubric_md=rubric or "# Rubric\n\n## Criterion: Quality\n+2 good.\n",
        rules_py=rules or "def rule_noop(trajectory):\n    return None\n",
        policy_py=policy
        or (
            "RUBRIC_CRITERIA = ['Quality']\n\n\ndef decide(trajectory, scores, triggered):\n"
            "    return {'decision': 'REVIEW', 'score': scores['Quality'], 'confidence': 0.5}\n"
        ),
    )
    return root


def _check(candidate: Path, initial: Path):
    from meta_filter.constraints import check_candidate

    return check_candidate(candidate, baseline_dir=initial, initial_dir=initial)


@pytest.mark.parametrize(
    ("where", "text"),
    [
        ("rules", "def rule_x(t):\n    return t.metadata.get('resolved')\n"),
        ("rules", "def rule_x(t):\n    return t.metadata['exit_status'] == 'Submitted'\n"),
        ("rubric", "# Rubric\n\n## Criterion: Quality\nCompare to `metadata.gold_patch`.\n"),
        ("rubric", "# Rubric\n\n## Criterion: Quality\n<!-- reads: resolved -->\n+2 good.\n"),
        (
            "policy",
            "RUBRIC_CRITERIA = ['Quality']\n# p2p_pass\ndef decide(t, s, r):\n    return {}\n",
        ),
    ],
)
def test_any_outcome_read_is_invalid_with_no_initial_filter_exemption(tmp_path, where, text):
    kwargs = {where: text}
    initial = _filter(tmp_path / "initial", **kwargs)
    candidate = _filter(tmp_path / "cand", **kwargs)

    strict = _check(candidate, initial)
    assert strict.status == "invalid"
    assert any("outcome metadata" in r for r in strict.reasons)


def test_resolved_as_an_english_word_is_not_a_read(tmp_path):
    prose = "# Rubric\n\n## Criterion: Quality\n+2 the issue is resolved and verified.\n"
    initial = _filter(tmp_path / "initial", rubric=prose)
    candidate = _filter(tmp_path / "cand", rubric=prose)
    assert not any("outcome" in r for r in _check(candidate, initial).reasons)


@pytest.mark.parametrize("seed", [BLANK_SEED, NONBLANK_SEED])
def test_both_seeds_pass_the_strict_precheck(tmp_path, seed):
    assert filter_outcome_reads(REPO / seed) == {}
    copy = tmp_path / "cand"
    shutil.copytree(REPO / seed, copy, ignore=shutil.ignore_patterns("__pycache__"))
    report = _check(copy, REPO / seed)
    assert not any("outcome metadata" in r for r in report.reasons)


def test_the_non_blank_seed_names_no_label_anywhere():
    text = "\n".join(
        (REPO / NONBLANK_SEED / name).read_text(encoding="utf-8")
        for name in ("rubric.md", "rules.py", "policy.py", "rationale.md")
    )
    assert not outcome_mentions(text)
    assert not re.search(r"(?i)\blabel|gold|instance_id|harness", text)


def test_the_non_blank_seed_fires_the_same_way_on_both_pools():
    from meta_filter.filter_loader import load_filter

    loaded = load_filter(REPO / NONBLANK_SEED)
    dsv4 = _traj(strip_row(dsv4pro_row()))
    nemo = _traj(nemotron_row())
    rule = loaded.rules_module.rule_nemotron_no_valid_submission
    assert rule(dsv4) is None and rule(nemo) is None
    broken = _traj({**nemotron_row(), "metadata": {"submission": "not a diff"}})
    assert rule(broken)["verdict"] == "REJECT"


def test_a_seed_that_names_an_outcome_is_refused_before_the_run_builds_anything(tmp_path):
    from meta_filter.optimizer import OptimizerConfig

    pool = write_jsonl(tmp_path / "train.jsonl", [dsv4pro_row("t0", 0)])
    selection = write_jsonl(tmp_path / "sel.jsonl", [dsv4pro_row("s0", 0)])
    with pytest.raises(ValueError, match="refusing initial filter"):
        _optimizer(
            tmp_path,
            train=pool,
            initial=REPO / "tests/fixtures/filters/negative_reads_outcome",
            config=OptimizerConfig(iterations=1, selection_data=selection),
        )
    assert not (tmp_path / "runs").exists()


# --- cache-key separation ----------------------------------------------------------


def test_the_scorer_key_of_the_stripped_row_differs_from_the_labelled_one():
    labelled = _traj(dsv4pro_row())
    stripped = _traj(strip_row(dsv4pro_row()))

    def key(traj: Trajectory) -> str:
        sig = field_set_signature("scorer", SCORER_BASELINE_FIELDS) + absent_field_signature(
            scorer_key_fields(SCORER_BASELINE_FIELDS), traj.metadata
        )
        return _criterion_cache_key("section", "Quality", traj, "luna", sig, "hdr")

    assert key(labelled) != key(stripped), "a labelled cell would be served to the stripped run"
    # Separate by the view alone, too: the same id, labelled vs stripped metadata.
    same_id = _traj({**strip_row(dsv4pro_row()), "trajectory_id": labelled.trajectory_id})
    assert key(labelled) != key(same_id)


def test_the_nemotron_scorer_key_is_unchanged_by_the_strip(tmp_path):
    pool = write_jsonl(tmp_path / "nemo.jsonl", [nemotron_row()])
    plain = Trajectory.model_validate(nemotron_row())
    (free,) = load_trajectories(pool)
    fields = scorer_key_fields(SCORER_BASELINE_FIELDS)
    sig = absent_field_signature(fields, plain.metadata)
    assert sig == absent_field_signature(fields, free.metadata)
    assert _criterion_cache_key("s", "Q", plain, "m", sig) == _criterion_cache_key(
        "s", "Q", free, "m", sig
    )


def test_the_pair_judge_key_of_the_stripped_pair_differs_from_the_labelled_one(tmp_path):
    labelled = [_traj(dsv4pro_row("a", 1)), _traj(dsv4pro_row("b", 2))]
    stripped = [_traj(strip_row(dsv4pro_row("a", 1))), _traj(strip_row(dsv4pro_row("b", 2)))]
    judge = PairJudge(client=MockLLMClient(), cache_dir=tmp_path / "pairs")

    lv_labelled = pair_view_signature(*labelled)
    lv_stripped = pair_view_signature(*stripped)
    assert lv_labelled == "" and lv_stripped != ""
    assert judge._cache_path("a", "b", lv_labelled, "ab") != judge._cache_path(
        "a", "b", lv_stripped, "ab"
    )
    # ... and it is the SAME namespace a Nemotron pair lives in.
    assert lv_stripped == pair_view_signature(_traj(nemotron_row("n1")), _traj(nemotron_row("n2")))


def test_the_trajectory_critic_keys_the_stripped_row_apart(tmp_path):
    from meta_filter.critics.trajectory_critic import TrajectoryCritic

    critic = TrajectoryCritic(client=MockLLMClient(), cache_dir=tmp_path / "judge")
    labelled = _traj(dsv4pro_row())
    critic.register_label_views([labelled])
    path_labelled = critic.cache_path_for(labelled.trajectory_id)
    critic2 = TrajectoryCritic(client=MockLLMClient(), cache_dir=tmp_path / "judge")
    stripped = _traj(strip_row(dsv4pro_row()))
    critic2.register_label_views([stripped])
    assert critic2.cache_path_for(stripped.trajectory_id) != path_labelled


# --- the selection set ---------------------------------------------------------------


def _provenance(rows: Path, **extra) -> dict:
    from meta_filter.metadata_free import file_sha256

    return {"rows_file": str(rows), "rows_sha256": file_sha256(rows), **extra}


def test_a_selection_set_is_accepted_only_when_label_freedom_is_shown(tmp_path):
    labelled = write_jsonl(tmp_path / "sel.jsonl", [dsv4pro_row(f"s{i}", i) for i in range(3)])
    nemo = write_jsonl(tmp_path / "nemo.jsonl", [nemotron_row(f"n{i}", i) for i in range(3)])

    assert "fields in view" in (selection_set_view(_provenance(labelled)) or "")
    assert selection_set_view(_provenance(labelled, metadata_free=True)) is None
    assert selection_set_view(_provenance(nemo)) is None
    # Every way of not being able to look is a refusal, never a pass.
    assert "no rows_file" in selection_set_view({})
    assert "cannot be opened" in selection_set_view(
        {"rows_file": str(tmp_path / "absent.jsonl"), "rows_sha256": "x"}
    )
    assert "no rows_sha256" in selection_set_view({"rows_file": str(nemo)})
    assert "changed since" in selection_set_view({"rows_file": str(nemo), "rows_sha256": "0" * 64})


def test_a_labelled_set_is_refused_from_a_cwd_that_lacks_its_rows_file(tmp_path, monkeypatch):
    """The bypass: stripped --selection-data, labelled set, relative rows_file not on disk here."""

    from typer.testing import CliRunner

    from meta_filter.cli import app
    from meta_filter.selection_set import SelectionSet

    rows = [dsv4pro_row(f"s{i}", i) for i in range(6)]
    stripped = write_jsonl(tmp_path / "stripped.jsonl", [strip_row(r) for r in rows])
    ids = [r["trajectory_id"] for r in rows]
    labelled_set = SelectionSet(
        row_ids=ids,
        pairs=[(ids[0], ids[1])],
        provenance={"rows_file": "data/dsv4pro_selection220.jsonl", "rows_sha256": "c3" * 32},
    ).save(tmp_path / "sel_luna.json")
    train = write_jsonl(tmp_path / "train.jsonl", [dsv4pro_row(f"t{i}", i) for i in range(4)])
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--llm-provider",
            "mock",
            "--train",
            str(train),
            "--initial-filter",
            str(REPO / BLANK_SEED),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--selection-data",
            str(stripped),
            "--selection-set",
            str(labelled_set),
        ],
    )
    assert result.exit_code != 0
    assert "cannot be opened" in " ".join(result.output.replace("│", " ").split())
    assert not (tmp_path / "runs").exists()


def _optimizer(tmp_path, *, train, config, initial=None, **kw):
    from meta_filter.optimizer import Optimizer
    from meta_filter.selector import Selector
    from meta_filter.solver import HeuristicRubricScorer, Solver
    from meta_filter.validator import Validator
    from tests.selection_gate_fixture import ScriptedProposer

    return Optimizer(
        initial_filter_dir=initial or (REPO / BLANK_SEED),
        train_data=train,
        validation_data=None,
        runs_dir=tmp_path / "runs",
        memory_dir=tmp_path / "memory",
        proposer=kw.pop("proposer", None) or ScriptedProposer({}, MockLLMClient()),
        selector=Selector(),
        solver=kw.pop("solver", None) or Solver(rubric_scorer=HeuristicRubricScorer()),
        validator=kw.pop("validator", None) or Validator(loss_weights={}),
        config=config,
        run_id=kw.pop("run_id", "mf"),
        probe_data=kw.pop("probe_data", None),
    )


def _labelled_selection(tmp_path: Path) -> tuple[Path, Path, Path]:
    from meta_filter.selection_set import SelectionSet

    train = write_jsonl(tmp_path / "train.jsonl", [dsv4pro_row(f"t{i}", i) for i in range(6)])
    rows = [dsv4pro_row(f"s{i}", i) for i in range(6)]
    selection = write_jsonl(tmp_path / "sel.jsonl", rows)
    ids = [r["trajectory_id"] for r in rows]
    prebuilt = SelectionSet(
        row_ids=ids,
        pairs=[(ids[0], ids[1]), (ids[2], ids[3]), (ids[4], ids[5])],
        tied=[(ids[1], ids[2])],
        provenance={**_provenance(selection), "judge_model": "luna"},
    )
    return train, selection, prebuilt.save(tmp_path / "sel_luna.json")


def _selection_config(selection: Path, prebuilt: Path, **kw):
    from meta_filter.optimizer import OptimizerConfig

    return OptimizerConfig(
        iterations=1,
        selection_data=selection,
        selection_set=prebuilt,
        pair_judge_cache_dir=selection.parent / "pair_cache",
        min_distinct_levels=1,
        **kw,
    )


def test_the_optimizer_refuses_a_label_visible_prebuilt_selection_set(tmp_path):
    from meta_filter.filter_loader import load_filter

    train, selection, prebuilt = _labelled_selection(tmp_path)
    optimizer = _optimizer(tmp_path, train=train, config=_selection_config(selection, prebuilt))
    with pytest.raises(RuntimeError, match="refusing selection set .*fields in view"):
        optimizer._ensure_selection_set(load_filter(REPO / BLANK_SEED))


def test_rejudge_selection_judges_the_same_pairs_on_stripped_rows(tmp_path):
    from meta_filter.filter_loader import load_filter
    from meta_filter.selection_set import SelectionSet, selection_set_path
    from meta_filter.validator import Validator
    from tests.selection_gate_fixture import Backend

    train, selection, prebuilt = _labelled_selection(tmp_path)
    judge = RecordingMock()
    optimizer = _optimizer(
        tmp_path,
        train=train,
        config=_selection_config(selection, prebuilt, rejudge_selection=True),
        validator=Validator(loss_weights={}, critic_backend=Backend(judge)),
    )
    optimizer._ensure_selection_set(load_filter(REPO / BLANK_SEED))

    saved = SelectionSet.load(selection_set_path(optimizer.store.run_dir))
    assert saved.provenance["metadata_free"] is True
    assert saved.provenance["rejudged_from"] == str(prebuilt)
    from meta_filter.metadata_free import pseudonymize_selection_set

    original = pseudonymize_selection_set(SelectionSet.load(prebuilt))
    asked = {frozenset(p) for p in [*saved.pairs, *saved.tied, *saved.unresolved]}
    assert asked == {frozenset(p) for p in [*original.pairs, *original.tied]}
    assert judge.prompts and all(not leaks(p) for p, _ in judge.prompts)


def test_build_selection_set_stamps_the_mode_and_judges_stripped_rows(tmp_path):
    from scripts.build_selection_set import build

    rows = write_jsonl(
        tmp_path / "sel.jsonl", [dsv4pro_row(f"org__repo-{i}::att0", i) for i in range(4)]
    )
    judge_client = RecordingMock()
    out = build(
        rows,
        PairJudge(client=judge_client, cache_dir=tmp_path / "pairs"),
        pairs="all",
        seed=1,
        exclude=[],
        out=tmp_path / "set.json",
    )
    assert out.provenance["metadata_free"] is True
    assert selection_set_view(out.provenance) is None
    assert all(i.startswith("traj_") for i in out.row_ids), "ids are pseudonymized"
    assert all(not leaks(p) for p, _ in judge_client.prompts)


# --- the four configs -----------------------------------------------------------------


def _cells() -> dict[str, dict]:
    from meta_filter.cli import _load_config

    return {c: _load_config(REPO / "configs/metadata_free" / f"{c}.yaml") for c in CELLS}


# Keys that are allowed to differ between cells; everything else is shared.
_PER_CELL = {
    ("launch", "seeded_pools"),
    ("paths", "train_data"),
    ("paths", "initial_filter"),
    ("paths", "memory_dir"),
    ("launch", "cell"),
    ("launch", "pool"),
}


def _flatten(cfg: dict, prefix: tuple = ()) -> dict[tuple, object]:
    out: dict[tuple, object] = {}
    for key, value in cfg.items():
        path = (*prefix, key)
        if isinstance(value, dict) and path not in _PER_CELL:
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def test_every_cell_loads():
    for cell, cfg in _cells().items():
        assert cfg["launch"]["cell"] == cell
        assert cfg["llm"] == {
            "provider": "codex",
            "model": "openai/gpt-5.6-sol-pro",
            "temperature": 0.4,
        }
        assert (cfg["seed"], cfg["iterations"], cfg["candidates_per_iteration"]) == (29, 20, 3)
        assert (REPO / cfg["paths"]["initial_filter"]).is_dir()


def test_the_cells_differ_only_in_pool_and_seed():
    flat = {cell: _flatten(cfg) for cell, cfg in _cells().items()}
    reference = flat[CELLS[0]]
    for cell, values in flat.items():
        assert values.keys() - _PER_CELL == reference.keys() - _PER_CELL, cell
        shared = {k: v for k, v in values.items() if k not in _PER_CELL}
        assert shared == {k: v for k, v in reference.items() if k not in _PER_CELL}, cell


def test_the_cell_grid_is_pool_by_seed():
    cells = _cells()
    for cell, cfg in cells.items():
        pool, seed = cell.split("_")
        train = cfg["paths"]["train_data"]
        sel = cfg["launch"]["pool"]
        if pool == "dsv4pro":
            assert train == "data/dsv4pro_train50_s29.jsonl"
            assert sel["selection_data"] == "data/dsv4pro_selection220.jsonl"
            assert sel["selection_set"] == "data/dsv4pro_selection220_judged.json"
        else:
            assert train == "data/nemotron_train50.jsonl"
            assert sel["selection_data"] == "data/nemotron_selection220.jsonl"
            assert sel["selection_set"] == "data/nemotron_selection220_judged.json"
        expected = BLANK_SEED if seed == "blank" else NONBLANK_SEED
        assert cfg["paths"]["initial_filter"] == expected


def test_the_shared_flags_are_the_production_settings():
    args = [part for flag in _cells()[CELLS[0]]["launch"]["args"] for part in flag]
    pairs = dict(zip(args, [*args[1:], None], strict=True))
    assert pairs["--pair-budget"] == "50"
    assert pairs["--solver-model"] == pairs["--judge-model"] == "openai/gpt-5.6-luna"
    assert "--score-weight" not in args, "lambda_complexity stays at its 0.15 default"
    assert "--exclude-consensus" in args and "--complexity-ceiling" in args


# --- the launcher ----------------------------------------------------------------------


def _run_options() -> set[str]:
    import typer.main

    from meta_filter.cli import app

    command = typer.main.get_command(app).commands["run"]
    return {opt for param in command.params for opt in (*param.opts, *param.secondary_opts)}


def test_the_launcher_dry_run_prints_the_command_and_calls_nothing(monkeypatch, capsys):
    import subprocess

    import scripts.run_metadata_free as launcher

    def _forbidden(*_a, **_k):
        raise AssertionError("a dry run must not launch or read the key")

    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    monkeypatch.setattr(launcher, "read_env_key", _forbidden)

    assert launcher.main(["--cell", "nemotron_blank", "--seed", "29", "--dry-run"]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("uv run"))
    argv = shlex.split(line)
    assert argv[:6] == ["uv", "run", "python", "-m", "meta_filter.cli", "run"]
    assert argv[argv.index("--config") + 1] == "configs/metadata_free/nemotron_blank.yaml"
    assert argv[argv.index("--selection-set") + 1] == "data/nemotron_selection220_judged.json"
    assert "<openrouter-key>" in out
    known = _run_options()
    assert all(tok in known for tok in argv[6:] if tok.startswith("--")), "unknown CLI flag"


def test_every_cell_dry_runs(capsys):
    import scripts.run_metadata_free as launcher

    assert launcher.main(["--all", "--dry-run", "--date", "20000101"]) == 0
    out = capsys.readouterr().out
    for cell in CELLS:
        assert f"--run-id run_20000101_metafree_{cell}_s29" in out


def test_a_seed_moves_the_dsv4pro_pools_and_not_the_nemotron_pool(tmp_path, monkeypatch):
    import scripts.run_metadata_free as launcher

    cmd = launcher.build_command("dsv4pro_blank", seed=17, date_tag="20000101")
    assert cmd[cmd.index("--seed") + 1] == "17"
    assert cmd[cmd.index("--memory-dir") + 1] == "memory/dsv4pro_blank_s17"
    assert "run_20000101_metafree_dsv4pro_blank_s17" in cmd
    assert cmd[cmd.index("--train") + 1] == "data/dsv4pro_train50_s17.jsonl"
    assert cmd[cmd.index("--probe") + 1] == "data/dsv4pro_probe25_s17.jsonl"
    at29 = launcher.build_command("dsv4pro_blank", date_tag="20000101")
    assert at29[at29.index("--train") + 1] == _cells()["dsv4pro_blank"]["paths"]["train_data"]

    nemo = launcher.build_command("nemotron_blank", seed=17, date_tag="20000101")
    assert nemo[nemo.index("--train") + 1] == "data/nemotron_train50.jsonl"
    assert nemo[nemo.index("--probe") + 1] == "data/nemotron_probe25.jsonl"

    # A seed with no pool on disk is refused by name; one that has them passes.
    with pytest.raises(SystemExit, match="no pool for seed 17"):
        launcher.check_seeded_pools("dsv4pro_blank", 17, root=tmp_path)
    (tmp_path / "data").mkdir()
    for name in ("dsv4pro_train50_s17.jsonl", "dsv4pro_probe25_s17.jsonl"):
        (tmp_path / "data" / name).write_text("{}\n", encoding="utf-8")
    launcher.check_seeded_pools("dsv4pro_blank", 17, root=tmp_path)
    launcher.check_seeded_pools("nemotron_blank", 17, root=tmp_path)
    shutil.copytree(REPO / "configs/metadata_free", tmp_path / "configs/metadata_free")
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    assert launcher.main(["--cell", "dsv4pro_blank", "--seed", "17", "--dry-run"]) == 0
    for name in ("dsv4pro_train50_s17.jsonl", "dsv4pro_probe25_s17.jsonl"):
        (tmp_path / "data" / name).unlink()
    with pytest.raises(SystemExit, match="no pool for seed 17"):
        launcher.main(["--cell", "dsv4pro_blank", "--seed", "17", "--dry-run"])


# --- mock end to end --------------------------------------------------------------------


@dataclass
class RecordingMock(MockLLMClient):
    """The deterministic mock, keeping every (prompt, system) it was sent."""

    prompts: list = field(default_factory=list)

    def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        self.prompts.append((prompt, system or ""))
        return super().generate(
            prompt, system=system, temperature=temperature, cache_system=cache_system
        )


def _pools(tmp_path: Path) -> dict[str, Path]:
    return {
        name: write_jsonl(
            tmp_path / f"{name}.jsonl",
            [dsv4pro_row(f"{name}-row-{i:03d}::att0", i) for i in range(n)],
        )
        for name, n in (("train", 12), ("probe", 4), ("selection", 10))
    }


def _scan_run_dir(run_dir: Path) -> dict[str, list[str]]:
    """Every file a tool-using proposer could open, scanned like a prompt."""

    out = {}
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.name == "_RUN_COMPLETE.json":
            continue  # its `exit_status` is the process's, not the harness's
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = leaks(text)
        if hits:
            out[str(path.relative_to(run_dir))] = hits
    return out


def _mock_cli_run(tmp_path, monkeypatch):
    """A two-iteration mock run through the CLI; returns what the LLMs saw."""

    from typer.testing import CliRunner

    from meta_filter.cli import app

    sent: list[tuple[str, str]] = []
    original = MockLLMClient.generate

    def recording(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        sent.append((prompt, system or ""))
        return original(self, prompt, system=system, temperature=temperature)

    monkeypatch.setattr(MockLLMClient, "generate", recording)
    pools = _pools(tmp_path)
    config = tmp_path / "cell.yaml"
    config.write_text(yaml.safe_dump({"seed": 17}), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--llm-provider",
            "mock",
            "--solver-backend",
            "llm",
            "--solver-provider",
            "mock",
            "--solver-cache-dir",
            str(tmp_path / "scorer_cache"),
            "--judge-provider",
            "mock",
            "--judge-cache-dir",
            str(tmp_path / "judge_cache"),
            "--pair-judge-cache-dir",
            str(tmp_path / "pair_cache"),
            "--train",
            str(pools["train"]),
            "--probe",
            str(pools["probe"]),
            "--selection-data",
            str(pools["selection"]),
            "--selection-pairs",
            "20",
            "--initial-filter",
            str(REPO / NONBLANK_SEED),
            "--iterations",
            "2",
            "--candidates",
            "2",
            "--pair-budget",
            "8",
            "--max-pairs",
            "60",
            "--patience",
            "5",
            "--min-iterations",
            "1",
            "--complexity-ceiling",
            "rubric_criteria=9,rule_count=13",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--memory-dir",
            str(tmp_path / "memory"),
            "--run-id",
            "mf",
        ],
    )
    assert result.exit_code == 0, result.output
    return sent, pools, tmp_path / "runs" / "mf"


def test_a_mock_cli_run_shows_no_llm_an_outcome_field(tmp_path, monkeypatch):
    sent, pools, run_dir = _mock_cli_run(tmp_path, monkeypatch)

    kinds = {
        "pair judge": [p for p, _ in sent if "=== TRAJECTORY A (" in p],
        "scorer": [p for p, _ in sent if "### RUBRIC (verbatim)" in p],
        "proposer": [p for p, _ in sent if "META_FILTER::PROPOSER" in p],
    }
    assert all(kinds.values()), {k: len(v) for k, v in kinds.items()}
    for prompt, system in sent:
        assert "::att" not in prompt, "an instance-bearing id reached an LLM"
        assert not leaks(prompt), leaks(prompt)
        assert not leaks(system), leaks(system)
    assert _scan_run_dir(run_dir) == {}
    assert not (run_dir / "reference").exists(), "the outcome-filtering notes stay out"
    rankings = list(run_dir.glob("iterations/*/ranking.json"))
    assert rankings
    for ranking in rankings:
        assert "resolved_mean" not in ranking.read_text(encoding="utf-8"), ranking
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["train_data"] == str(run_dir / METADATA_FREE_TRAIN_FILE)
    from meta_filter.metadata_free import file_sha256

    assert cfg["train_data_source_sha256"] == file_sha256(pools["train"])
    assert str(pools["train"]) not in (run_dir / "config.yaml").read_text(encoding="utf-8")
    record = json.loads((tmp_path / "runs/_metadata_free/mf.json").read_text(encoding="utf-8"))
    assert record["sources"]["train"]["path"] == str(pools["train"])
    assert cfg["complexity_ceiling"] == {"rubric_criteria": 9, "rule_count": 13}


def test_the_scan_has_teeth():
    """Negative control: the raw rows the runs are fed do carry what the scans look for."""

    leaked = set(leaks(json.dumps(dsv4pro_row())))
    assert {"resolved", "f2p_pass", "exit_status", SENTINEL} <= leaked


def test_the_cli_refuses_a_label_visible_selection_set_before_building_anything(tmp_path):
    from typer.testing import CliRunner

    from meta_filter.cli import app

    train, selection, prebuilt = _labelled_selection(tmp_path)
    base = [
        "run",
        "--llm-provider",
        "mock",
        "--train",
        str(train),
        "--initial-filter",
        str(REPO / BLANK_SEED),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--memory-dir",
        str(tmp_path / "memory"),
        "--selection-data",
        str(selection),
        "--selection-set",
        str(prebuilt),
    ]
    refused = CliRunner().invoke(app, base)
    assert refused.exit_code != 0
    assert "fields in view" in " ".join(refused.output.replace("│", " ").split())
    assert not (tmp_path / "runs").exists()


def _copy_baseline_writer(run_dir: Path):
    """What the scripted tool-using proposer does: copy the baseline, edit a line."""

    def write(prompt: str) -> None:
        match = re.search(r"iterations/(\d{3})/candidates", prompt)
        if not match:
            return
        iteration = int(match.group(1))
        root = run_dir / "iterations" / f"{iteration:03d}" / "candidates"
        base = root / f"iter{iteration:03d}_baseline"
        cand = root / f"iter{iteration:03d}_00_reword"
        if not base.is_dir() or cand.exists():
            return
        shutil.copytree(base, cand, ignore=shutil.ignore_patterns("__pycache__", "*.json"))
        rubric = cand / "rubric.md"
        rubric.write_text(rubric.read_text(encoding="utf-8") + "\nCite one line.\n", "utf-8")
        (cand / "rationale.md").write_text("Hypothesis: a citation sharpens scores.\n", "utf-8")
        (root / "diagnosis.json").write_text(
            json.dumps(
                {
                    "diagnosis": "scores are vague",
                    "candidates": [{"name": "reword", "hypothesis": "cite", "notes": "rubric"}],
                }
            ),
            encoding="utf-8",
        )

    return write


@dataclass
class _ScriptedAgent(ClaudeCodeHeadlessClient):
    """Stands in for the tool-using harness; every copy shares `calls`."""

    calls: list = field(default_factory=list)
    writer: object = None
    # Paths the harness audit reports as read outside the workspace.
    reads: tuple = ()

    def generate(self, prompt, *, system=None, temperature=0.0, cache_system=False):
        from meta_filter.llm.codex_cli import CodexResponse

        self.calls.append((prompt, system or ""))
        if self.writer is not None:
            self.writer(prompt)
        return CodexResponse(
            text="done",
            prompt_tokens=1,
            completion_tokens=1,
            model="opus",
            reads_outside_workspace=tuple(self.reads) if "diagnosis.json" in prompt else (),
        )


def _agentic_optimizer(tmp_path, *, reads=(), iterations=1, run_id="mfa"):
    from meta_filter.agentic_proposer import AgenticProposer
    from meta_filter.critics.trajectory_critic import TrajectoryCritic
    from meta_filter.llm_scorer import LLMRubricScorer
    from meta_filter.optimizer import OptimizerConfig
    from meta_filter.solver import Solver
    from meta_filter.validator import Validator

    pools = _pools(tmp_path)
    run_dir = tmp_path / "runs" / run_id
    agent = _ScriptedAgent(
        model="opus",
        scratch_dir=tmp_path / "scratch",
        writer=_copy_baseline_writer(run_dir),
        reads=tuple(reads),
    )
    scorer_client, judge_client = RecordingMock(), RecordingMock()
    optimizer = _optimizer(
        tmp_path,
        train=pools["train"],
        probe_data=pools["probe"],
        initial=REPO / BLANK_SEED,
        run_id=run_id,
        proposer=AgenticProposer(client=agent, n_candidates=1),
        solver=Solver(
            rubric_scorer=LLMRubricScorer(client=scorer_client, cache_dir=tmp_path / "sc")
        ),
        validator=Validator(
            loss_weights={},
            critic_backend=TrajectoryCritic(client=judge_client, cache_dir=tmp_path / "jc"),
        ),
        config=OptimizerConfig(
            iterations=iterations,
            candidates_per_iteration=1,
            pair_budget=8,
            max_pairs=60,
            min_distinct_levels=1,
            pair_judge_cache_dir=tmp_path / "pc",
            selection_data=pools["selection"],
            selection_pairs=20,
            patience=5,
            min_iterations=1,
            agentic_critics=True,
        ),
    )
    return optimizer, agent, scorer_client, judge_client, pools, run_dir


def test_a_mock_run_with_the_tool_using_proposer_shows_no_llm_an_outcome_field(tmp_path):
    optimizer, agent, scorer_client, judge_client, _, run_dir = _agentic_optimizer(tmp_path)
    optimizer.run()

    proposer_legs = [p for p, _ in agent.calls if "diagnosis.json" in p]
    assert proposer_legs, "the tool-using proposer was never asked"
    assert any("This run carries no outcome metadata." in p for p in proposer_legs)
    assert (run_dir / "iterations/000/candidates/iter000_00_reword").is_dir()
    everything = [*agent.calls, *scorer_client.prompts, *judge_client.prompts]
    assert scorer_client.prompts and judge_client.prompts
    for prompt, system in everything:
        assert "::att" not in prompt, "an instance-bearing id reached an LLM"
        assert not leaks(prompt), leaks(prompt)
        assert not leaks(system), leaks(system)
    assert _scan_run_dir(run_dir) == {}


def test_the_rule_tool_checks_rules_on_the_stripped_rows(tmp_path):
    """A rule on an outcome field fires on the raw pool and never on this run's view."""

    from meta_filter.rule_tool import RuleRequest, check_rules

    root = tmp_path / "repo"
    run_dir = root / "runs" / "run_x"
    (root / "data").mkdir(parents=True)
    write_jsonl(root / "data" / "train.jsonl", [dsv4pro_row(f"t{i}", i) for i in range(6)])
    (run_dir / "iterations" / "000").mkdir(parents=True)
    rules = tmp_path / "rules.py"
    rules.write_text(
        "def rule_outcome(trajectory):\n"
        "    if trajectory.metadata.get('exit_status'):\n"
        "        return {'name': 'outcome', 'verdict': 'REJECT', 'evidence': ['x']}\n"
        "    return None\n",
        encoding="utf-8",
    )

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"train_data": "data/train.jsonl"}), encoding="utf-8"
    )
    payload = check_rules(
        RuleRequest(run_dir=run_dir, rules=rules, repo_root=root, sample=6)
    ).payload
    fires = next(r for r in payload["rule_results"] if r["name"] == "rule_outcome")["fires"]
    assert fires == 0
    # Negative control: the same rule fires on every raw row.
    from meta_filter.rule_tool import load_rule_functions

    (rule,) = load_rule_functions(rules)
    raw = [_traj(dsv4pro_row(f"t{i}", i)) for i in range(6)]
    assert all(rule(t) is not None for t in raw)


def test_build_ranking_subset_names_its_pool_and_superset_explicitly():
    from scripts.build_ranking_subset import build_parser

    base = ["--filter-dir", NONBLANK_SEED, "--name", "n", "--readme", "r", "--note", "x"]
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "--out", "o"])
    args = build_parser().parse_args(
        [*base, "--out", "o", "--pool", "p.jsonl", "--superset-dir", "s"]
    )
    assert args.eligibility_ledger is None


def test_build_selection_set_can_rejudge_an_existing_sets_pairs(tmp_path):
    from meta_filter.selection_set import SelectionSet
    from scripts.build_selection_set import build

    _, selection, prebuilt = _labelled_selection(tmp_path)
    out = build(
        selection,
        PairJudge(client=RecordingMock(), cache_dir=tmp_path / "pairs2"),
        pairs="all",
        seed=1,
        exclude=[],
        out=tmp_path / "nometa.json",
        pairs_from=prebuilt,
    )
    from meta_filter.metadata_free import pseudonymize_selection_set

    source = pseudonymize_selection_set(SelectionSet.load(prebuilt))
    asked = {frozenset(p) for p in [*out.pairs, *out.tied, *out.unresolved]}
    assert asked == {frozenset(p) for p in [*source.pairs, *source.tied]}
    assert out.provenance["metadata_free"] is True
    assert out.provenance["pairs_from"] == str(prebuilt)


# --- the run hard cap ----------------------------------------------------------------------

# The reference filter the cells' cap was derived from: its
# `measure_filter_complexity` breakdown. Hard-coded so this test needs no copy
# of that filter; the cap is floor(1.5 x each non-exempt component).
_REFERENCE_BREAKDOWN = {
    "rubric_criteria": 6,
    "rule_count": 9,
    "rules_loc": 348,
    "policy_loc": 74,
    "rubric_chars": 5857,
}
CELL_CAP = "rubric_criteria=9,rule_count=13,rules_loc=522,rubric_chars=8785"


def test_the_cells_cap_equals_one_and_a_half_times_the_reference_breakdown():
    import math

    from meta_filter.constraints import ConstraintLimits, parse_complexity_ceiling

    limits = ConstraintLimits.DEFAULT
    expected = {
        name: math.floor(limits.hard_ratio * size)
        for name, size in _REFERENCE_BREAKDOWN.items()
        if name not in limits.hard_exempt
    }
    assert parse_complexity_ceiling(CELL_CAP) == expected
    for cfg in _cells().values():
        args = [part for flag in cfg["launch"]["args"] for part in flag]
        assert args[args.index("--complexity-ceiling") + 1] == CELL_CAP


def test_the_numeric_cap_refuses_exactly_what_the_multiple_refused(tmp_path):
    """For every count around each cap, `value > cap` == `value > 1.5 x size`."""

    import math

    from meta_filter.constraints import ConstraintLimits, parse_complexity_ceiling

    caps = parse_complexity_ceiling(CELL_CAP)
    ratio = ConstraintLimits.DEFAULT.hard_ratio
    for name, cap in caps.items():
        size = _REFERENCE_BREAKDOWN[name]
        for value in range(math.floor(ratio * size) - 3, math.floor(ratio * size) + 4):
            assert (value > cap) == (value > ratio * size), (name, value)


def test_check_candidate_applies_the_numeric_cap(tmp_path):
    from meta_filter.constraints import check_candidate

    initial = _filter(tmp_path / "initial")
    rubric = "# Rubric\n" + "".join(f"\n## Criterion: C{i}\n+2 good.\n" for i in range(3))
    policy = (
        "RUBRIC_CRITERIA = ['C0', 'C1', 'C2']\n\n\ndef decide(trajectory, scores, triggered):\n"
        "    return {'decision': 'REVIEW', 'score': sum(scores.values()), 'confidence': 0.5}\n"
    )
    cand = _filter(tmp_path / "cand", rubric=rubric, policy=policy)

    def invalid(caps):
        report = check_candidate(cand, baseline_dir=cand, initial_dir=initial, hard_caps=caps)
        return [r for r in report.reasons if "(hard)" in r]

    assert invalid({"rubric_criteria": 2}) == ["rubric_criteria 3 > run cap 2 (hard)"]
    assert invalid({"rubric_criteria": 3}) == []


@pytest.mark.parametrize(
    "bad", ["rubric_criteria", "policy_loc=5", "rules_loc=x", "nope=3", "rule_count=-1"]
)
def test_a_malformed_cap_is_refused(bad):
    from meta_filter.constraints import parse_complexity_ceiling

    with pytest.raises(ValueError):
        parse_complexity_ceiling(bad)


def test_a_read_outside_the_workspace_invalidates_the_proposal(tmp_path):
    import sys

    raw_pool = tmp_path / "train.jsonl"
    optimizer, _, _, _, _, run_dir = _agentic_optimizer(
        tmp_path, reads=(str(raw_pool), sys.executable)
    )
    # Like any iteration whose proposer failed, a final one ends the run
    # without recording it complete.
    with pytest.raises(RuntimeError, match="must not be recorded as complete"):
        optimizer.run()

    candidate = "iter000_00_reword"
    assert not (run_dir / "iterations/000/candidates" / candidate).exists()
    moved = tmp_path / "runs/_invalidated/mfa/000"
    assert (moved / candidate).is_dir()
    assert json.loads((moved / "reads.json").read_text())["reads"] == [str(raw_pool)]
    note = json.loads((run_dir / "iterations/000/proposer/invalidated.json").read_text())
    assert note["invalidated"] is True and note["n_reads"] == 1
    assert str(raw_pool) not in json.dumps(note), "the workspace is not told where the rows are"
    assert not (run_dir / "iterations/000/selection.json").exists(), "nothing was scored"


def test_repeated_outside_reads_end_the_run(tmp_path):
    optimizer, *_ = _agentic_optimizer(
        tmp_path, reads=(str(tmp_path / "train.jsonl"),), iterations=3
    )
    with pytest.raises(RuntimeError, match="outside its workspace 3 times in a row"):
        optimizer.run()


def test_the_interpreter_is_the_only_outside_read_allowed(tmp_path):
    import sys

    from meta_filter.llm.codex_cli import CodexResponse
    from meta_filter.optimizer import MetadataFreeReadError

    optimizer, *_ = _agentic_optimizer(tmp_path)
    ok = CodexResponse(text="", reads_outside_workspace=(sys.executable,))
    optimizer._assert_no_forbidden_reads(ok, iteration=0, leg="propose")
    for path in (tmp_path / "train.jsonl", REPO / "README.md", REPO / "data"):
        bad = CodexResponse(text="", reads_outside_workspace=(str(path),))
        with pytest.raises(MetadataFreeReadError):
            optimizer._assert_no_forbidden_reads(bad, iteration=0, leg="propose")


# --- instance-free ids ----------------------------------------------------------------------


def test_instance_bearing_ids_are_pseudonymized_and_nemotron_ids_are_not(tmp_path):
    import hashlib

    from meta_filter.metadata_free import PSEUDONYM_SALT, pseudonym_map, pseudonymize_id

    raw = "owner__repo-12::att2"
    expected = "traj_" + hashlib.sha256((PSEUDONYM_SALT + raw).encode()).hexdigest()[:16]
    assert pseudonymize_id(raw) == expected
    assert pseudonymize_id(expected) == expected, "idempotent"
    nemo = "nemotron.00000000-0000-4000-8000-000000000002"
    assert pseudonymize_id(nemo) == nemo
    assert strip_row(dsv4pro_row(raw))["trajectory_id"] == expected
    assert strip_row(nemotron_row(nemo))["trajectory_id"] == nemo

    pool = write_jsonl(tmp_path / "p.jsonl", [dsv4pro_row(raw), nemotron_row(nemo)])
    assert pseudonym_map(pool) == {expected: raw, nemo: nemo}
    loaded = load_trajectories(pool)
    assert [t.trajectory_id for t in loaded] == [expected, nemo]


def test_the_stripped_train_file_carries_no_instance(tmp_path, monkeypatch):
    _, _, run_dir = _mock_cli_run(tmp_path, monkeypatch)
    text = (run_dir / METADATA_FREE_TRAIN_FILE).read_text(encoding="utf-8")
    assert "::att" not in text and "train-row-" not in text
    assert all(json.loads(line)["trajectory_id"].startswith("traj_") for line in text.splitlines())


# --- SWE-bench test-list names ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["FAIL_TO_PASS", "PASS_TO_PASS", "fail_to_pass", "pass_to_pass"])
def test_swe_bench_test_lists_are_outcome_reads(tmp_path, name):
    rules = f"def rule_x(t):\n    return t.metadata.get('{name}')\n"
    initial = _filter(tmp_path / "initial")
    candidate = _filter(tmp_path / "cand", rules=rules)
    report = _check(candidate, initial)
    assert report.status == "invalid" and any(name in r for r in report.reasons)


def test_the_test_list_names_do_not_change_the_nemotron_row(tmp_path):
    row = {**nemotron_row(), "metadata": {**nemotron_row()["metadata"], "FAIL_TO_PASS": "[x]"}}
    assert "FAIL_TO_PASS" not in strip_row(row)["metadata"]
    assert strip_row(nemotron_row()) == nemotron_row()
