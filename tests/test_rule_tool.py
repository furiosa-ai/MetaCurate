"""The proposer's rule-checking tool (`meta_filter.rule_tool`).

A rule is pure Python over the trajectory, so a draft can be measured before
an iteration is spent on it — and until this tool existed, nothing did. The
prompt already warned the proposer that "a rule that reads a field these rows
do not carry returns None on every row and NOTHING in this run will tell
you"; on the nemotron exam the seed's only rule fired 0/220 and two evolved
rules fired 213/220 and 214/220. What has to hold, and is pinned here:

* the two degenerate shapes are named in words, not left to be read off a
  ratio: a rule that never fires is DEAD, one that fires on (nearly) every
  row is SATURATED;
* a rule that raises is reported as ERROR with the exception type, and does
  not cost the rules beside it their report;
* the rule's own verdict is credited against the judged pairs among the
  sampled rows, restricted to the pairs it actually splits;
* the train boundary is `score_tool`'s boundary, in `score_tool`'s words, so
  the hidden selection rows are unreachable through this tool too.

Every fixture is built for real in `tmp_path` — pools, run directory, rules
files. There is nothing to mock: the tool makes no LLM call at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from meta_filter.rule_tool import (
    DEFAULT_SAMPLE,
    RuleRequest,
    build_parser,
    check_rules,
    main,
)

# --- fixtures: a run small enough to state every count in one number -------

DEAD_RULE = '''
def rule_never(trajectory):
    """Reads a field these rows do not carry."""

    if trajectory.metadata.get("submission_bytes"):
        return {"name": "never", "verdict": "REJECT", "evidence": ["huge patch"]}
    return None
'''

SATURATED_RULE = '''
def rule_always(trajectory):
    """Fires on anything with a turn, which is everything."""

    if trajectory.turns:
        return {
            "name": "always",
            "verdict": "ACCEPT",
            "evidence": [f"{len(trajectory.turns)} turn(s)"],
        }
    return None
'''

RAISING_RULE = """
def rule_boom(trajectory):
    raise KeyError("model_patch")
"""

# Fires on the rows whose task says `weak`, which is exactly the `lo` side of
# every judged pair below: read as an ordering, REJECT ranks those lower.
DISCRIMINATING_RULE = """
def rule_weak(trajectory):
    if "weak" in trajectory.task:
        return {"name": "weak", "verdict": "REJECT", "evidence": ["task looks weak"]}
    return None
"""


def _pool(path: Path, ids: list[str], *, weak: set[str] = frozenset()) -> None:
    """A JSONL pool of rows that carry turns and nothing exotic."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "trajectory_id": tid,
            "task": f"fix something ({'weak' if tid in weak else 'solid'})",
            "turns": [{"role": "assistant", "action": "edit", "content": "patch"}],
            "metadata": {"source": "test"},
        }
        for tid in ids
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _write_rules(root: Path, name: str, *bodies: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("from __future__ import annotations\n" + "".join(bodies), encoding="utf-8")
    return path


@pytest.fixture()
def run(tmp_path) -> Path:
    """A run directory shaped like the optimizer leaves it, with no optimizer.

    Six train rows, three of them `weak`; four selection rows the tool must
    refuse; three judged pairs among the train rows, each one `solid` over
    `weak`, so a rule firing on the weak side alone splits all three.
    """

    root = tmp_path / "repo"
    run_dir = root / "runs" / "run_x"
    train_ids = [f"train_{i}" for i in range(6)]
    weak = {"train_1", "train_3", "train_5"}
    _pool(root / "data" / "train.jsonl", train_ids, weak=weak)
    _pool(root / "data" / "selection.jsonl", [f"sel_{i}" for i in range(4)])
    pairs = [
        {"hi": "train_0", "lo": "train_1", "tie": False},
        {"hi": "train_2", "lo": "train_3", "tie": False},
        {"hi": "train_4", "lo": "train_5", "tie": False},
        # A tie is not an ordering, so it must not be counted either way.
        {"hi": "train_0", "lo": "train_2", "tie": True},
    ]
    (run_dir / "iterations" / "000").mkdir(parents=True)
    (run_dir / "iterations" / "000" / "ranking_pairs.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "train_data": "data/train.jsonl",
                "selection_data": "data/selection.jsonl",
                "selection_set": "data/selection_set.json",
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _check(run: Path, rules: Path, **kwargs) -> tuple[str, dict]:
    report = check_rules(
        RuleRequest(run_dir=run, rules=rules, repo_root=run.parent.parent, **kwargs)
    )
    return report.text, report.payload


def _result(payload: dict, name: str) -> dict:
    return next(r for r in payload["rule_results"] if r["name"] == name)


# --- what the report says about a rule -------------------------------------


def test_rule_that_never_fires_is_dead(run, tmp_path):
    rules = _write_rules(tmp_path, "dead/rules.py", DEAD_RULE)
    text, payload = _check(run, rules)

    result = _result(payload, "rule_never")
    assert result["status"] == "DEAD"
    assert result["fires"] == 0
    assert "0/6 — DEAD: this rule never fires on these rows" in text
    assert payload["summary"] == {"rules": 1, "dead": 1, "saturated": 0, "raised": 0}


def test_rule_that_fires_on_every_row_is_saturated(run, tmp_path):
    rules = _write_rules(tmp_path, "sat/rules.py", SATURATED_RULE)
    text, payload = _check(run, rules)

    result = _result(payload, "rule_always")
    assert result["status"] == "SATURATED"
    assert result["fires"] == 6
    assert "6/6 — SATURATED: fires on nearly every row, so it orders almost nothing" in text
    # The evidence the rule itself emitted is quoted back, up to five rows.
    assert len(result["fired_rows"]) == 5
    assert result["fired_rows"][0]["evidence"] == ["1 turn(s)"]


def test_a_raising_rule_is_an_error_and_does_not_stop_the_others(run, tmp_path):
    rules = _write_rules(tmp_path, "boom/rules.py", RAISING_RULE, SATURATED_RULE, DEAD_RULE)
    text, payload = _check(run, rules)

    boom = _result(payload, "rule_boom")
    assert boom["status"] == "ERROR"
    assert boom["error_types"] == ["KeyError"]
    assert boom["raised"] == 6
    # An ERROR is not a DEAD: it fired on nothing, but for a reason that wants
    # the opposite fix, and the summary counts it as raised rather than dead.
    assert "ERROR — raised KeyError on all 6 row(s)" in text
    assert payload["summary"] == {"rules": 3, "dead": 1, "saturated": 1, "raised": 1}
    assert _result(payload, "rule_always")["status"] == "SATURATED"
    assert _result(payload, "rule_never")["status"] == "DEAD"


def test_a_discriminating_rule_reports_its_pair_credit(run, tmp_path):
    rules = _write_rules(tmp_path, "weak/rules.py", DISCRIMINATING_RULE)
    text, payload = _check(run, rules)

    result = _result(payload, "rule_weak")
    assert result["status"] == "OK"
    assert result["fires"] == 3
    # Three judged pairs, each split by the rule (one side weak, one solid);
    # REJECT on the weak side is the judge's own order every time. The tie is
    # not among them.
    assert result["pairs_split_by_rule"] == 3
    assert result["ordered_the_judges_way"] == 3
    assert "ordered the judge's way: 3/3 (100%)" in text


def test_pair_credit_is_absent_rather_than_faked_when_no_pair_is_split(run, tmp_path):
    rules = _write_rules(tmp_path, "sat2/rules.py", SATURATED_RULE)
    text, payload = _check(run, rules)

    # Every row fires, so no judged pair has exactly one side firing: there is
    # no denominator, and the report says so instead of printing a percentage.
    assert _result(payload, "rule_always")["pairs_split_by_rule"] == 0
    assert "no judged pair among these rows has exactly one side firing" in text
    assert "%" not in text


def test_a_rule_carrying_no_ordering_is_called_out_at_half(run, tmp_path):
    """A rule firing on one row of each judged pair, half of them the wrong way."""

    rules = _write_rules(
        tmp_path,
        "half/rules.py",
        """
def rule_half(trajectory):
    if trajectory.trajectory_id in ("train_1", "train_2", "train_4", "train_5"):
        return {"name": "half", "verdict": "REJECT", "evidence": ["by id"]}
    return None
""",
    )
    text, payload = _check(run, rules)

    result = _result(payload, "rule_half")
    # train_0>train_1 and train_2>train_3 and train_4>train_5: the rule fires
    # on the lo side of the first, the hi side of the second, and on both of
    # the third (which therefore drops).
    assert result["pairs_split_by_rule"] == 2
    assert result["ordered_the_judges_way"] == 1
    assert "ordered the judge's way: 1/2 (50%)" in text
    assert "near 50% means the rule is not carrying ordering information" in text


# --- the boundary and the caps ---------------------------------------------


def test_a_selection_row_is_refused_in_score_tools_words(run, tmp_path):
    rules = _write_rules(tmp_path, "b/rules.py", SATURATED_RULE)
    report = check_rules(
        RuleRequest(
            run_dir=run, rules=rules, rows=["train_0", "sel_2"], repo_root=run.parent.parent
        )
    )

    assert report.exit_code == 2
    assert report.payload["refused"] == (
        "refused: 1 id(s) are not in this run's train pool (sel_2); "
        "only train rows can be scored here"
    )


def test_an_unrecognised_id_is_refused_as_a_typo(run, tmp_path):
    """With a pseudonym map the tool can tell a typo from the boundary."""

    idmap = run.parent / "_idmap" / f"{run.name}.json"
    idmap.parent.mkdir(parents=True, exist_ok=True)
    names = {f"train_{i}": f"traj_t{i}" for i in range(6)}
    names.update({f"sel_{i}": f"traj_s{i}" for i in range(4)})
    idmap.write_text(
        json.dumps(
            {
                "pseudonym_to_id": {v: k for k, v in names.items()},
                "id_to_pseudonym": names,
            }
        ),
        encoding="utf-8",
    )
    rules = _write_rules(tmp_path, "c/rules.py", SATURATED_RULE)

    typo = check_rules(
        RuleRequest(run_dir=run, rules=rules, rows=["traj_nope"], repo_root=run.parent.parent)
    )
    assert typo.exit_code == 2
    assert "unrecognised id(s) (traj_nope)" in typo.payload["refused"]

    # The hidden rows stay hidden under the same map, and a train pseudonym
    # works and is reported back in the spelling the agent used.
    hidden = check_rules(
        RuleRequest(run_dir=run, rules=rules, rows=["traj_s1"], repo_root=run.parent.parent)
    )
    assert "not in this run's train pool" in hidden.payload["refused"]

    ok = check_rules(
        RuleRequest(run_dir=run, rules=rules, rows=["traj_t0"], repo_root=run.parent.parent)
    )
    assert ok.exit_code == 0
    assert ok.payload["rows"] == ["traj_t0"]


def test_more_than_twenty_rows_is_refused(run, tmp_path):
    rules = _write_rules(tmp_path, "d/rules.py", SATURATED_RULE)
    report = check_rules(
        RuleRequest(
            run_dir=run,
            rules=rules,
            rows=[f"train_{i}" for i in range(21)],
            repo_root=run.parent.parent,
        )
    )

    assert report.exit_code == 2
    assert report.payload["refused"] == "refused: 21 rows requested; at most 20 per call"


def test_a_directory_that_is_not_a_run_is_refused(tmp_path):
    report = check_rules(RuleRequest(run_dir=tmp_path, rules=tmp_path / "rules.py"))

    assert report.exit_code == 2
    assert "has no config.yaml — is --run-dir the run?" in report.payload["refused"]


# --- sampling and the CLI ---------------------------------------------------


def test_the_sample_is_deterministic_and_capped_by_the_pool(run, tmp_path):
    rules = _write_rules(tmp_path, "e/rules.py", SATURATED_RULE)
    first = _check(run, rules, sample=4)[1]
    second = _check(run, rules, sample=4)[1]

    assert first["rows"] == second["rows"]
    assert len(first["rows"]) == 4
    # The pool has six rows; the default sample asks for twenty and gets six.
    assert len(_check(run, rules)[1]["rows"]) == 6
    assert build_parser().parse_args(["--run-dir", ".", "--rules", "r.py"]).sample is None
    assert DEFAULT_SAMPLE == 20


def test_the_cli_prints_the_report_and_says_nothing_about_money(run, tmp_path, capsys):
    rules = _write_rules(tmp_path, "f/rules.py", DISCRIMINATING_RULE)
    code = main(
        [
            "--run-dir",
            str(run),
            "--rules",
            str(rules),
            "--sample",
            "6",
            "--repo-root",
            str(run.parent.parent),
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "check-rules: 1 rule(s)" in out
    assert "--- json ---" in out
    # Running a rule is free, so there is no ledger, no cap and no budget line
    # — the one place this tool deliberately says less than `score_tool`.
    assert "budget" not in out
    assert "$" not in out
