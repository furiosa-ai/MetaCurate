"""Check a draft `rules.py` on a few train rows, for the proposer. No LLM, free.

Why this exists. The proposer can measure a rubric criterion before it
commits to one (`meta_filter.score_tool`); this is the same for a RULE. The
proposer prompt says the consequence out loud: "A rule that reads a field
these rows do not carry returns None on every row and NOTHING in this run
will tell you." On a 220-row exam, for example:

* a seed's only rule can fire 0/220 on every pool;
* a rule reading `metadata["submission_bytes"]`, a field Nemotron rows do
  not carry, fires 0/220;
* rules firing 214/220 and 213/220 are saturated, so they order almost
  nothing;
* a rule firing 101/220 has the shape a useful ranking signal has.

A rule that fires on none, or on nearly all, of the rows contributes no
ordering, and otherwise nothing reports that until an iteration has been
spent on it.

What it does, exactly:

* imports the `rules.py` it is pointed at and runs every top-level `rule_*`
  callable over a few TRAIN rows, exactly as `solver.py` does — a rule is
  pure Python over the trajectory, so this is the whole rule, not a model of
  it. There is no LLM call anywhere in this module and nothing to charge:
  where `score_tool` keeps a dollar ledger, this tool says nothing, because
  running a rule is free;
* refuses any row that is not in the TRAIN pool, in `score_tool`'s words, so
  the hidden selection rows and the probe rows are unreachable through it;
* speaks the proposer's vocabulary. `--rows` accepts the `traj_...`
  pseudonyms the agent actually has as well as pool ids, and reports rows
  back under the pseudonym (`score_tool.IdMap`);
* reports, per rule, the fire rate with the two degenerate cases named in
  words, the evidence of a few rows that fired, any exception the rule
  raised — one raising rule never stops the others — and how the rule's own
  verdict orders the judged pairs among the sampled rows.

Invocation:

    uv run python -m meta_filter.rule_tool --run-dir <run> \\
        --rules filters/initial_blank_wired/rules.py --sample 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from meta_filter.core.schemas import Trajectory
from meta_filter.score_tool import (
    EXIT_OK,
    EXIT_REFUSED,
    MAX_ROWS_PER_CALL,
    REPO_ROOT,
    IdMap,
    _judged_pairs,
    _pick_rows,
    _resolve,
)
from meta_filter.visibility.trajectory import load_trajectories

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE = 20
# At or above this share of the sampled rows a rule separates almost nothing,
# so it is called out even when it does not fire on literally every row.
SATURATION_SHARE = 0.9
# How many firing rows the report quotes per rule, and how many raising rows.
MAX_EXAMPLES = 5
# A rule's verdict read as an ordering: REJECT ranks the row it fires on
# lower, ACCEPT ranks it higher, REVIEW says nothing and drops the pair.
RANKS_LOWER = "REJECT"
RANKS_HIGHER = "ACCEPT"


class RuleToolRefusal(Exception):
    """A request the tool will not serve, with the sentence the agent reads."""


@dataclass
class RuleRunSettings:
    """What the tool needs from `config.yaml`, resolved to absolute paths.

    Far less than `score_tool.RunSettings`: no scorer, no cache, no model, no
    budget, because nothing here talks to a model. A run whose config records
    no solver at all is served happily.
    """

    run_dir: Path
    train_data: Path


def load_run_settings(run_dir: Path, *, repo_root: Path = REPO_ROOT) -> RuleRunSettings:
    """Read `config.yaml` for the train pool. Same resolution as `score_tool`."""

    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise RuleToolRefusal(f"refused: {run_dir} has no config.yaml — is --run-dir the run?")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    train = cfg.get("train_data")
    if not train:
        raise RuleToolRefusal("refused: config.yaml records no train_data")
    return RuleRunSettings(run_dir=run_dir, train_data=_resolve(str(train), repo_root))


def load_rule_functions(rules_path: Path) -> list:
    """Import `rules.py` and return its top-level `rule_*` callables.

    The same discovery `filter_loader.load_filter` does, but off the one file:
    a draft is often checked before the policy beside it declares anything,
    and requiring the whole filter directory would refuse exactly the moment
    this tool is for.
    """

    rules_path = Path(rules_path)
    spec = importlib.util.spec_from_file_location(f"rule_tool_draft_{uuid.uuid4().hex}", rules_path)
    if spec is None or spec.loader is None:
        raise RuleToolRefusal(f"refused: cannot import {rules_path} as Python")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except SyntaxError as exc:
        raise RuleToolRefusal(
            f"refused: {rules_path} has a syntax error at line {exc.lineno}: {exc.msg}"
        ) from None
    except Exception as exc:  # noqa: BLE001 — a half-written draft is normal here
        raise RuleToolRefusal(
            f"refused: importing {rules_path} raised {type(exc).__name__}: {exc}"
        ) from None
    finally:
        sys.modules.pop(spec.name, None)
    return [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("rule_") and callable(getattr(module, name))
    ]


@dataclass
class RuleRequest:
    run_dir: Path
    rules: Path
    rows: list[str] | None = None
    sample: int = DEFAULT_SAMPLE
    seed: int = 0
    repo_root: Path = REPO_ROOT


@dataclass
class RuleReport:
    text: str
    payload: dict[str, Any]
    exit_code: int = EXIT_OK

    def render(self) -> str:
        return f"{self.text}\n--- json ---\n{json.dumps(self.payload, indent=2)}"


@dataclass
class RuleOutcome:
    """One rule's behaviour over the sampled rows."""

    name: str
    fired: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    rows: int = 0

    @property
    def fire_count(self) -> int:
        return len(self.fired)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def status(self) -> str:
        """`ERROR`, `DEAD`, `SATURATED` or `OK` — in that order of precedence.

        A rule that raised on every row is an ERROR, never DEAD: it fired on
        nothing, but the reason is a bug in the rule and not a pool that does
        not carry what it reads, and the two want opposite fixes.
        """

        if self.rows and self.error_count == self.rows:
            return "ERROR"
        if self.fire_count == 0:
            return "DEAD"
        if self.rows and self.fire_count / self.rows >= SATURATION_SHARE:
            return "SATURATED"
        return "OK"

    @property
    def error_types(self) -> list[str]:
        seen: list[str] = []
        for err in self.errors:
            if err["error_type"] not in seen:
                seen.append(err["error_type"])
        return seen


def _run_rule(rule, rows: list[Trajectory], idmap: IdMap) -> RuleOutcome:
    """Run one rule over the rows, collecting fires and exceptions alike.

    Every exception is caught per row, exactly as `solver.py` catches it: a
    rule that raises must cost the proposer its own report, not the reports
    of the rules beside it.
    """

    outcome = RuleOutcome(name=rule.__name__, rows=len(rows))
    for traj in rows:
        shown = idmap.display_id(traj.trajectory_id)
        try:
            fired = rule(traj)
        except Exception as exc:  # noqa: BLE001 — a draft rule raising is the news here
            outcome.errors.append(
                {
                    "trajectory_id": shown,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:200],
                }
            )
            continue
        if not fired:
            continue
        verdict = str((fired or {}).get("verdict") or "").upper() if isinstance(fired, dict) else ""
        evidence = []
        if isinstance(fired, dict):
            evidence = [str(line) for line in (fired.get("evidence") or [])]
        outcome.fired.append({"trajectory_id": shown, "verdict": verdict, "evidence": evidence})
    return outcome


def measure_rules(
    rules_path: Path,
    trajectories: Sequence[Trajectory],
    *,
    idmap: IdMap | None = None,
) -> list[RuleOutcome]:
    """Run every `rule_*` in `rules_path` over `trajectories`. No LLM, no cost.

    The importable half of what the CLI does, for callers that already hold
    the rows and want nothing else the CLI adds — no run directory, no train
    pool resolution, no pair credit. `precheck.py` is the caller: it measures
    the same rules over the precheck's own row set, which is not a sample of
    the train pool and has no `config.yaml` to find.

    Args:
        rules_path: A `rules.py` to import. Must exist and be importable.
        trajectories: The rows to run every rule over, in order.
        idmap: Names rows the way the agent does; the default identity map
            reports rows under their own trajectory ids.

    Returns:
        One `RuleOutcome` per discovered rule, in `dir()` order.

    Raises:
        RuleToolRefusal: `rules_path` cannot be imported as Python.
    """

    rules = load_rule_functions(Path(rules_path))
    resolved = idmap if idmap is not None else IdMap()
    rows = list(trajectories)
    return [_run_rule(rule, rows, resolved) for rule in rules]


def _pair_credit(
    outcome: RuleOutcome, sampled: list[str], judged: list[tuple[str, str]]
) -> dict[str, Any]:
    """How the rule's verdict orders the judged pairs among the sampled rows.

    Only pairs with BOTH rows sampled and EXACTLY ONE side firing say
    anything: when neither side fires the rule is silent about the pair, and
    when both fire it says the same thing about both. `REVIEW` (and any
    verdict that is neither `ACCEPT` nor `REJECT`) is neutral, so those pairs
    drop too. This is the rule-side counterpart of the judged-pair line
    `score_tool` prints for a criterion, and it is read the same way: near
    half is what a rule carrying no ordering information gives.
    """

    verdicts = {row["trajectory_id"]: row["verdict"] for row in outcome.fired}
    in_sample = set(sampled)
    considered = 0
    agreed = 0
    for hi, lo in judged:
        if hi not in in_sample or lo not in in_sample:
            continue
        hi_fires, lo_fires = hi in verdicts, lo in verdicts
        if hi_fires == lo_fires:
            continue
        side, verdict = (hi, verdicts[hi]) if hi_fires else (lo, verdicts[lo])
        if verdict not in (RANKS_LOWER, RANKS_HIGHER):
            continue
        considered += 1
        ranks_lower = verdict == RANKS_LOWER
        if (side == lo) == ranks_lower:
            agreed += 1
    return {"pairs_split_by_rule": considered, "ordered_the_judges_way": agreed}


def check_rules(request: RuleRequest) -> RuleReport:
    """Run one request end to end; refusals come back as a report, not a raise."""

    try:
        settings = load_run_settings(request.run_dir, repo_root=request.repo_root)
        return _check(request, settings)
    except RuleToolRefusal as exc:
        return RuleReport(text=str(exc), payload={"refused": str(exc)}, exit_code=EXIT_REFUSED)


def _resolve_rules_path(rules: Path, settings: RuleRunSettings, repo_root: Path) -> Path:
    """Find the `rules.py` a relative `--rules` names.

    `score_tool` resolves its `--rubric` against `--run-dir` alone, because
    the proposer's cwd IS the run directory and every rubric it drafts lives
    under it. That is still the first place looked, but not the only one: a
    rules.py is also checked by hand from a checkout, against a filter
    directory in the repository that no run copy exists of yet, and refusing
    `filters/<x>/rules.py` there would be a refusal about nothing.
    """

    if rules.is_absolute():
        candidates = [rules]
    else:
        candidates = [settings.run_dir / rules, rules, repo_root / rules]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuleToolRefusal(f"refused: rules {candidates[0]} does not exist")


def _check(request: RuleRequest, settings: RuleRunSettings) -> RuleReport:
    rules_path = _resolve_rules_path(Path(request.rules), settings, request.repo_root)
    rule_functions = load_rule_functions(rules_path)
    if not rule_functions:
        raise RuleToolRefusal(
            f"refused: {rules_path} defines no top-level `rule_*` function; there is "
            "nothing to check"
        )

    # The same stripped view the run scored.
    train = load_trajectories(settings.train_data)
    by_id = {t.trajectory_id: t for t in train}
    idmap = IdMap.for_run(settings.run_dir)
    row_ids = _pick_rows(request, list(by_id), idmap, refusal=RuleToolRefusal)
    rows = [by_id[r] for r in row_ids]
    sampled = [idmap.display_id(r) for r in row_ids]
    judged = _judged_pairs(settings.run_dir)

    outcomes = [_run_rule(rule, rows, idmap) for rule in rule_functions]
    credits = {o.name: _pair_credit(o, sampled, judged) for o in outcomes}
    text = _render_text(rules_path, outcomes, credits, len(rows))
    payload = {
        "rules": str(rules_path),
        "rows": sampled,
        "n_rows": len(rows),
        "rule_results": [
            {
                "name": o.name,
                "status": o.status,
                "fires": o.fire_count,
                "raised": o.error_count,
                "error_types": o.error_types,
                "fired_rows": o.fired[:MAX_EXAMPLES],
                "errors": o.errors[:MAX_EXAMPLES],
                **credits[o.name],
            }
            for o in outcomes
        ],
        "summary": _summary(outcomes),
    }
    return RuleReport(text=text, payload=payload)


def _summary(outcomes: list[RuleOutcome]) -> dict[str, int]:
    return {
        "rules": len(outcomes),
        "dead": sum(1 for o in outcomes if o.status == "DEAD"),
        "saturated": sum(1 for o in outcomes if o.status == "SATURATED"),
        "raised": sum(1 for o in outcomes if o.error_count),
    }


def _fire_line(outcome: RuleOutcome) -> str:
    """The rule's headline: the count, and the degenerate cases in words."""

    k, n = outcome.fire_count, outcome.rows
    match outcome.status:
        case "ERROR":
            types = ", ".join(outcome.error_types) or "an exception"
            return f"ERROR — raised {types} on all {n} row(s), so it was never measured"
        case "DEAD":
            return f"{k}/{n} — DEAD: this rule never fires on these rows"
        case "SATURATED":
            return f"{k}/{n} — SATURATED: fires on nearly every row, so it orders almost nothing"
        case _:
            return f"{k}/{n}"


def _credit_line(credit: dict[str, Any]) -> str:
    """The pair-ordering line, or the reason there is no percentage to print."""

    b = credit["pairs_split_by_rule"]
    if not b:
        return (
            "pair credit: no judged pair among these rows has exactly one side firing, "
            "so this rule's ordering is unmeasured here"
        )
    a = credit["ordered_the_judges_way"]
    pct = 100.0 * a / b
    line = f"pair credit: ordered the judge's way: {a}/{b} ({pct:.0f}%)"
    if abs(pct - 50.0) <= 10.0:
        line += " — near 50% means the rule is not carrying ordering information"
    return line


def _render_text(
    rules_path: Path,
    outcomes: list[RuleOutcome],
    credits: dict[str, dict[str, Any]],
    n_rows: int,
) -> str:
    lines = [
        f"check-rules: {len(outcomes)} rule(s) from {rules_path} on {n_rows} train row(s) "
        "(pure Python, no LLM, no cost)"
    ]
    for outcome in outcomes:
        lines.append(f"  {outcome.name}: {_fire_line(outcome)}")
        for row in outcome.fired[:MAX_EXAMPLES]:
            verdict = f" [{row['verdict']}]" if row["verdict"] else ""
            lines.append(f"    fired on {row['trajectory_id']}{verdict}")
            for ev in row["evidence"]:
                lines.append(f"      evidence: {ev}")
        if outcome.errors and outcome.status != "ERROR":
            lines.append(
                f"    raised on {outcome.error_count}/{outcome.rows} row(s): "
                + ", ".join(outcome.error_types)
            )
        for err in outcome.errors[:MAX_EXAMPLES]:
            lines.append(f"      {err['trajectory_id']}: {err['error_type']}: {err['error']}")
        lines.append(f"    {_credit_line(credits[outcome.name])}")
    summary = _summary(outcomes)
    lines.append(
        f"summary: {summary['rules']} rule(s) — {summary['dead']} dead, "
        f"{summary['saturated']} saturated, {summary['raised']} raised"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check-rules",
        description=(
            "Run a draft rules.py over a few TRAIN rows and report what each rule "
            "actually fires on. Pure Python: no LLM call, no cost, no budget."
        ),
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="The run directory.")
    parser.add_argument(
        "--rules",
        required=True,
        type=Path,
        help="A rules.py — typically a candidate directory's; relative to --run-dir.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--rows",
        help="Comma-separated trajectory ids, in either spelling — the `traj_...` names the "
        "run's diagnostics print, or pool ids; every one must resolve to a train row.",
    )
    selection.add_argument(
        "--sample",
        type=int,
        default=None,
        help=f"Pick this many train rows deterministically (default {DEFAULT_SAMPLE}, "
        f"max {MAX_ROWS_PER_CALL}).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for --sample.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Where relative paths in config.yaml resolve from.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    request = RuleRequest(
        run_dir=args.run_dir,
        rules=args.rules,
        rows=args.rows.split(",") if args.rows else None,
        sample=args.sample if args.sample is not None else DEFAULT_SAMPLE,
        seed=args.seed,
        repo_root=Path(args.repo_root),
    )
    report = check_rules(request)
    print(report.render())
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
