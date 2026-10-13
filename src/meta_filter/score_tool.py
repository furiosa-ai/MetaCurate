"""Score one rubric criterion on a few train rows, for the proposer, with a budget.

Why this exists. The agentic proposer only builds what it can verify
offline. `rules.py` and `policy.py` run in its workspace — it prototypes a
rule against `solver_outputs/`, reads the result, revises. `rubric.md` needs
the LLM scorer, which it cannot run, so without this tool a rubric edit is a
blind bet and a proposer rarely makes one. Meta-Harness's proposer could run
its harness; this tool lets ours run the rubric: draft a
criterion, score it on ten train rows under the run's own scorer model and
prompt, read the evidence lines, revise — the same loop the proposer already
runs on rules.

What it does, exactly:

* reads the run's `config.yaml` for the train pool, the solver model / request
  body / cache directory (recorded there by the optimizer). Cells already in
  the run's scorer cache are read from it (read-only, and the tool only ever
  asks for train rows); every NEW cell is
  written under the same key into a staging directory inside the workspace,
  `<run>/proposer_score_cache/`, which the optimizer merges into the run
  cache after the proposer leg — train-row cells only, never over an
  existing cell (`merge_staged_cells`). The run cache is not writable from
  the agent's sandbox: it also holds the hidden selection rows' cells, whose
  ids are visible in the filenames, and a writable cache is a forgeable gate;
* scores ONLY the named criterion, through `LLMRubricScorer.score_one`, the
  same per-criterion path and prompt the real pass uses (preamble + that
  criterion's section, split at every criterion the candidate's policy
  declares);
* refuses any row that is not in the TRAIN pool. The hidden selection rows
  and the probe rows are therefore unreachable, whatever id is passed, and
  the tool never opens `runs/_selection/`;
* speaks the proposer's vocabulary. Everything the agent reads names
  trajectories by pseudonym (`traj_21c3fa9a`); the pool ids
  (`dask__dask-9235::att0`) exist only outside the run directory, in
  `runs/_idmap/<run_id>.json`. Matching `--rows` against pool ids alone
  would refuse every targeted call the proposer makes. The tool runs on the
  harness side and may read the map, so it resolves pseudonyms before the
  train check and reports rows back under the same pseudonyms. The agent
  still never sees a pool id: answering a pseudonym with a pool id would
  hand over the join the map exists to keep;

* keeps a per-leg dollar ledger (`<run>/proposer_score_budget.json`). The
  optimizer resets it at the start of every proposer leg and folds what was
  spent into `cost_proposer`; the tool refuses a call whose estimated cost
  would take the leg past its cap, and every output ends with the budget
  line so the agent always knows what it has left.

Invocation (the optimizer prints the exact command into the prompt):

    <venv python> -m meta_filter.score_tool --run-dir <run> \\
        --rubric iterations/003/candidates/iter003_00_x/rubric.md \\
        --criterion "Verification Completeness" --sample 10

All LLM traffic goes through OpenRouter under `OPENROUTER_API_KEY`, read from
the environment or line by line from `~/.env` (the harnesses strip the
provider variables from the agent's shell). The key value is never printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from meta_filter.filter_loader import load_filter
from meta_filter.llm.cost_meter import CostLedger, MeteredClient
from meta_filter.llm.pricing import list_price_usd
from meta_filter.llm_scorer import (
    LLMRubricScorer,
    _split_rubric_sections,
    cache_key_trajectory_id,
    sanitize_trajectory_id,
)
from meta_filter.pair_dump import FILENAME as RANKING_PAIRS_FILENAME
from meta_filter.rendering import render_submission, render_turns
from meta_filter.schemas import Trajectory
from meta_filter.trajectory import load_trajectories

logger = logging.getLogger(__name__)

LEDGER_FILENAME = "proposer_score_budget.json"
# Where the tool writes the cells it buys, inside the run directory; the
# optimizer merges it into the run cache after each proposer leg.
STAGING_DIRNAME = "proposer_score_cache"
# Beside the runs, never inside one: `optimizer.py` writes the pseudonym map
# to `<runs_dir>/_idmap/<run_id>.json` precisely so the agent's sandbox
# cannot open it. The tool is a harness-side process and may.
IDMAP_DIRNAME = "_idmap"
DEFAULT_BUDGET_USD = 1.50
DEFAULT_SAMPLE = 10
MAX_ROWS_PER_CALL = 20
# What ten rows cost, measured rather than guessed: ten cells at the loop's
# scorer settings cost about $0.054. The prompt turns the cap into a call
# count with this, because a cap in dollars tells the proposer nothing.
USD_PER_TEN_ROW_CALL = 0.055
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Output tokens one per-criterion answer typically takes: a JSON object with
# one score and one evidence line plus a summary sentence.
_ESTIMATED_COMPLETION_TOKENS = 200
# `meta_filter.score_tool` is `src/meta_filter/score_tool.py`; the repository
# root is two levels up. Relative paths in `config.yaml` (`data/...`) were
# written relative to the cwd the optimizer ran from, which is that root.
REPO_ROOT = Path(__file__).resolve().parents[2]

EXIT_OK = 0
EXIT_REFUSED = 2


class ScoreToolRefusal(Exception):
    """A request the tool will not serve, with the sentence the agent reads."""


@dataclass(frozen=True)
class ScoreToolAccess:
    """What the optimizer hands the proposer so it can call the tool.

    `command` is spelled into the prompt verbatim; `executable` is what the
    Claude Code guard must allow a Bash call to name (it lies in the venv,
    outside the run directory). Nothing here widens what the sandbox may
    WRITE: the tool writes only inside the run directory, and the Codex
    hole it needs is the network, not a root.
    """

    command: str
    budget_usd: float
    executable: str

    @property
    def affordable_ten_row_calls(self) -> int:
        """Roughly how many ten-row calls the cap buys — for the prompt.

        The prompt has to say this in calls, not in dollars: $1.50 is about
        27 ten-row calls, and the proposer — told only the ceiling — spent
        3.6% of one leg's cap on a single call.
        """

        return max(1, int(self.budget_usd / USD_PER_TEN_ROW_CALL))


def access_for(run_dir: Path, *, budget_usd: float) -> ScoreToolAccess:
    """The `ScoreToolAccess` for one run, built from the running interpreter.

    `sys.executable` rather than `uv run`: the agent's cwd is the run
    directory, which may lie outside the repository (`--runs-dir`), and
    `uv run` would then find no project. The venv interpreter is absolute
    and already has the package installed.
    """

    command = f"{sys.executable} -m meta_filter.score_tool --run-dir {Path(run_dir).resolve()}"
    return ScoreToolAccess(command=command, budget_usd=float(budget_usd), executable=sys.executable)


def staging_dir_for(run_dir: Path) -> Path:
    return Path(run_dir) / STAGING_DIRNAME


@dataclass
class IdMap:
    """The run's pseudonym map, read from beside the runs rather than inside.

    `meta_filter.pseudonym` gives every pool row a per-run `traj_<8 hex>`
    name and the optimizer writes both directions to
    `<runs_dir>/_idmap/<run_id>.json`. Everything the proposer reads — the
    prompt, `critic_outputs/`, `ranking_pairs.jsonl` — carries the pseudonym
    and nothing carries the pool id, so the pseudonym is the only name the
    agent can possibly type.

    A run with no map (every non-agentic run, and the fixtures) yields an
    empty one, and then every method is the identity: the tool behaves
    exactly as it did before pseudonyms existed.
    """

    pseudonym_to_id: dict[str, str] = field(default_factory=dict)
    id_to_pseudonym: dict[str, str] = field(default_factory=dict)

    @classmethod
    def for_run(cls, run_dir: Path) -> IdMap:
        """Load `<runs_dir>/_idmap/<run_id>.json`; an absent map is empty."""

        run_dir = Path(run_dir)
        path = run_dir.parent / IDMAP_DIRNAME / f"{run_dir.name}.json"
        if not path.is_file():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable idmap at %s; treating ids as pool ids", path)
            return cls()
        forward = payload.get("pseudonym_to_id")
        backward = payload.get("id_to_pseudonym")
        return cls(
            pseudonym_to_id=dict(forward) if isinstance(forward, dict) else {},
            id_to_pseudonym=dict(backward) if isinstance(backward, dict) else {},
        )

    def __bool__(self) -> bool:
        return bool(self.pseudonym_to_id or self.id_to_pseudonym)

    def real_id(self, token: str) -> str:
        """The pool id behind `token`, or `token` when it is one already."""

        return self.pseudonym_to_id.get(token, token)

    def display_id(self, trajectory_id: str) -> str:
        """What the agent calls this row, so the report speaks its language."""

        return self.id_to_pseudonym.get(trajectory_id, trajectory_id)

    def names_a_row(self, token: str) -> bool:
        """Whether `token` names a row of this run's pools at all.

        Both spellings count: the pseudonym the agent was given and the pool
        id behind it. What this separates is a name from a non-name — never
        one pool from another, which is why it is not called `is_train`.
        """

        return token in self.pseudonym_to_id or token in self.id_to_pseudonym


@dataclass
class StagedRubricScorer(LLMRubricScorer):
    """The run's scorer with its writes redirected into the workspace.

    `cache_dir` (the base class's) is the staging directory: every write
    goes there. Reads try the run cache first, read-only, then the staging
    directory, so a cell the full pass already paid for is served and a
    cell the tool bought earlier this leg is served too. The run cache may
    be unreadable under the sandbox; that reads as a miss and the cell is
    bought again, which is the price of not opening the cache to writes.

    `train_ids` bounds what is ever read from the run cache. The tool only
    asks for train rows in the first place, so this is a second fence on
    the same door rather than a policy of its own.
    """

    run_cache_dir: Path | None = None
    train_ids: frozenset[str] = frozenset()

    def _read_criterion_entry(
        self,
        trajectory: Trajectory,
        criterion: str,
        section_text: str,
        model: str,
        field_sig: str = "",
        header: str = "",
    ) -> tuple[float, list[str]] | None:
        staged = super()._read_criterion_entry(
            trajectory, criterion, section_text, model, field_sig, header
        )
        if staged is not None or self.run_cache_dir is None:
            return staged
        if trajectory.trajectory_id not in self.train_ids:
            return None
        from meta_filter.llm_scorer import _criterion_cache_key

        key = _criterion_cache_key(section_text, criterion, trajectory, model, field_sig, header)
        path = Path(self.run_cache_dir) / "per_criterion" / f"{key}.json"
        try:
            if not path.is_file():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return float(data["score"]), [str(line) for line in data.get("evidence", [])]
        except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
            return None


def merge_staged_cells(
    staging_dir: Path, cache_dir: Path, train_ids: Iterable[str]
) -> dict[str, int]:
    """Move the tool's staged cells into the run cache, then clear the staging.

    Two refusals, both about what a proposer (or a runaway tool) could have
    put in a directory it can write to: a cell whose leading trajectory id
    is not a TRAIN row is dropped — the selection and probe rows' cells are
    the gate's and nobody in the workspace may supply them — and a cell
    whose key the run cache already holds is dropped, because the canonical
    value of a cell is the one the loop scored, never one handed in. The
    filename's id and the JSON's `trajectory_id` must agree, and both must
    be train. Counts are returned for the log: ``merged``,
    ``rejected_by_id``, ``rejected_existing``, ``rejected_unreadable``.
    """

    counts = {"merged": 0, "rejected_by_id": 0, "rejected_existing": 0, "rejected_unreadable": 0}
    staging_dir = Path(staging_dir)
    source = staging_dir / "per_criterion"
    if not source.is_dir():
        if staging_dir.is_dir():
            shutil.rmtree(staging_dir, ignore_errors=True)
        return counts
    allowed_ids = set(train_ids)
    allowed_keys = {sanitize_trajectory_id(tid) for tid in allowed_ids}
    target = Path(cache_dir) / "per_criterion"
    target.mkdir(parents=True, exist_ok=True)
    for cell in sorted(source.glob("*.json")):
        leading = cache_key_trajectory_id(cell.name)
        if leading is None or leading not in allowed_keys:
            counts["rejected_by_id"] += 1
            continue
        try:
            data = json.loads(cell.read_text(encoding="utf-8"))
            recorded = str(data["trajectory_id"])
            float(data["score"])
        except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
            counts["rejected_unreadable"] += 1
            continue
        if recorded not in allowed_ids or sanitize_trajectory_id(recorded) != leading:
            counts["rejected_by_id"] += 1
            continue
        destination = target / cell.name
        if destination.exists():
            counts["rejected_existing"] += 1
            continue
        shutil.copyfile(cell, destination)
        counts["merged"] += 1
    shutil.rmtree(staging_dir, ignore_errors=True)
    return counts


# --------------------------------------------------------------------- ledger


@dataclass
class ScoreBudgetLedger:
    """`<run>/proposer_score_budget.json`: what this leg's tool calls cost.

    One live section (`iteration`, `leg`, `cap_usd`, `spent_usd`, `calls`,
    `entries`) and a `history` of closed legs. The optimizer resets the live
    section at every proposer leg start and reads it back afterwards to bill
    the proposer role; the tool charges it after every call it makes. The
    file lives in the run directory on purpose: it is the one place both the
    optimizer process and the agent's subprocess can reach, and the agent is
    welcome to read it.
    """

    path: Path
    iteration: int | None = None
    leg: str = "unknown"
    cap_usd: float = DEFAULT_BUDGET_USD
    spent_usd: float = 0.0
    calls: int = 0
    tokens: int = 0
    unpriced_calls: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, run_dir: Path, *, default_cap: float = DEFAULT_BUDGET_USD) -> ScoreBudgetLedger:
        path = Path(run_dir) / LEDGER_FILENAME
        ledger = cls(path=path, cap_usd=float(default_cap))
        if not path.is_file():
            return ledger
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable score budget ledger at %s; starting a fresh one", path)
            return ledger
        ledger.iteration = data.get("iteration")
        ledger.leg = str(data.get("leg") or "unknown")
        ledger.cap_usd = float(data.get("cap_usd", default_cap))
        ledger.spent_usd = float(data.get("spent_usd", 0.0))
        ledger.calls = int(data.get("calls", 0))
        ledger.tokens = int(data.get("tokens", 0))
        ledger.unpriced_calls = int(data.get("unpriced_calls", 0))
        ledger.entries = list(data.get("entries") or [])
        ledger.history = list(data.get("history") or [])
        return ledger

    @property
    def remaining_usd(self) -> float:
        return max(0.0, round(self.cap_usd - self.spent_usd, 6))

    def budget_line(self) -> str:
        """The sentence every tool output ends with."""

        return (
            f"budget: spent ${self.spent_usd:.2f} of ${self.cap_usd:.2f} this leg, "
            f"remaining ${self.remaining_usd:.2f}"
        )

    def reset(self, *, iteration: int, leg: str, cap_usd: float) -> None:
        """Close the live section into `history` and open a fresh leg."""

        if self.calls or self.spent_usd:
            self.history.append(self.closed_section())
        self.iteration = iteration
        self.leg = leg
        self.cap_usd = float(cap_usd)
        self.spent_usd = 0.0
        self.calls = 0
        self.tokens = 0
        self.unpriced_calls = 0
        self.entries = []

    def closed_section(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "leg": self.leg,
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 6),
            "calls": self.calls,
            "tokens": self.tokens,
            "unpriced_calls": self.unpriced_calls,
        }

    def charge(self, entry: dict[str, Any]) -> None:
        self.spent_usd = round(self.spent_usd + float(entry.get("cost_usd", 0.0)), 6)
        self.calls += int(entry.get("llm_calls", 0))
        self.tokens += int(entry.get("llm_tokens", 0))
        self.unpriced_calls += int(entry.get("unpriced_calls", 0))
        self.entries.append(entry)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self.closed_section(),
            "entries": self.entries,
            "history": self.history,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- request


@dataclass
class RunSettings:
    """What the tool needs from `config.yaml`, resolved to absolute paths."""

    run_dir: Path
    train_data: Path
    solver_backend: str
    solver_provider: str
    solver_model: str
    solver_base_url: str
    solver_extra_body: dict | None
    scorer_cache_dir: Path
    budget_usd: float


def _resolve(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (repo_root / path)


def load_run_settings(
    run_dir: Path, *, repo_root: Path = REPO_ROOT, provider_override: str | None = None
) -> RunSettings:
    """Read `config.yaml`; refuse a run whose scorer the tool cannot rebuild.

    `provider_override` (`--solver-provider`, offline checks) names the
    client to build instead of the recorded one and lifts the backend
    refusal with it: a heuristic-scored mock run can then still exercise
    the tool end to end against the mock, which is what the smoke does.
    """

    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise ScoreToolRefusal(f"refused: {run_dir} has no config.yaml — is --run-dir the run?")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    backend = cfg.get("solver_backend")
    provider = str(provider_override or cfg.get("solver_provider") or "").lower()
    if provider_override is None:
        if backend is None:
            raise ScoreToolRefusal(
                "refused: this run's config.yaml records no solver, so the tool "
                "cannot rebuild the run's scorer"
            )
        if backend != "llm":
            raise ScoreToolRefusal(
                f"refused: this run scores with the {backend} scorer, which reads no rubric "
                "text; there is nothing an LLM could tell you about a criterion here"
            )
    if provider not in ("openai", "mock"):
        raise ScoreToolRefusal(
            f"refused: solver provider {provider!r} is not routed through OpenRouter; "
            "the tool only speaks to OpenAI-compatible endpoints (or the mock)"
        )
    cache = cfg.get("solver_cache_dir")
    if not cache:
        raise ScoreToolRefusal("refused: config.yaml records no solver_cache_dir")
    return RunSettings(
        run_dir=run_dir,
        train_data=_resolve(str(cfg["train_data"]), repo_root),
        solver_backend=str(backend or provider_override),
        solver_provider=provider,
        solver_model=str(cfg.get("solver_model") or ""),
        solver_base_url=str(cfg.get("solver_base_url") or OPENROUTER_BASE_URL),
        solver_extra_body=cfg.get("solver_extra_body") or None,
        scorer_cache_dir=_resolve(str(cache), repo_root),
        budget_usd=float(cfg.get("proposer_score_budget") or DEFAULT_BUDGET_USD),
    )


def build_solver_client(settings: RunSettings, *, provider_override: str | None = None):
    """The run's scorer client, rebuilt for one subprocess.

    OpenRouter only. The key comes from `OPENROUTER_API_KEY` or `~/.env`
    (`codex_cli.read_provider_key`), never from `OPENAI_API_KEY`, and is
    handed straight to the client without being logged or echoed.
    """

    provider = (provider_override or settings.solver_provider).lower()
    if provider == "mock":
        from meta_filter.llm.mock import MockLLMClient

        return MockLLMClient(model=settings.solver_model or "mock-1")
    if provider != "openai":
        raise ScoreToolRefusal(f"refused: unsupported solver provider {provider!r}")
    from meta_filter.llm.codex_cli import read_provider_key
    from meta_filter.llm.openai_compatible import OpenAICompatibleClient

    try:
        key = read_provider_key("OPENROUTER_API_KEY")
    except KeyError as exc:
        raise ScoreToolRefusal(
            f"refused: {exc.args[0]}; the tool cannot reach OpenRouter"
        ) from None
    return OpenAICompatibleClient(
        model=settings.solver_model,
        api_key=key,
        base_url=settings.solver_base_url or OPENROUTER_BASE_URL,
        extra_body=dict(settings.solver_extra_body) if settings.solver_extra_body else None,
    )


def _latest_iteration_dir(run_dir: Path, iteration: int | None) -> Path | None:
    root = run_dir / "iterations"
    if iteration is not None:
        path = root / f"{iteration:03d}"
        return path if path.is_dir() else None
    if not root.is_dir():
        return None
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    return dirs[-1] if dirs else None


def _criteria_for(rubric_path: Path, criterion: str) -> tuple[list[str], str]:
    """The criteria list the full pass will split the rubric on, and a note.

    The policy beside the rubric is the source of truth (`RUBRIC_CRITERIA`);
    the note is non-empty when the requested criterion is not in it yet,
    because cells scored under a criteria list the policy does not declare
    are filed under a key the full pass never reads.
    """

    policy = rubric_path.parent / "policy.py"
    criteria: list[str] = []
    if policy.is_file():
        try:
            loaded = load_filter(rubric_path.parent)
            criteria = [str(c) for c in getattr(loaded.policy_module, "RUBRIC_CRITERIA", [])]
        except Exception as exc:  # noqa: BLE001 — a half-written candidate is normal here
            logger.warning("could not import %s (%s); splitting on headings", policy, exc)
    if not criteria:
        from meta_filter.constraints import criterion_names

        criteria = criterion_names(rubric_path)
    note = ""
    if criterion not in criteria:
        note = (
            f"note: {criterion!r} is not in RUBRIC_CRITERIA of {policy.name} beside this "
            "rubric; the cells scored now are reused by the full pass only once the "
            "policy declares it"
        )
    return criteria, note


def _judged_pairs(run_dir: Path) -> list[tuple[str, str]]:
    """Judge-ordered (hi, lo) pairs from the latest `ranking_pairs.jsonl`."""

    root = run_dir / "iterations"
    if not root.is_dir():
        return []
    for it_dir in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        path = it_dir / RANKING_PAIRS_FILENAME
        if not path.is_file():
            continue
        pairs: list[tuple[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("tie") or not row.get("hi") or not row.get("lo"):
                continue
            pairs.append((str(row["hi"]), str(row["lo"])))
        return pairs
    return []


def _estimate_cost_usd(
    model: str, trajectories: list[Trajectory], rubric_text: str
) -> float | None:
    """List-price estimate for scoring one criterion on `trajectories`.

    Tokens are approximated from the rendered transcript at four characters
    each — the same rendering the prompt uses, so a long trajectory costs a
    long estimate. None when the model has no list price, in which case only
    the spent-versus-cap check can refuse.
    """

    total = 0.0
    for traj in trajectories:
        chars = (
            len(rubric_text)
            + len(traj.task or "")
            + len(render_turns(traj))
            + len(render_submission(traj))
            + 600
        )
        cost = list_price_usd(
            model,
            prompt_tokens=chars // 4,
            completion_tokens=_ESTIMATED_COMPLETION_TOKENS,
        )
        if cost is None:
            return None
        total += cost
    return round(total, 6)


@dataclass
class ScoreRequest:
    run_dir: Path
    rubric: Path
    criterion: str
    rows: list[str] | None = None
    sample: int = DEFAULT_SAMPLE
    seed: int = 0
    iteration: int | None = None
    baseline_rubric: Path | None = None
    provider_override: str | None = None
    repo_root: Path = REPO_ROOT


@dataclass
class ScoreReport:
    text: str
    payload: dict[str, Any]
    budget_line: str
    exit_code: int = EXIT_OK

    def render(self) -> str:
        return (
            f"{self.text}\n--- json ---\n{json.dumps(self.payload, indent=2)}\n{self.budget_line}"
        )


def _pick_rows(
    request: ScoreRequest,
    train_ids: list[str],
    idmap: IdMap,
    refusal: type[Exception] = ScoreToolRefusal,
) -> list[str]:
    """The pool ids to score, with pseudonyms resolved first.

    Resolution has to happen before the train check, not after: the agent
    only ever sees `traj_...` names, so matching `--rows` against pool ids
    alone refused every targeted call it made and the refusal read as a
    policy boundary rather than a miss. Both spellings are accepted, and a
    call naming a row twice (once each way) scores it once.

    The boundary itself does not move — a pseudonym that resolves outside
    the train pool is refused exactly as before, in the same words. What is
    new is that a name the run does not know is told so: that case is a typo
    the agent can fix, and it has to be told apart from the hidden rows the
    agent must not reach.

    `rule_tool` shares this function with its own `refusal` class, so both
    tools refuse in the same words.
    """

    if request.rows:
        wanted = [r.strip() for r in request.rows if r.strip()]
        if len(wanted) > MAX_ROWS_PER_CALL:
            raise refusal(
                f"refused: {len(wanted)} rows requested; at most {MAX_ROWS_PER_CALL} per call"
            )
        train = set(train_ids)
        resolved: list[str] = []
        outside: list[str] = []
        unknown: list[str] = []
        for token in wanted:
            real = idmap.real_id(token)
            if real in train:
                if real not in resolved:
                    resolved.append(real)
            elif idmap and not idmap.names_a_row(token):
                unknown.append(token)
            else:
                outside.append(token)
        if unknown:
            raise refusal(
                f"refused: {len(unknown)} unrecognised id(s) ({', '.join(unknown[:5])}); "
                "no trajectory of this run goes by that name — check the spelling against "
                "the ids the diagnostics print and call again"
            )
        if outside:
            # The same sentence whether the id is a selection row or a probe
            # row: saying which would be its own leak.
            raise refusal(
                f"refused: {len(outside)} id(s) are not in this run's train pool "
                f"({', '.join(outside[:5])}); only train rows can be scored here"
            )
        return resolved
    n = request.sample
    if n < 1 or n > MAX_ROWS_PER_CALL:
        raise refusal(f"refused: --sample must be between 1 and {MAX_ROWS_PER_CALL}")
    ordered = sorted(train_ids)
    return random.Random(request.seed).sample(ordered, min(n, len(ordered)))


def score_criterion(request: ScoreRequest, *, client_factory=None) -> ScoreReport:
    """Run one request end to end; refusals come back as a report, not a raise.

    `client_factory(settings)` replaces `build_solver_client` in tests. The
    ledger is read before anything is scored and written after, so a refusal
    leaves it untouched.
    """

    settings = load_run_settings(
        request.run_dir, repo_root=request.repo_root, provider_override=request.provider_override
    )
    ledger = ScoreBudgetLedger.load(settings.run_dir, default_cap=settings.budget_usd)
    try:
        return _score(request, settings, ledger, client_factory=client_factory)
    except ScoreToolRefusal as exc:
        return ScoreReport(
            text=str(exc),
            payload={"refused": str(exc)},
            budget_line=ledger.budget_line(),
            exit_code=EXIT_REFUSED,
        )


def _score(
    request: ScoreRequest,
    settings: RunSettings,
    ledger: ScoreBudgetLedger,
    *,
    client_factory,
) -> ScoreReport:
    rubric_path = Path(request.rubric)
    if not rubric_path.is_absolute():
        rubric_path = settings.run_dir / rubric_path
    if not rubric_path.is_file():
        raise ScoreToolRefusal(f"refused: rubric {rubric_path} does not exist")
    rubric_md = rubric_path.read_text(encoding="utf-8")
    criterion = request.criterion.strip()
    if not criterion:
        raise ScoreToolRefusal("refused: --criterion is empty")
    criteria, criteria_note = _criteria_for(rubric_path, criterion)
    split_on = criteria if criterion in criteria else [*criteria, criterion]
    sections = _split_rubric_sections(rubric_md, split_on)
    if sections.get(criterion, "") == rubric_md:
        raise ScoreToolRefusal(
            f"refused: no line of {rubric_path.name} names the criterion {criterion!r}; "
            "the scorer would fall back to the whole rubric, which is not the cell the "
            "full pass reads"
        )

    # The same stripped view the run scored, so a cell is keyed exactly as
    # the run's own pass keys it.
    train = load_trajectories(settings.train_data)
    by_id = {t.trajectory_id: t for t in train}
    idmap = IdMap.for_run(settings.run_dir)
    row_ids = _pick_rows(request, list(by_id), idmap)
    rows = [by_id[r] for r in row_ids]

    if ledger.spent_usd >= ledger.cap_usd:
        raise ScoreToolRefusal(
            f"refused: this leg's scoring budget is spent (${ledger.spent_usd:.2f} of "
            f"${ledger.cap_usd:.2f}); revise from the evidence you already have"
        )

    client = (
        client_factory(settings)
        if client_factory is not None
        else build_solver_client(settings, provider_override=request.provider_override)
    )
    meter = CostLedger()
    metered_client = MeteredClient(client=client, role="score_tool", ledger=meter)
    scorer = StagedRubricScorer(
        client=metered_client,
        cache_dir=staging_dir_for(settings.run_dir),
        run_cache_dir=settings.scorer_cache_dir,
        train_ids=frozenset(by_id),
    )

    # Cells already cached cost nothing; estimate only what will be bought.
    uncached = [
        t
        for t in rows
        if scorer.score_one(t, criterion, criteria, rubric_md, cache_only=True).score is None
    ]
    estimate = _estimate_cost_usd(scorer._model_name(), uncached, sections[criterion])
    if estimate is not None and ledger.spent_usd + estimate > ledger.cap_usd:
        raise ScoreToolRefusal(
            f"refused: scoring {len(uncached)} uncached cell(s) is estimated at "
            f"${estimate:.2f}, which would take this leg past its cap "
            f"(${ledger.spent_usd:.2f} spent of ${ledger.cap_usd:.2f}); ask for fewer rows"
        )

    baseline = _baseline(request, settings, scorer, criterion)
    started = time.time()
    per_row: list[dict[str, Any]] = []
    for traj in rows:
        # Reported under the name the agent knows the row by. Printing pool
        # ids would hand over the join the pseudonym map exists to withhold;
        # the pseudonym is also what makes the judged-pair block
        # below work, since `ranking_pairs.jsonl` is pseudonymized too.
        shown = idmap.display_id(traj.trajectory_id)
        try:
            cell = scorer.score_one(traj, criterion, criteria, rubric_md)
        except Exception as exc:  # noqa: BLE001 — one unreadable row must not lose the rest
            per_row.append({"trajectory_id": shown, "error": str(exc)[:200]})
            continue
        old = baseline.get(traj.trajectory_id) if baseline is not None else None
        per_row.append(
            {
                "trajectory_id": shown,
                "score": cell.score,
                "cached": cell.cached,
                "evidence": [line.removeprefix(f"[{criterion}] ") for line in cell.evidence],
                "baseline_score": old,
            }
        )

    spent = meter.snapshot().get("score_tool")
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "criterion": criterion,
        "rubric": str(rubric_path),
        "rows": len(rows),
        "cells_scored": sum(1 for r in per_row if "score" in r and not r["cached"]),
        "cells_cached": sum(1 for r in per_row if r.get("cached")),
        "cost_usd": round(spent.cost_usd, 6) if spent else 0.0,
        "llm_calls": spent.calls if spent else 0,
        "llm_tokens": spent.tokens if spent else 0,
        "unpriced_calls": spent.unpriced_calls if spent else 0,
    }
    ledger.charge(entry)
    ledger.save()

    pairs_block = _pairs_block(settings.run_dir, per_row)
    text = _render_text(
        criterion, rubric_path, per_row, baseline, criteria_note, pairs_block, entry
    )
    payload = {
        "criterion": criterion,
        "rubric": str(rubric_path),
        "criteria_split_on": criteria,
        "rows": per_row,
        "baseline_available": baseline is not None,
        "judged_pairs": pairs_block,
        "call": entry,
        "budget": ledger.closed_section(),
    }
    return ScoreReport(text=text, payload=payload, budget_line=ledger.budget_line())


def _baseline(
    request: ScoreRequest, settings: RunSettings, scorer: LLMRubricScorer, criterion: str
) -> dict[str, float | None] | None:
    """The parent filter's cached score on the same criterion, per row, or None.

    Read-only: the baseline's cells were paid for at the top of the iteration,
    and a miss is reported as None rather than bought. None as a whole when
    the parent rubric has no criterion of that name — a new criterion has no
    old score.
    """

    path = request.baseline_rubric
    if path is None:
        it_dir = _latest_iteration_dir(settings.run_dir, request.iteration)
        if it_dir is None:
            return None
        path = it_dir / "candidates" / f"iter{int(it_dir.name):03d}_baseline" / "rubric.md"
    path = Path(path)
    if not path.is_absolute():
        path = settings.run_dir / path
    if not path.is_file():
        return None
    rubric_md = path.read_text(encoding="utf-8")
    criteria, _ = _criteria_for(path, criterion)
    if criterion not in criteria:
        return None
    rows = load_trajectories(settings.train_data)
    by_id = {t.trajectory_id: t for t in rows}
    return _LazyBaseline(scorer, rubric_md, criteria, criterion, by_id)


class _LazyBaseline(dict):
    """A dict that fills itself from the cache on first read of each row.

    Lazy so the tool reads exactly the parent cells it reports and no more;
    a read is a cache hit or None, never a call.
    """

    def __init__(self, scorer, rubric_md, criteria, criterion, by_id):
        super().__init__()
        self._scorer = scorer
        self._rubric_md = rubric_md
        self._criteria = criteria
        self._criterion = criterion
        self._by_id = by_id

    def get(self, key, default=None):  # type: ignore[override]
        if key in self:
            return super().get(key, default)
        traj = self._by_id.get(key)
        if traj is None:
            return default
        cell = self._scorer.score_one(
            traj, self._criterion, self._criteria, self._rubric_md, cache_only=True
        )
        self[key] = cell.score
        return cell.score


def _pairs_block(run_dir: Path, per_row: list[dict[str, Any]]) -> dict[str, Any]:
    """How the new criterion alone orders the judged pairs among these rows.

    Only pairs among the requested rows that the judge has already ordered
    are counted; with ten rows that is usually a handful, and zero on
    iteration 0. It is a sanity check on direction, not a concordance.

    Both sides are keyed by the name the report shows, which is the
    pseudonym wherever the run has a map. `ranking_pairs.jsonl` is written
    redacted, so keying this on pool ids made the join silently empty and
    every pseudonymized run read "judged pairs among these rows: none yet".
    """

    scores = {r["trajectory_id"]: r["score"] for r in per_row if r.get("score") is not None}
    old = {
        r["trajectory_id"]: r["baseline_score"]
        for r in per_row
        if r.get("score") is not None and r.get("baseline_score") is not None
    }
    judged = [(hi, lo) for hi, lo in _judged_pairs(run_dir) if hi in scores and lo in scores]
    new_right = sum(1 for hi, lo in judged if scores[hi] > scores[lo])
    old_pairs = [(hi, lo) for hi, lo in judged if hi in old and lo in old]
    old_right = sum(1 for hi, lo in old_pairs if old[hi] > old[lo])
    return {
        "n_judged_among_rows": len(judged),
        "new_criterion_orders_correctly": new_right,
        "old_criterion_pairs": len(old_pairs),
        "old_criterion_orders_correctly": old_right,
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:+g}"


def _render_text(
    criterion: str,
    rubric_path: Path,
    per_row: list[dict[str, Any]],
    baseline: dict | None,
    criteria_note: str,
    pairs: dict[str, Any],
    entry: dict[str, Any],
) -> str:
    lines = [
        f"score-criterion: {criterion!r} on {len(per_row)} train row(s) from {rubric_path}",
        f"cells: {entry['cells_scored']} scored now, {entry['cells_cached']} already cached "
        f"(${entry['cost_usd']:.4f}, {entry['llm_calls']} call(s))",
    ]
    if criteria_note:
        lines.append(criteria_note)
    if baseline is None:
        lines.append("baseline: the parent rubric has no criterion of this name (new criterion)")
    for row in per_row:
        if "error" in row:
            lines.append(f"  {row['trajectory_id']}: UNSCORED — {row['error']}")
            continue
        old = row.get("baseline_score")
        delta = ""
        if old is not None and row["score"] is not None:
            delta = f" (delta {row['score'] - old:+g})"
        lines.append(
            f"  {row['trajectory_id']}: new {_fmt(row['score'])}"
            + (f" | old {_fmt(old)}{delta}" if baseline is not None else "")
            + (" [cached]" if row.get("cached") else "")
        )
        for ev in row.get("evidence") or []:
            lines.append(f"      evidence: {ev}")
    n = pairs["n_judged_among_rows"]
    if n:
        line = (
            f"judged pairs among these rows: {n} — the new criterion alone orders "
            f"{pairs['new_criterion_orders_correctly']}/{n} the judge's way"
        )
        if pairs["old_criterion_pairs"]:
            line += (
                f"; the old criterion alone {pairs['old_criterion_orders_correctly']}"
                f"/{pairs['old_criterion_pairs']}"
            )
        lines.append(line)
    else:
        lines.append("judged pairs among these rows: none yet")
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="score-criterion",
        description=(
            "Score one rubric criterion on a few TRAIN rows under the run's own scorer, "
            "within this proposer leg's dollar budget."
        ),
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="The run directory.")
    parser.add_argument(
        "--rubric",
        required=True,
        type=Path,
        help="A rubric.md — typically a candidate directory's; relative to --run-dir.",
    )
    parser.add_argument(
        "--criterion", required=True, help="The criterion's name, as in its heading."
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
        "--iteration",
        type=int,
        default=None,
        help="Iteration whose baseline rubric to compare against (default: the latest).",
    )
    parser.add_argument(
        "--baseline-rubric",
        type=Path,
        default=None,
        help="Rubric to read the criterion's OLD score from (default: the iteration's baseline).",
    )
    parser.add_argument(
        "--solver-provider",
        default=None,
        help="Override the recorded provider (offline checks only: `mock`).",
    )
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
    request = ScoreRequest(
        run_dir=args.run_dir,
        rubric=args.rubric,
        criterion=args.criterion,
        rows=args.rows.split(",") if args.rows else None,
        sample=args.sample if args.sample is not None else DEFAULT_SAMPLE,
        seed=args.seed,
        iteration=args.iteration,
        baseline_rubric=args.baseline_rubric,
        provider_override=args.solver_provider,
        repo_root=Path(args.repo_root),
    )
    try:
        report = score_criterion(request)
    except ScoreToolRefusal as exc:
        # A refusal before the ledger could be read (no config.yaml).
        print(str(exc))
        print(f"budget: spent $0.00 of ${DEFAULT_BUDGET_USD:.2f} this leg, remaining (unknown)")
        return EXIT_REFUSED
    print(report.render())
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
