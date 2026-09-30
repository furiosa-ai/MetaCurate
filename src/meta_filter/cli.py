"""Command-line entry point for meta-filter.

Usage
-----

    python -m meta_filter init
    python -m meta_filter run --config configs/metadata_free/nemotron_blank.yaml ...
    python -m meta_filter evaluate --filter filters/initial_blank_wired_v2 \\
        --data data/demo_validation.jsonl
    python -m meta_filter analyze-run runs/<run_id>

`scripts/run_metadata_free.py` builds the full `run` command line for each of
the four configured cells; the README has an offline demo on the mock
provider.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import typer
import yaml
from rich.console import Console

from meta_filter.agentic_proposer import AgenticProposer
from meta_filter.analysis.filter_evolution import analyze_run
from meta_filter.concurrency import DEFAULT_PER_CALL_GB, AdaptiveGovernor, log_plan
from meta_filter.constraints import FROZEN_FILES, parse_complexity_ceiling, parse_freeze
from meta_filter.critics import TrajectoryCritic
from meta_filter.critics.agentic import AgenticTrajectoryCritic
from meta_filter.filter_loader import load_filter
from meta_filter.llm import build_client
from meta_filter.llm.cost_meter import metered
from meta_filter.llm_scorer import LLMRubricScorer
from meta_filter.metadata_free import (
    filter_outcome_reads,
    pseudonymize_id,
    pseudonymize_selection_set,
    selection_set_view,
)
from meta_filter.metrics import DEFAULT_SCORE_WEIGHTS, compute_metrics
from meta_filter.optimizer import Optimizer, OptimizerConfig
from meta_filter.proposer import Proposer
from meta_filter.selection_set import ALL_PAIRS, PairCount, SelectionSet, assert_disjoint
from meta_filter.selector import Selector
from meta_filter.solver import HeuristicRubricScorer, Solver
from meta_filter.trajectory import load_trajectories
from meta_filter.validator import Validator

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()


DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_INITIAL_FILTER = Path("filters/initial_blank_wired_v2")

# Providers whose client accepts an `extra_body` passthrough. Only the
# OpenAI-compatible one does: the Claude and Codex CLIs build their own
# request, and the mock builds none.
_EXTRA_BODY_PROVIDERS = {"openai"}

# Providers whose client is a coding harness with tools (`with_workspace`,
# `allowed_tools`, an audit log). Only these can run the tool-using proposer.
_AGENTIC_PROVIDERS = {"claude-code", "claude_code", "claude-cli", "codex", "codex-cli", "codex_cli"}


def _extra_body_kwargs(raw: str | None, provider: str, flag: str) -> dict:
    """Parse one `--*-extra-body` flag into client kwargs.

    Both failure modes exit rather than warn. Bad JSON is obvious; the
    quieter one is a pin aimed at a provider that cannot apply it, which
    would leave the run unpinned while its launch line says otherwise.
    """

    if not raw:
        return {}
    if provider not in _EXTRA_BODY_PROVIDERS:
        typer.echo(
            f"{flag} needs an OpenAI-compatible provider; this role runs on "
            f"{provider!r}, which builds its own request body and would "
            f"ignore it.",
            err=True,
        )
        raise typer.Exit(2)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        typer.echo(f"{flag} is not valid JSON: {exc}", err=True)
        raise typer.Exit(2) from exc
    if not isinstance(body, dict):
        typer.echo(f"{flag} must be a JSON object, got {type(body).__name__}", err=True)
        raise typer.Exit(2)
    return {"extra_body": body}


def resolve_agentic(agentic: bool | None, llm_provider: str) -> bool:
    """Default to the tool-using proposer wherever the provider has tools.

    Omitting the flag must not quietly select the static proposer on a
    harness provider: the static path sees only its prompt, so a run launched
    that way is a different experiment from one launched with tools.
    """

    if agentic is None:
        return llm_provider.lower() in _AGENTIC_PROVIDERS
    return agentic


def _parse_selection_pairs(raw: str | None) -> PairCount:
    """`--selection-pairs`: unset is 500, 'all' is every pair, else a positive int."""

    if raw is None:
        return 500
    text = raw.strip().lower()
    if text == ALL_PAIRS:
        return ALL_PAIRS
    try:
        value = int(text)
    except ValueError:
        raise typer.BadParameter(
            f"--selection-pairs must be a positive integer or 'all', got {raw!r}",
            param_hint="--selection-pairs",
        ) from None
    if value < 1:
        raise typer.BadParameter(
            f"--selection-pairs must be positive, got {value}", param_hint="--selection-pairs"
        )
    return value


def _trajectory_ids(path: Path) -> list[str]:
    """The `trajectory_id` of every row in a JSONL pool, without building models.

    A boot-time disjointness check over three pools does not need the turns
    parsed, and selection rows can be large.
    """

    ids: list[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            ids.append(str(json.loads(line)["trajectory_id"]))
    return ids


def _load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------


@app.command()
def init(
    target: Path = typer.Argument(
        DEFAULT_INITIAL_FILTER.parent / "initial",
        help="Where to copy the blank initial filter.",
    ),
) -> None:
    """Copy the blank initial filter into `target` (idempotent)."""

    src = Path(__file__).resolve().parents[2] / DEFAULT_INITIAL_FILTER
    if not src.exists():
        typer.echo(f"reference filter missing at {src}", err=True)
        raise typer.Exit(2)
    if target.exists():
        typer.echo(f"{target} already exists — nothing to do.")
        return
    shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__"))
    typer.echo(f"initial filter installed at {target}")


@app.command()
def run(
    config: Path = typer.Option(DEFAULT_CONFIG, help="YAML config file."),
    train: Path = typer.Option(None, help="Training JSONL trajectories."),
    validation: Path = typer.Option(None, help="Held-out JSONL trajectories."),
    probe: Path = typer.Option(
        None,
        help="Fixed probe JSONL: the selected filter runs on it every iteration "
        "for drift monitoring only.",
    ),
    iterations: int = typer.Option(None, help="Number of iterations."),
    candidates: int = typer.Option(None, help="Candidates per iteration."),
    llm_provider: str = typer.Option(
        None,
        "--llm-provider",
        help="Proposer provider: mock | openai | anthropic | codex | claude-code.",
    ),
    llm_model: str = typer.Option(None, help="Override the proposer's model id."),
    proposer_extra_body: str = typer.Option(
        None,
        help=(
            "JSON merged into every proposer request body (--llm-provider "
            "openai only). Same shape as --solver-extra-body."
        ),
    ),
    initial_filter: Path = typer.Option(None, help="Directory of the starting filter."),
    runs_dir: Path = typer.Option(None, help="Root directory for run artifacts."),
    memory_dir: Path = typer.Option(None, help="Root directory for persistent regression memory."),
    run_id: str = typer.Option(None, help="Override the auto-generated run id."),
    seed: int = typer.Option(None, help="Random seed."),
    agentic: bool | None = typer.Option(
        None,
        "--agentic/--static",
        help=(
            "Give the proposer and the trajectory critic tools and a working "
            "directory instead of a fixed prompt. The solver stays static. "
            "Defaults to agentic under a harness provider (codex, claude-code) "
            "and to static everywhere else, since no other provider exposes tools."
        ),
    ),
    judge_provider: str = typer.Option(
        None, help="Provider for the trajectory critic and pair judge. Defaults to --llm-provider."
    ),
    judge_model: str = typer.Option(
        None,
        help=(
            "Model id for the trajectory critic and pair judge. Defaults to the "
            "strongest tier: every filter is measured against these verdicts, "
            "while the solver only applies the rubric it is handed."
        ),
    ),
    judge_extra_body: str = typer.Option(
        None,
        help=(
            "JSON merged into every judge request body (--judge-provider "
            "openai only). Covers the pair judge too, which reuses the judge "
            "client. Same shape as --solver-extra-body."
        ),
    ),
    judge_cache_dir: Path = typer.Option(
        Path("data/judge_cache"),
        help="Where to cache the trajectory critic's verdicts.",
    ),
    pair_judge_cache_dir: Path = typer.Option(
        Path("data/pair_judge_cache"),
        help="Where to cache pairwise judge verdicts.",
    ),
    judge_parallel: int = typer.Option(8, help="Parallel worker count for judge calls."),
    pair_budget: int = typer.Option(48, help="Disputed pairs adjudicated per iteration."),
    score_weight: list[str] = typer.Option(
        [],
        "--score-weight",
        help=(
            "Override one term of the selection objective, as NAME=VALUE, "
            "repeatable. e.g. --score-weight lambda_complexity=0 removes the "
            "complexity penalty. Runs that disagree on these are not "
            "comparable."
        ),
    ),
    max_pairs: int = typer.Option(
        4000,
        help=(
            "How many of the pool's trajectory pairs are classified into "
            "consensus/disagreement. Classification is free (sign arithmetic "
            "over scores already paid for); the cap bounds the disagreement "
            "pool the judge budget samples from."
        ),
    ),
    consensus_audit_share: float = typer.Option(
        0.0,
        help=(
            "Share of the judge budget spent auditing pairs the candidates "
            "already agree on. Above 0 the judge checks consensus every "
            "iteration, which is the only way a committee that ranks the pool "
            "backwards in unison can be detected: it produces no disagreement."
        ),
    ),
    exclude_consensus: bool = typer.Option(
        False,
        "--exclude-consensus",
        help=(
            "Drop candidate-consensus pairs from the truth set, leaving only "
            "judge-resolved pairs. Consensus pairs are identical for every "
            "candidate, so they carry no selection signal while making up most "
            "of the truth set and rewarding conformity. Raise --pair-budget "
            "with this: the truth set shrinks to what you pay for."
        ),
    ),
    pair_evidence: int = typer.Option(
        0,
        help=(
            "Show the next proposer this many consensus-audit pairs the winner "
            "mis-ordered, with the judge's rationale: pairs every candidate "
            "ordered the same wrong way, which the misordered-pair blocks rank "
            "last and a disagreement-driven sample never surfaces. "
            "Deduplicated against those blocks. 0 = aggregates only."
        ),
    ),
    redundancy_penalty: float = typer.Option(
        0.5,
        help=(
            "Reported penalty for inter-criterion redundancy: ramps linearly "
            "from 0 at --redundancy-threshold to this value at |Spearman|=1 "
            "between the candidate's two most-correlated live criteria. It is "
            "charged to the selected filter's reported score_with_redundancy "
            "and shown to the proposer, never to the score selection sorts on."
        ),
    ),
    redundancy_threshold: float = typer.Option(
        0.7, help="|Spearman| where the redundancy penalty starts."
    ),
    min_distinct_levels: int = typer.Option(
        None,
        "--min-distinct-levels",
        help=(
            "A candidate emitting fewer distinct score values than this cannot "
            "express an ordering and has its concordance zeroed. Unset scales "
            "with the pool: max(10, 0.4 * n_train)."
        ),
    ),
    selection_data: Path = typer.Option(
        ...,
        "--selection-data",
        help=(
            "JSONL of hidden selection rows, disjoint from --train and --probe. "
            "Every candidate is scored on them each iteration and a paired "
            "McNemar gate on judged pairs over them decides acceptance. The "
            "proposer never sees them."
        ),
    ),
    selection_pairs: str = typer.Option(
        None,
        "--selection-pairs",
        help=(
            "Without --selection-set: pairs planned over the selection rows at "
            "run start (half from the initial filter's hard quartile), default "
            "500; or 'all' for every unordered pair, no stratification."
        ),
    ),
    selection_set: Path = typer.Option(
        None,
        "--selection-set",
        help=(
            "A prebuilt, already-judged selection set (scripts.build_selection_set) "
            "copied into runs/_selection/<run_id>.json instead of planning and "
            "judging pairs at start, so every run pointed at it is graded on the "
            "same pairs. Its rows must all be in --selection-data."
        ),
    ),
    selection_seed: int = typer.Option(
        4242, help="Seed for the selection pair plan; same rows + seed = same exam."
    ),
    gate_z: float = typer.Option(
        None,
        "--gate-z",
        help=(
            "Per-comparison z a candidate must clear on the paired test. Unset "
            "derives it from the number of scored candidates (~2.43 for three, "
            "Sidak-corrected one-sided 2.0)."
        ),
    ),
    gate_min_discordant: int = typer.Option(
        5, help="Fewer discordant selection pairs than this and the candidate cannot pass the gate."
    ),
    patience: int = typer.Option(
        None,
        "--patience",
        help=(
            "Stop after this many consecutive iterations that scored a "
            "candidate and kept the baseline (proposer failures do not count). "
            "Unset runs every iteration."
        ),
    ),
    min_iterations: int = typer.Option(
        4, help="Iterations that must run before --patience may stop the run."
    ),
    candidates_scored: int = typer.Option(
        None,
        "--candidates-scored",
        help=(
            "Of the --candidates the proposer writes, only the top N by "
            "precheck score on the already-judged pairs get the full train + "
            "selection solver pass; the rest are recorded as gated_out in the "
            "iteration's precheck.json. Unset scores every proposed candidate."
        ),
    ),
    proposer_score_budget: float = typer.Option(
        1.50,
        "--proposer-score-budget",
        help=(
            "Tool-using proposer with the LLM solver: dollars per proposer leg "
            "the criterion-scoring tool (meta_filter.score_tool) may spend "
            "scoring a drafted criterion on train rows under the run's own "
            "scorer. Folded into cost_proposer; the ledger is "
            "<run>/proposer_score_budget.json."
        ),
    ),
    complexity_ceiling: str = typer.Option(
        None,
        "--complexity-ceiling",
        help=(
            "The run hard cap as numbers, e.g. "
            "'rubric_criteria=9,rule_count=13,rules_loc=522,rubric_chars=8785': the "
            "largest count a candidate may reach per component over the run. "
            "Unset, the cap is 1.5x the initial filter's size. A component left "
            "out is uncapped."
        ),
    ),
    freeze: str = typer.Option(
        None,
        "--freeze",
        help=(
            "Comma-separated files a candidate may not change: rubric, rules, "
            "policy (e.g. --freeze rules,policy leaves only rubric.md editable). "
            "A changed frozen file is invalid at precheck, whatever else the "
            "candidate does."
        ),
    ),
    solver_backend: str = typer.Option(
        "heuristic",
        help="heuristic | llm — 'llm' lets a model read rubric.md and score each trajectory.",
    ),
    solver_provider: str = typer.Option(
        None, help="Provider for the LLM solver. Defaults to --llm-provider."
    ),
    solver_model: str = typer.Option(None, help="Model id for the LLM solver."),
    solver_extra_body: str = typer.Option(
        None,
        help=(
            "JSON merged into every solver request body (--solver-provider "
            "openai only). This is where OpenRouter's non-OpenAI controls go: "
            '\'{"provider": {"order": ["MiniMax"], "allow_fallbacks": false, '
            '"quantizations": ["fp8"]}, "reasoning": {"enabled": false}}\' '
            "pins the serving provider and turns thinking off. Unpinned, one "
            "model slug is whichever provider and quantization the router "
            "picked that minute."
        ),
    ),
    solver_cache_dir: Path = typer.Option(
        Path("data/scorer_cache"),
        help="Where the LLM solver caches rubric scores by (trajectory, rubric hash).",
    ),
    solver_parallel: int = typer.Option(
        8, help="Parallel worker count for the LLM solver prefetch."
    ),
    per_call_gb: float = typer.Option(
        DEFAULT_PER_CALL_GB,
        help="Assumed peak memory of one headless LLM call, in GB. The "
        "concurrency governor divides the available memory by this to cap "
        "candidates x solver workers. Raise it for pools with large "
        "trajectories; lower it only with measurements in hand.",
    ),
    memory_share: float = typer.Option(
        1.0,
        help="Fraction of the memory budget this run may claim. Pass 0.5 to "
        "each of two runs sharing a machine: the governor's startup snapshot "
        "cannot see a sibling that launched at the same moment.",
    ),
    lambda_review: float = typer.Option(
        None,
        "--lambda-review",
        help="Override the review penalty in the reported scalar loss.",
    ),
    rule_critic_recommendations: bool = typer.Option(
        False,
        "--rule-critic-recommendations",
        help=(
            "Have RuleCritic explain its per-rule numbers with keep / "
            "move_to_rubric / split / drop recommendations. Costs one cached "
            "LLM call per candidate; does not affect rule_quality or the score."
        ),
    ),
    rule_design_review: bool = typer.Option(
        False,
        "--rule-design-review",
        help=(
            "Let an LLM score the rule set's design (coverage, precision "
            "discipline, factoring, division of labour, robustness) and fold "
            "that into rule_quality. It moves the score only when "
            "lambda_rule_quality is non-zero."
        ),
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Continue an existing --run-id instead of starting at iteration 0. "
        "Picks up from the last iteration that wrote a summary row, reusing the "
        "filter that iteration was working from and replaying the proposer's "
        "own past diagnoses. --iterations is the target total, not a count to add.",
    ),
    train_sample_schedule: str = typer.Option(
        "",
        help="Comma-separated per-iteration train sample rates (e.g. '0.3,0.6,1.0'). "
        "Empty = use full pool every iteration.",
    ),
    candidates_parallel: int = typer.Option(
        0,
        help="Number of candidates to evaluate concurrently (prefetch + solver.run). "
        "0 = one worker per candidate. 1 forces the serial flow. Concurrency "
        "multiplies with --solver-parallel, so watch your rate limit.",
    ),
    rejudge_selection: bool = typer.Option(
        False,
        "--rejudge-selection",
        help=(
            "Instead of refusing a --selection-set whose provenance does not "
            "show it was judged on stripped rows, re-judge its pairs on the "
            "stripped rows at run start (paid)."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Execute the optimization loop end-to-end.

    Every pool (train, validation, probe, selection) is stripped at load to
    the transcript, the submitted diff and a few process counts
    (`meta_filter.metadata_free`); no component of the loop sees a harness
    outcome.
    """

    _configure_logging(verbose)
    cfg = _load_config(config)
    paths = cfg.get("paths", {})
    llm_cfg = cfg.get("llm", {})
    v_cfg = cfg.get("validator", {})
    b_cfg = cfg.get("boundary", {})

    train_path = train or Path(paths.get("train_data", "data/demo_train.jsonl"))
    validation_path = validation or (
        Path(paths.get("validation_data", "data/demo_validation.jsonl"))
        if paths.get("validation_data")
        else None
    )
    iterations = iterations if iterations is not None else int(cfg.get("iterations", 3))
    candidates = (
        candidates if candidates is not None else int(cfg.get("candidates_per_iteration", 3))
    )
    llm_provider = llm_provider or llm_cfg.get("provider", "mock")
    llm_model = llm_model or llm_cfg.get("model")
    seed = seed if seed is not None else int(cfg.get("seed", 17))
    initial_filter = initial_filter or Path(
        paths.get("initial_filter", str(DEFAULT_INITIAL_FILTER))
    )
    runs_dir = runs_dir or Path(paths.get("runs_dir", "runs"))
    memory_dir = memory_dir or Path(paths.get("memory_dir", "memory"))

    loss_weights = v_cfg.get("loss_weights", {}) or {}
    boundary_confidence = float(b_cfg.get("confidence_threshold", 0.55))
    boundary_margin = float(b_cfg.get("score_margin", 1.0))

    client_kwargs: dict = {}
    if llm_model:
        client_kwargs["model"] = llm_model
    if llm_provider == "mock":
        client_kwargs.setdefault("seed", seed)
    client_kwargs.update(
        _extra_body_kwargs(proposer_extra_body, llm_provider, "--proposer-extra-body")
    )
    # Metered per role so `summary.csv` can say what the proposer cost as
    # distinct from the scoring calls around it. The wrapper delegates
    # everything else, so the tool-using proposer still sees its harness
    # client.
    llm_client = metered(build_client(llm_provider, **client_kwargs), "proposer")

    # Tools only exist behind a coding harness (the Claude Code and Codex
    # CLIs); every other provider is a plain completion endpoint.
    agentic = resolve_agentic(agentic, llm_provider)
    if agentic and llm_provider.lower() not in _AGENTIC_PROVIDERS:
        typer.echo(
            f"--agentic requires a harness provider ({', '.join(sorted(_AGENTIC_PROVIDERS))})",
            err=True,
        )
        raise typer.Exit(2)
    if not agentic and llm_provider.lower() in _AGENTIC_PROVIDERS:
        console.print(
            "[bold yellow]warning:[/bold yellow] --static on a harness provider: the "
            "proposer and the trajectory critic answer from an excerpt-based prompt "
            "and cannot read the run directory."
        )

    proposer_cls = AgenticProposer if agentic else Proposer
    proposer = proposer_cls(
        client=llm_client,
        n_candidates=candidates,
        temperature=float(llm_cfg.get("temperature", 0.4)),
    )
    if agentic:
        console.print("[bold cyan]proposer:[/bold cyan] agentic (reads the run dir with tools)")
    # Every value below is checked before anything is built, so a bad flag
    # fails in seconds rather than after the first proposer call.
    parsed_score_weights: dict[str, float] = {}
    for item in score_weight:
        name, sep, value = item.partition("=")
        if not sep:
            raise typer.BadParameter(
                f"--score-weight takes NAME=VALUE, got {item!r}", param_hint="--score-weight"
            )
        name = name.strip()
        if name not in DEFAULT_SCORE_WEIGHTS:
            # Refused rather than ignored: a misspelled term would otherwise
            # leave the run with the weight it was launched to change.
            raise typer.BadParameter(
                f"--score-weight {name!r} is not a term of the objective. "
                f"Known: {', '.join(sorted(DEFAULT_SCORE_WEIGHTS))}",
                param_hint="--score-weight",
            )
        try:
            parsed_score_weights[name] = float(value)
        except ValueError:
            raise typer.BadParameter(
                f"--score-weight {name} needs a number, got {value!r}",
                param_hint="--score-weight",
            ) from None
    if proposer_score_budget < 0:
        raise typer.BadParameter(
            f"--proposer-score-budget must be non-negative, got {proposer_score_budget}",
            param_hint="--proposer-score-budget",
        )
    seed_reads = filter_outcome_reads(Path(initial_filter))
    if seed_reads:
        raise typer.BadParameter(
            f"refusing {initial_filter}: "
            + "; ".join(f"{name} names {', '.join(v)}" for name, v in seed_reads.items())
            + " (the precheck would refuse every candidate copied from it)",
            param_hint="--initial-filter",
        )
    try:
        parsed_ceiling = parse_complexity_ceiling(complexity_ceiling)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--complexity-ceiling") from None
    try:
        frozen_files = parse_freeze(freeze)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--freeze") from None
    if not Path(selection_data).is_file():
        raise typer.BadParameter(
            f"--selection-data {selection_data} does not exist",
            param_hint="--selection-data",
        )
    # Disjointness is the whole point of a hidden set. Checked on ids read
    # straight off the files, before any model is built.
    selection_ids = _trajectory_ids(Path(selection_data))
    try:
        assert_disjoint(
            selection_ids,
            train_ids=_trajectory_ids(Path(train_path)),
            probe_ids=_trajectory_ids(Path(probe)) if probe else (),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--selection-data") from None
    if selection_set is not None:
        if not Path(selection_set).is_file():
            raise typer.BadParameter(
                f"--selection-set {selection_set} does not exist",
                param_hint="--selection-set",
            )
        # The run sees pseudonymized ids; compare in that namespace, whichever
        # one the file was written in.
        prebuilt = pseudonymize_selection_set(SelectionSet.load(selection_set))
        known_ids = {pseudonymize_id(i) for i in selection_ids}
        missing = sorted((set(prebuilt.row_ids) | prebuilt.pair_ids()) - known_ids)
        if missing:
            raise typer.BadParameter(
                f"--selection-set {selection_set} names {len(missing)} row(s) not in "
                f"--selection-data {selection_data}: {', '.join(missing[:10])}",
                param_hint="--selection-set",
            )
        why = selection_set_view(prebuilt.provenance)
        if why is not None and not rejudge_selection:
            raise typer.BadParameter(
                f"refusing --selection-set {selection_set}: {why}. "
                "Build one with `scripts.build_selection_set`, or pass "
                "--rejudge-selection to re-judge its pairs at run start",
                param_hint="--selection-set",
            )
        if why is not None:
            console.print(
                f"[bold yellow]selection set:[/bold yellow] {selection_set} does not show "
                "it was judged on stripped rows; its pairs will be re-judged"
            )
        if selection_pairs is not None:
            console.print(
                f"[bold yellow]warning:[/bold yellow] --selection-pairs {selection_pairs} "
                f"is ignored: --selection-set {selection_set} fixes the pairs"
            )
    parsed_selection_pairs = _parse_selection_pairs(selection_pairs)
    if patience is not None and patience < 1:
        raise typer.BadParameter(
            f"--patience must be positive, got {patience}", param_hint="--patience"
        )
    if min_iterations < 0:
        raise typer.BadParameter(
            f"--min-iterations must be non-negative, got {min_iterations}",
            param_hint="--min-iterations",
        )
    # Only when patience is armed: `min_iterations` means nothing without it,
    # and its default (4) is above the default iteration count (3), so an
    # unconditional check would reject every short run.
    if patience is not None and min_iterations > iterations:
        raise typer.BadParameter(
            f"--min-iterations {min_iterations} exceeds --iterations {iterations}: the "
            "early stop could never fire and the floor is unsatisfiable",
            param_hint="--min-iterations",
        )
    if candidates_scored is not None and not 1 <= candidates_scored <= candidates:
        raise typer.BadParameter(
            f"--candidates-scored must be between 1 and --candidates "
            f"({candidates}), got {candidates_scored}",
            param_hint="--candidates-scored",
        )

    # The governor re-plans every iteration inside the Optimizer, because the
    # prompt carries the whole rubric and a filter that grows changes the
    # memory one call needs. Build it here, let the Optimizer drive it.
    governor = AdaptiveGovernor(
        per_call_gb=per_call_gb,
        memory_share=memory_share,
    )
    log_plan(
        governor.plan(
            candidates=candidates_parallel or candidates,
            solver=solver_parallel,
            judge=judge_parallel,
        )
    )

    if solver_backend == "llm":
        s_provider = solver_provider or llm_provider
        s_kwargs: dict = {}
        if solver_model:
            s_kwargs["model"] = solver_model
        if s_provider == "mock":
            s_kwargs.setdefault("seed", seed)
        s_kwargs.update(_extra_body_kwargs(solver_extra_body, s_provider, "--solver-extra-body"))
        solver_client = metered(build_client(s_provider, **s_kwargs), "solver")
        rubric_scorer = LLMRubricScorer(
            client=solver_client,
            cache_dir=solver_cache_dir,
            max_workers=solver_parallel,
        )
        console.print(
            f"[bold cyan]solver rubric scorer:[/bold cyan] LLM "
            f"({s_provider} / {getattr(solver_client, 'model', '?')}) "
            f"cache={solver_cache_dir}"
        )
    else:
        rubric_scorer = HeuristicRubricScorer()
        console.print("[bold cyan]solver rubric scorer:[/bold cyan] heuristic")
    solver = Solver(rubric_scorer=rubric_scorer)

    if lambda_review is not None:
        loss_weights["lambda_review"] = lambda_review

    validator = Validator(
        loss_weights={
            "lambda_fn": float(loss_weights.get("lambda_fn", 5.0)),
            "lambda_fp": float(loss_weights.get("lambda_fp", 4.0)),
            "lambda_review": float(loss_weights.get("lambda_review", 0.5)),
            "lambda_cost": float(loss_weights.get("lambda_cost", 0.1)),
            "lambda_complexity": float(loss_weights.get("lambda_complexity", 0.05)),
        },
        boundary_confidence=boundary_confidence,
        boundary_margin=boundary_margin,
    )
    selector = Selector()

    j_provider = judge_provider or llm_provider
    j_kwargs: dict = {}
    if judge_model:
        j_kwargs["model"] = judge_model
    elif j_provider in ("claude-code", "claude_code", "claude-cli"):
        # The client default is sonnet, which is right for the solver and
        # wrong here: every filter in the run is scored against these
        # verdicts, so a cheap judge is a cheap ceiling on the whole run.
        j_kwargs["model"] = "opus"
    if j_provider == "mock":
        j_kwargs.setdefault("seed", seed)
    j_kwargs.update(_extra_body_kwargs(judge_extra_body, j_provider, "--judge-extra-body"))
    # The pair judge reuses this client (`optimizer.py` builds PairJudge from
    # `backend.client`), so one flag and one meter cover both the
    # per-trajectory critic and the pair verdicts.
    judge_client = metered(build_client(j_provider, **j_kwargs), "judge")
    agentic_judge = agentic and j_provider == "claude-code"
    judge_cls = AgenticTrajectoryCritic if agentic_judge else TrajectoryCritic
    validator.critic_backend = judge_cls(
        client=judge_client,
        cache_dir=judge_cache_dir,
        model_tag=getattr(judge_client, "model", "unknown"),
    )
    console.print(
        f"[bold cyan]critic backend:[/bold cyan] "
        f"{'agentic ' if agentic_judge else ''}LLM judge "
        f"({j_provider} / {getattr(judge_client, 'model', '?')}) "
        f"cache={judge_cache_dir}"
    )

    schedule_parsed: list[float] | None = None
    if train_sample_schedule:
        schedule_parsed = [float(x) for x in train_sample_schedule.split(",") if x.strip()]

    opt_cfg = OptimizerConfig(
        iterations=iterations,
        candidates_per_iteration=candidates,
        seed=seed,
        loss_weights=validator.loss_weights,
        boundary_confidence=boundary_confidence,
        boundary_margin=boundary_margin,
        judge_parallel=judge_parallel,
        pair_budget=pair_budget,
        max_pairs=max_pairs,
        truth_includes_consensus=not exclude_consensus,
        consensus_audit_share=consensus_audit_share,
        score_weights=parsed_score_weights,
        pair_evidence=pair_evidence,
        redundancy_penalty=redundancy_penalty,
        redundancy_threshold=redundancy_threshold,
        min_distinct_levels=min_distinct_levels,
        train_sample_schedule=schedule_parsed,
        candidates_parallel=candidates_parallel,
        agentic_critics=agentic,
        rule_critic_llm_recommendations=rule_critic_recommendations,
        rule_design_review=rule_design_review,
        resume=resume,
        pair_judge_cache_dir=pair_judge_cache_dir,
        selection_data=selection_data,
        selection_pairs=parsed_selection_pairs,
        selection_seed=selection_seed,
        selection_set=selection_set,
        gate_z=gate_z,
        gate_min_discordant=gate_min_discordant,
        patience=patience,
        min_iterations=min_iterations,
        candidates_scored=candidates_scored,
        proposer_score_budget=proposer_score_budget,
        complexity_ceiling=parsed_ceiling,
        freeze=frozen_files,
        rejudge_selection=rejudge_selection,
    )
    if frozen_files:
        console.print(
            "[bold cyan]frozen:[/bold cyan] "
            + ", ".join(FROZEN_FILES[name] for name in frozen_files)
            + " — a candidate that changes them is invalid at precheck"
        )
    console.print(
        "[bold cyan]pools:[/bold cyan] stripped at load to the transcript, the submitted "
        "diff and process counts; a candidate that names an outcome field in rules.py / "
        "policy.py / rubric.md is invalid at precheck"
    )
    if parsed_ceiling:
        console.print(
            "[bold cyan]hard cap:[/bold cyan] "
            + ", ".join(f"{k} <= {v}" for k, v in parsed_ceiling.items())
            + " — soft caps and the deletion floor still read the initial filter"
        )
    if candidates_scored is not None:
        console.print(
            f"[bold cyan]candidate gate:[/bold cyan] {candidates} proposed, top "
            f"{candidates_scored} by precheck score fully scored"
        )
    pairs_note = (
        f"prebuilt pairs from {selection_set}"
        if selection_set is not None
        else (
            "every pair"
            if parsed_selection_pairs == ALL_PAIRS
            else f"{parsed_selection_pairs} pairs, seed {selection_seed}"
        )
    )
    console.print(
        f"[bold cyan]selection set:[/bold cyan] {selection_data} — {pairs_note}; "
        f"paired gate z={'auto' if gate_z is None else gate_z}, "
        f"min discordant {gate_min_discordant}"
    )
    if patience is not None:
        console.print(
            f"[bold cyan]early stop:[/bold cyan] patience {patience} after "
            f"{min_iterations} iteration(s)"
        )

    optimizer = Optimizer(
        initial_filter_dir=initial_filter,
        train_data=train_path,
        validation_data=validation_path,
        runs_dir=runs_dir,
        memory_dir=memory_dir,
        proposer=proposer,
        selector=selector,
        solver=solver,
        validator=validator,
        config=opt_cfg,
        run_id=run_id,
        governor=governor,
        requested_solver_parallel=solver_parallel,
        probe_data=probe,
    )
    run_path = optimizer.run()
    console.print(f"[bold green]run complete:[/bold green] {run_path}")


@app.command()
def evaluate(
    filter: Path = typer.Option(..., help="Filter directory to evaluate."),  # noqa: A002
    data: Path = typer.Option(..., help="JSONL trajectory file."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run one filter over a trajectory file with the heuristic scorer; print metrics as JSON.

    The rows are stripped at load exactly as `run` strips them. No critic is
    consulted, so the agreement counters are zero and the output reports the
    filter's own decision counts.
    """

    _configure_logging(verbose)
    loaded = load_filter(filter)
    trajectories = load_trajectories(data)
    solver = Solver(rubric_scorer=HeuristicRubricScorer())
    decisions = solver.run(loaded, trajectories)
    metrics = compute_metrics(trajectories, decisions, verdicts={})
    console.print_json(data=json.loads(metrics.model_dump_json()))


@app.command("analyze-run")
def analyze_run_cmd(
    run_dir: Path = typer.Argument(..., help="Run directory to analyze."),
) -> None:
    """Retrospectively analyze a completed run for structural changes."""

    report = analyze_run(run_dir)
    console.print(report)


if __name__ == "__main__":
    app()
