"""LLM-based rubric scorer.

Reads the actual `rubric.md` text (which the proposer is free to rewrite each
iteration) and asks a strong LLM to score every criterion on the trajectory.
This is what makes rubric edits *observable* in solver behavior — the
heuristic scorer ignores the rubric text and always applies the same
hard-coded logic, so proposer edits to `rubric.md` have no effect on the
decisions it drives. Swapping this class in fixes that.

Parallel prefetching is supported: the Solver runs scoring one trajectory at
a time by default, but callers can use `score_batch` (see below) to fan out
across trajectories with a thread pool.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from meta_filter.core.rendering import (
    DEFAULT_TRAJECTORY_TOOL_BUDGET,
    render_submission,
    render_turns,
    view_signature,
)
from meta_filter.core.schemas import Trajectory
from meta_filter.core.solver import ScorerCost
from meta_filter.llm.base import LLMClient
from meta_filter.visibility.view_policy import (
    SCORER_BASELINE_FIELDS,
    absent_field_signature,
    declared_fields,
    field_set_signature,
    fields_for,
    scorer_key_fields,
    scorer_metadata_block,
    scorer_reference_blocks,
)

# Named, not `__name__`: score_tool and rule_tool print WARNING records as
# "LEVEL name: message" to the sandboxed proposer's stderr, so this name is part
# of what the proposer reads. It is the module's name from before it moved into
# core/; tests/test_scorer_parse_failure.py filters on it.
logger = logging.getLogger("meta_filter.llm_scorer")

# Attempts at one criterion before the row is failed. The prompt is identical
# every time: at temperature 0 a retry re-rolls the sampler and nothing else,
# which is what a truncated or fence-wrapped answer needs and all a
# deterministic refusal will ever give back.
PARSE_ATTEMPTS = 3


@dataclass(frozen=True)
class ScoreFailure:
    """One (row, criterion) the model would not answer readably."""

    trajectory_id: str
    criterion: str
    attempts: int
    excerpt: str = ""

    def describe(self) -> str:
        return (
            f"{self.trajectory_id} / {self.criterion}: unreadable after "
            f"{self.attempts} attempts; last response began {self.excerpt!r}"
        )


class ScorerParseError(RuntimeError):
    """The scorer has no score for a row, and inventing one would be a lie.

    Raised rather than returned so the failure cannot be mistaken for a
    verdict anywhere downstream. The Solver already contains a per-row scorer
    failure (`solver.py`: route to REVIEW, leave `rank_score` None), which is
    exactly the treatment this needs — a row with no `rank_score` is absent
    from the candidate's score map and therefore out of the ranking exam.
    Turning an unparseable answer into 0.0 instead would put a real point on
    the -2..+2 scale in the cache under the criterion's key, served for the
    rest of the run.
    """

    def __init__(self, failure: ScoreFailure, cost: ScorerCost | None = None) -> None:
        super().__init__(failure.describe())
        self.failure = failure
        # What the failed attempts still cost. The caller cannot get a score
        # out of them but should not lose the spend from its accounting.
        self.cost = cost or ScorerCost()


@dataclass(frozen=True)
class CriterionCell:
    """One scored (trajectory, criterion) cell and where it came from.

    `cached` says whether the value was served from the per-criterion cache
    or bought with a call; `score` is None only under `cache_only`, when
    nothing was cached and nothing was asked.
    """

    score: float | None
    evidence: list[str]
    cost: ScorerCost
    cached: bool


SCORER_SYSTEM = (
    "You are a strict evaluator scoring an agent trajectory against a rubric. "
    "You must honor the rubric text verbatim — do NOT invent new criteria. "
    "Output exactly one JSON object with the required schema."
)


SCORER_PROMPT = """\
Score the following agent trajectory using the rubric BELOW. Each criterion
gets an integer score on the scale +2 / +1 / 0 / -1 / -2 and one short line of
concrete evidence from the trajectory.

### RUBRIC (verbatim)
{rubric_md}

### TRAJECTORY

Task:
{task}

Metadata:
{metadata}

Action histogram (top actions; navigation actions like `cd`/`ls` are
bookkeeping and rarely worth penalizing directly):
{histogram}

Full transcript. Agent reasoning and commands are verbatim; long tool output
is elided in the middle, keeping the head (what ran) and the tail (how it
ended):
{turns_block}

Final submitted patch:
{submission}{reference_blocks}

### OUTPUT FORMAT

Return exactly this JSON. `scores` MUST contain one entry per criterion listed
below, using the criterion's title text as the key.

Criteria to score (in order): {criteria_list}
{scope_note}
{{
  "scores": {{
    "<Criterion 1 Title>": {{"score": -2..2, "evidence": "..."}},
    ...
  }},
  "summary": "one sentence overall assessment"
}}"""


def _histogram(traj: Trajectory) -> dict[str, int]:
    from collections import Counter

    c: Counter[str] = Counter()
    for t in traj.turns:
        key = (t.action or t.role or "").strip()
        if key:
            c[key] += 1
    return dict(c.most_common(10))


def _split_rubric_sections(rubric_md: str, criteria: list[str]) -> dict[str, str]:
    """Carve the rubric into one text block per criterion.

    A criterion's section starts at the first line that contains the criterion
    name (case-insensitive) — typically its markdown heading — and runs to the
    next line where any *other* criterion's section starts, or to the end of
    the document. A criterion whose name appears nowhere falls back to the
    ENTIRE rubric text: that over-invalidates its cache on every rubric edit,
    which is safe, instead of ever serving a stale score.
    """

    lines = rubric_md.splitlines()
    starts: dict[str, int | None] = {}
    for criterion in criteria:
        needle = criterion.lower()
        starts[criterion] = next(
            (i for i, line in enumerate(lines) if needle in line.lower()),
            None,
        )
    boundaries = sorted({i for i in starts.values() if i is not None})
    sections: dict[str, str] = {}
    for criterion in criteria:
        start = starts[criterion]
        if start is None:
            sections[criterion] = rubric_md
            continue
        end = next((b for b in boundaries if b > start), len(lines))
        # rstrip so the blank line separating this section from the next one
        # (or from a newly appended one) is not part of the section identity:
        # adding a criterion below must not invalidate the criterion above.
        sections[criterion] = "\n".join(lines[start:end]).rstrip()
    return sections


def _rubric_header(rubric_md: str, sections: dict[str, str]) -> str:
    """Everything before the first criterion section.

    The preamble is not decoration: it defines the +2/-2 scale, the "doubt is
    0" rule and the navigation-actions instruction. A criterion scored without
    it is scored against a different instrument, so it goes into every
    per-criterion prompt -- and therefore into every per-criterion key.

    That is also the limit on what per-criterion caching can save here. The
    proposer rewrites the preamble most iterations, and a preamble edit
    legitimately invalidates every criterion: measured over three 20-iteration
    runs, keying on (criterion, section) saves 83.5-85.7% of criterion
    scorings while keying on (criterion, section, header) saves 15.0-49.5%.
    Getting the rest would mean taking the scale definition away from the
    proposer and into the harness, which is a separate decision.
    """

    if not sections:
        return rubric_md
    starts = [rubric_md.find(text) for text in sections.values() if text and text in rubric_md]
    starts = [i for i in starts if i >= 0]
    return rubric_md[: min(starts)] if starts else rubric_md


def _criterion_prompt_rubric(header: str, section: str) -> str:
    """The rubric text one criterion is scored against: preamble + its own."""

    return f"{header.rstrip()}\n\n{section.strip()}\n" if header.strip() else f"{section.strip()}\n"


def _criterion_cache_key(
    section_text: str,
    criterion: str,
    trajectory: Trajectory,
    model: str = "",
    field_sig: str = "",
    header: str = "",
) -> str:
    """Cache key for one (trajectory, criterion) pair.

    Keyed on the criterion's *section* text, not the whole rubric, so editing
    one criterion leaves every other criterion's cached score untouched. The
    criterion name is hashed in as well because two criteria can share a
    section text (both fall back to the whole document when their headings are
    missing) and must still get distinct entries.

    `field_sig` carries the fields *this criterion* was rendered with, not the
    union over the rubric, which keeps the cost of widening one criterion
    proportional to that criterion: on the six-criterion `initial_ranking`,
    declaring `gold_patch` on the reference-reading axis re-scores 1 criterion
    over the pool instead of 6.

    `header` is the rubric preamble, and it is in the key because it is in the
    prompt. Without it the key did not determine the value: a score formed as
    "criterion 1 of 9" was filed under a key recording only criterion 1's
    section, so 21 of the 50 keys the 20 filters on disk generate were
    reachable from more than one filter -- one `Verification Completeness` key
    shared by nine of them, spanning a 4-criterion 2,724-character rubric to
    an 11-criterion 20,574-character one, first writer winning.

    Together with `score()` sending only (header + this section), the key now
    covers every input: trajectory, model, view, metadata fields, preamble and
    the criterion's own text.

    Every input is part of the prompt, and nothing else is in the key. There
    is deliberately no salt: filing a fresh draw beside the cached one would
    leave the unsalted key holding the draw that WON selection, which every
    rubric-identical candidate would then read while the baseline used the
    fresh one — pure scorer noise in `b − c`. `LLMRubricScorer.fresh_draw`
    overwrites in place instead, so there is exactly one value per key and no
    second key to diverge from it.
    """

    import hashlib

    sec_h = hashlib.sha1(section_text.encode("utf-8")).hexdigest()[:10]
    crit_h = hashlib.sha1(criterion.encode("utf-8")).hexdigest()[:8]
    tid = sanitize_trajectory_id(trajectory.trajectory_id)
    suffix = f"__hdr_{hashlib.sha1(header.encode('utf-8')).hexdigest()[:8]}" if header else ""
    suffix += f"__meta_{field_sig}" if field_sig else ""
    suffix += f"__model_{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}" if model else ""
    return (
        f"{tid}__crit_{crit_h}_{sec_h}"
        f"__view_{view_signature(DEFAULT_TRAJECTORY_TOOL_BUDGET)}{suffix}"
    )


# Separates the trajectory id from the rest of a per-criterion cache key.
_KEY_CRITERION_MARK = "__crit_"


def sanitize_trajectory_id(trajectory_id: str) -> str:
    """The trajectory id as it leads every per-criterion cache filename."""

    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", trajectory_id)[:150]


def cache_key_trajectory_id(key: str) -> str | None:
    """The sanitized trajectory id a per-criterion cache key (or filename) starts
    with, or None for a name that is not such a key.

    Two readers depend on the id being recoverable from the filename alone:
    the merge of the proposer's staged cells into the run cache, which may
    copy only train-row cells, and the read audit, which forbids a
    tool-using proposer the cells of the hidden selection rows. Neither
    may need to open the file to decide.
    """

    name = key[: -len(".json")] if key.endswith(".json") else key
    head, mark, _ = name.partition(_KEY_CRITERION_MARK)
    return head if mark and head else None


def criterion_field_sets(
    rubric_md: str,
    criteria: Sequence[str],
    role: str = "scorer",
) -> dict[str, frozenset[str]]:
    """`criterion -> the fields it is rendered with`, from its own declaration.

    A criterion that declares nothing gets the role's baseline, so a rubric
    with no `<!-- reads: -->` comment produces exactly the sets the scorer
    used before `view_policy` existed.
    """

    sections = _split_rubric_sections(rubric_md, list(criteria))
    return {
        criterion: fields_for(role, declared_fields(sections[criterion])) for criterion in criteria
    }


def _criterion_evidence(evidence: Sequence[str], criterion: str) -> list[str]:
    """The evidence lines attributed to one criterion (`[<criterion>] ...`)."""

    prefix = f"[{criterion}] "
    return [line for line in evidence if line.startswith(prefix)]


@dataclass
class LLMRubricScorer:
    """LLM-driven rubric scorer with a per-criterion filesystem cache.

    Entries are keyed by (trajectory_id, criterion, hash of the criterion's
    rubric section, model), so editing one criterion re-scores only that
    criterion: unchanged criteria return bit-identical cached scores instead
    of fresh LLM noise ("variance ratchet"). Whole-rubric entries keyed by
    (trajectory_id, rubric_md hash) are also read first and migrated into
    per-criterion entries on hit, and still written.
    """

    client: LLMClient
    cache_dir: Path = Path("data/scorer_cache")
    max_workers: int = 8
    # Per-criterion memo: criterion cache key -> (score, evidence lines).
    _crit_mem_cache: dict[str, tuple[float, list[str]]] = field(default_factory=dict)
    # Inside a `fresh_draw` block: the keys this block has already re-drawn.
    # None outside the block. A key not in the set reads as a miss (so the
    # model is asked again and the answer overwrites the cached one); a key
    # in it reads normally, so one block never pays twice for one cell —
    # the optimizer prefetches and then scores the same rows.
    _fresh_keys: set[str] | None = field(default=None, init=False, repr=False)
    # Rows this scoring pass could not read an answer for. Not an init arg:
    # it is an outcome of the pass, and `reset_failure_stats` starts the next
    # one. Guarded by a lock because `prefetch_parallel` scores in threads.
    parse_failures: int = field(default=0, init=False)
    failed_row_ids: set[str] = field(default_factory=set, init=False)
    _failure_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Per-criterion entries live in a subdirectory. Whole-rubric files at
        # the top level are single-call scorings and are not read, because
        # serving one would answer a per-criterion question with a
        # whole-rubric score.
        self._crit_dir = self.cache_dir / "per_criterion"
        self._crit_dir.mkdir(parents=True, exist_ok=True)

    def _model_name(self) -> str:
        """The scoring model, as it goes into the cache key.

        Empty when the client does not name one, which keeps every existing
        cache entry readable — the suffix is simply absent, exactly as it is
        in the ten thousand entries written before this field existed.
        """

        return str(getattr(self.client, "model", "") or "")

    @contextmanager
    def fresh_draw(self) -> Iterator[None]:
        """Ask the model again for every cell scored in the block, and keep it.

        Inside the block the first read of each (trajectory, criterion) key
        is a cache miss whatever the cache holds, and the answer overwrites
        the entry under the NORMAL key — memo and file alike. After the block
        the cache holds the fresh draw as the one canonical value for those
        cells, so every later reader (a rubric-identical candidate, a resumed
        run's baseline) sees exactly what the caller saw.

        This is what the promotion re-score draws through. Filing the fresh
        draw under a salted key would leave the cached draw — the one that had
        just WON selection, the upward-biased sample of several — under the
        plain key. Next iteration a candidate whose rubric.md is byte-identical
        (a rules or policy edit) would read the lucky cells while the baseline
        carried the fresh ones, so `b − c` would contain ~20 % of cells moving
        on scorer noise alone, about z 3 on 500 pairs from no change at all;
        and a `--resume` would re-score the baseline from the plain key and
        get the lucky draw back. Overwriting in place leaves nothing for the
        two paths to disagree on.

        A key already re-drawn inside the block is served from the memo, so
        `prefetch_parallel` followed by `Solver.run` on the same rows pays
        once. The block is restored on exit, including on an exception. It is
        not re-entrant: the optimizer opens one at a time.
        """

        previous = self._fresh_keys
        self._fresh_keys = set()
        try:
            yield
        finally:
            self._fresh_keys = previous

    # --- protocol -----------------------------------------------------------

    def score(
        self,
        trajectory: Trajectory,
        criteria: Sequence[str],
        rubric_md: str = "",
    ) -> tuple[dict[str, float], list[str], ScorerCost]:
        """Score each criterion in its own call, cached per criterion.

        One call per criterion, each shown the rubric's preamble and that
        criterion's own section and nothing else — per-criterion calls rather
        than a single call over the whole rubric, which lets one criterion's
        judgement colour its siblings' (a halo). Narrowing only the model's
        *output* to one criterion while formatting the whole `rubric_md` into
        its *input* would not separate them.

        What this buys is the property a per-criterion cache needs: the key
        determines the value. A score formed as "criterion 1 of 9" and filed
        under a key recording only criterion 1's section text would be read
        by any later rubric reusing that section, first writer winning.

        What it costs: several times more LLM calls (about 6x for a
        six-criterion rubric). Sibling context does move scores — cells with
        byte-identical section text and differing sibling sets agree most of
        the time overall, but markedly less at the widest context gap.
        """

        criteria = list(criteria)
        scores: dict[str, float] = {}
        evidence: list[str] = []
        llm_calls = llm_tokens = 0
        for criterion in criteria:
            cell = self.score_one(trajectory, criterion, criteria, rubric_md)
            assert cell.score is not None  # never cache_only here
            scores[criterion] = cell.score
            evidence.extend(cell.evidence)
            llm_calls += cell.cost.llm_calls
            llm_tokens += cell.cost.llm_tokens

        return scores, evidence, ScorerCost(llm_calls=llm_calls, llm_tokens=llm_tokens)

    def score_one(
        self,
        trajectory: Trajectory,
        criterion: str,
        criteria: Sequence[str],
        rubric_md: str,
        *,
        cache_only: bool = False,
    ) -> CriterionCell:
        """One (trajectory, criterion) cell, exactly as `score()` would produce it.

        `criteria` is the FULL list the policy declares, not just the one
        being scored: the rubric is carved into sections at the headings of
        every declared criterion (`_split_rubric_sections`), so the section
        text — and with it the cache key and the prompt — depends on which
        other criteria exist. A caller that passed only `[criterion]` would
        score a section running to the end of the document and file it under
        a key the full pass never reads. This is the entry point the
        proposer's scoring tool (`meta_filter.score_tool`) uses, and the
        whole point of that tool is that the cells it pays for are the cells
        the full pass later hits.

        `cache_only` reads and never asks the model — the tool uses it to
        show the parent filter's score on the same rows without spending
        anything. `CriterionCell.score` is None when nothing is cached.
        """

        model = self._model_name()
        criteria = list(criteria)
        if criterion not in criteria:
            criteria = [*criteria, criterion]
        sections = _split_rubric_sections(rubric_md, criteria)
        header = _rubric_header(rubric_md, sections)
        fields = criterion_field_sets(rubric_md, criteria)[criterion]
        sig = field_set_signature("scorer", fields) + absent_field_signature(
            scorer_key_fields(fields), trajectory.metadata
        )
        entry = self._read_criterion_entry(
            trajectory, criterion, sections[criterion], model, sig, header
        )
        if entry is not None:
            value, lines = entry
            return CriterionCell(score=value, evidence=lines, cost=ScorerCost(), cached=True)
        if cache_only:
            return CriterionCell(score=None, evidence=[], cost=ScorerCost(), cached=False)
        one_score, one_evidence, cost = self._query(
            trajectory,
            [criterion],
            _criterion_prompt_rubric(header, sections[criterion]),
            fields=fields,
        )
        value = one_score.get(criterion, 0.0)
        lines = _criterion_evidence(one_evidence, criterion)
        self._write_criterion_entry(
            trajectory,
            criterion,
            sections[criterion],
            model,
            value,
            lines,
            field_sig=sig,
            header=header,
        )
        return CriterionCell(score=value, evidence=lines, cost=cost, cached=False)

    # --- per-criterion cache helpers -----------------------------------------

    def _criterion_path(self, crit_key: str) -> Path:
        return self._crit_dir / f"{crit_key}.json"

    def _read_criterion_entry(
        self,
        trajectory: Trajectory,
        criterion: str,
        section_text: str,
        model: str,
        field_sig: str = "",
        header: str = "",
    ) -> tuple[float, list[str]] | None:
        crit_key = _criterion_cache_key(
            section_text, criterion, trajectory, model, field_sig, header
        )
        if self._fresh_keys is not None and crit_key not in self._fresh_keys:
            # `fresh_draw`: whatever is cached is the draw being replaced.
            return None
        if crit_key in self._crit_mem_cache:
            score, evidence = self._crit_mem_cache[crit_key]
            return score, list(evidence)
        path = self._criterion_path(crit_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            score = float(data["score"])
            evidence = [str(line) for line in data.get("evidence", [])]
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            logger.warning("bad per-criterion cache entry: %s", path)
            return None
        self._crit_mem_cache[crit_key] = (score, evidence)
        return score, list(evidence)

    def _write_criterion_entry(
        self,
        trajectory: Trajectory,
        criterion: str,
        section_text: str,
        model: str,
        score: float,
        evidence: list[str],
        *,
        overwrite: bool = True,
        field_sig: str = "",
        header: str = "",
    ) -> None:
        crit_key = _criterion_cache_key(
            section_text, criterion, trajectory, model, field_sig, header
        )
        path = self._criterion_path(crit_key)
        fresh = self._fresh_keys is not None
        if fresh:
            # Overwrite regardless of `overwrite`: replacing the cached draw
            # is the point of the block, and later reads in it must hit.
            self._fresh_keys.add(crit_key)
        elif not overwrite and (crit_key in self._crit_mem_cache or path.exists()):
            return
        import hashlib

        path.write_text(
            json.dumps(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "model": model,
                    "criterion": criterion,
                    "section_sha1": hashlib.sha1(section_text.encode("utf-8")).hexdigest(),
                    "score": score,
                    "evidence": evidence,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._crit_mem_cache[crit_key] = (score, list(evidence))

    # --- parallel prefetch (used by the Optimizer) --------------------------

    def prefetch_parallel(
        self,
        trajectories: list[Trajectory],
        criteria: Sequence[str],
        rubric_md: str,
    ) -> int:
        """Warm the cache for every (trajectory, criterion) pair in parallel.

        Returns the number of trajectories that needed at least one call.

        Every trajectory is submitted rather than pre-filtered: a row is a
        set of per-criterion entries and a partial hit is the normal case, so
        `score()` is the only thing that can say what is actually missing. It
        makes no call for a row that is fully cached.
        """

        criteria = list(criteria)
        sections = _split_rubric_sections(rubric_md, criteria)
        header = _rubric_header(rubric_md, sections)
        field_sets = criterion_field_sets(rubric_md, criteria)
        model = self._model_name()

        missing_calls = 0
        for t in trajectories:
            for criterion in criteria:
                sig = field_set_signature("scorer", field_sets[criterion]) + absent_field_signature(
                    scorer_key_fields(field_sets[criterion]), t.metadata
                )
                if (
                    self._read_criterion_entry(
                        t, criterion, sections[criterion], model, sig, header
                    )
                    is None
                ):
                    missing_calls += 1
        logger.info(
            "LLM scorer prefetch: %d of %d (trajectory, criterion) cells need a "
            "call across %d rows x %d criteria (parallel workers=%d)",
            missing_calls,
            len(trajectories) * len(criteria),
            len(trajectories),
            len(criteria),
            self.max_workers,
        )
        if not missing_calls:
            return 0

        rows_with_calls = 0
        done = 0
        total = len(trajectories)
        failures_before = self.parse_failures
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(self.score, t, criteria, rubric_md): t for t in trajectories}
            for fut in as_completed(futs):
                try:
                    _, _, cost = fut.result()
                    if cost.llm_calls:
                        rows_with_calls += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "scorer failed for %s: %s",
                        futs[fut].trajectory_id,
                        exc,
                    )
                done += 1
                # `progress: N/M` is the line a job monitor reads to
                # put a rate and an ETA on a running job. Without it a full-pool
                # scan would be one INFO line at the start and then silence for
                # tens of hours — indistinguishable, from outside, from a job
                # that slowed down and died with no error. Same cadence as
                # score_pool_baseline: every 25 rows and the last.
                if done % 25 == 0 or done == total:
                    logger.info("progress: %d/%d rows scored", done, total)

        failed = self.parse_failures - failures_before
        if failed:
            # Per pass, because that is the number that says whether this
            # iteration's metrics rest on the pool they claim to.
            logger.warning(
                "scorer parse failures this pass: %d row(s) of %d had no "
                "readable answer after %d attempts and carry no score: %s",
                failed,
                total,
                PARSE_ATTEMPTS,
                ", ".join(sorted(self.failed_row_ids)[:5]),
            )
        return rows_with_calls

    # --- underlying LLM call ------------------------------------------------

    def _query(
        self,
        trajectory: Trajectory,
        criteria: Sequence[str],
        rubric_md: str,
        *,
        score_only: Sequence[str] | None = None,
        fields: frozenset[str] | None = None,
    ) -> tuple[dict[str, float], list[str], ScorerCost]:
        """One LLM call scoring `criteria`, or only `score_only` when given.

        A partial call still carries the full rubric as context but instructs
        the model to score only the missing criteria; the cached ones are
        merged back in by the caller.

        `fields` is the metadata field set to render, which the caller derives
        from what the scored criteria declare. Defaulting to the baseline
        keeps every direct caller — tests included — on the pre-`view_policy`
        prompt, byte for byte.
        """

        target = list(score_only) if score_only is not None else list(criteria)
        criteria_list = ", ".join(target) or "(none)"
        scope_note = ""
        if score_only is not None:
            scope_note = (
                "NOTE: the rubric above may describe additional criteria; those are "
                "already scored. Score ONLY the criteria listed above and do NOT "
                "include any other criterion in `scores`."
            )
        prompt = SCORER_PROMPT.format(
            rubric_md=rubric_md or "(empty rubric)",
            task=(trajectory.task or "(no task)")[:1000],
            metadata=scorer_metadata_block(
                trajectory.metadata,
                fields if fields is not None else SCORER_BASELINE_FIELDS,
            ),
            histogram=_histogram(trajectory),
            turns_block=render_turns(trajectory),
            submission=render_submission(trajectory),
            reference_blocks=scorer_reference_blocks(
                trajectory.metadata,
                fields if fields is not None else SCORER_BASELINE_FIELDS,
            ),
            criteria_list=criteria_list,
            scope_note=scope_note,
        )
        calls = tokens = 0
        last_text = ""
        for attempt in range(1, PARSE_ATTEMPTS + 1):
            response = self.client.generate(
                prompt,
                system=SCORER_SYSTEM,
                temperature=0.0,
            )
            calls += response.calls
            tokens += response.total_tokens
            parsed = _parse_scorer_response(response.text, target)
            if parsed is not None:
                scores, evidence = parsed
                return scores, evidence, ScorerCost(llm_calls=calls, llm_tokens=tokens)
            last_text = response.text or ""
            logger.warning(
                "scorer response for %s / %s was unreadable (attempt %d of %d): %r",
                trajectory.trajectory_id,
                criteria_list,
                attempt,
                PARSE_ATTEMPTS,
                last_text[:160],
            )

        failure = ScoreFailure(
            trajectory_id=trajectory.trajectory_id,
            criterion=criteria_list,
            attempts=PARSE_ATTEMPTS,
            excerpt=last_text[:160],
        )
        self._record_failure(failure)
        raise ScorerParseError(failure, ScorerCost(llm_calls=calls, llm_tokens=tokens))

    # --- failure accounting --------------------------------------------------

    def _record_failure(self, failure: ScoreFailure) -> None:
        with self._failure_lock:
            self.parse_failures += 1
            self.failed_row_ids.add(failure.trajectory_id)
        logger.warning("scorer failed a row: %s", failure.describe())

    def reset_failure_stats(self) -> None:
        """Start a new scoring pass's count. The optimizer calls this per
        iteration so `parse_failures` reads as "this iteration", not "ever"."""

        with self._failure_lock:
            self.parse_failures = 0
            self.failed_row_ids = set()


def _parse_scorer_response(
    text: str, criteria: Sequence[str]
) -> tuple[dict[str, float], list[str]] | None:
    """Scores and evidence, or None when the answer cannot be read.

    None covers both shapes of unreadable: text that is not JSON at all, and
    JSON that omits a criterion the prompt asked for. Filling the second with
    0.0 would be indistinguishable in the cache from a model that looked and
    judged the criterion neutral.
    """

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                obj = None
    scores: dict[str, float] = {}
    evidence: list[str] = []
    if isinstance(obj, dict):
        raw_scores = obj.get("scores", {}) or {}
        if isinstance(raw_scores, dict):
            for c in criteria:
                block = raw_scores.get(c)
                if isinstance(block, dict) and "score" in block:
                    try:
                        v = float(block["score"])
                    except (TypeError, ValueError):
                        continue
                    scores[c] = max(-2.0, min(2.0, v))
                    ev = block.get("evidence")
                    if ev:
                        evidence.append(f"[{c}] {ev}")
                elif isinstance(block, int | float):
                    scores[c] = max(-2.0, min(2.0, float(block)))
        summary = obj.get("summary")
        if isinstance(summary, str) and summary:
            evidence.append(f"[summary] {summary}")
    if any(c not in scores for c in criteria):
        return None
    return scores, evidence
