"""Proposer variant that explores the run directory with tools.

The static `Proposer` gets one shot: everything it may look at is pasted
into the prompt, and it answers with a JSON blob of three candidates. That
caps what it can notice at whatever we thought to include.

This variant instead hands the model a working directory — the run's own
`runs/<run_id>/` tree — plus Read/Glob/Grep/Bash/Write/Edit, and lets it go
find its own evidence before writing candidate directories itself. Every
artifact the loop has ever produced is therefore in scope: past proposer
responses, per-instance solver decisions, critic reports, selection records.

This mirrors Meta-Harness, which grants its proposer the same tool set with
no path allow-list and treats the working directory as the sandbox boundary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from meta_filter.complexity import measure_filter_complexity
from meta_filter.filter_loader import load_filter
from meta_filter.llm.claude_code import ClaudeCodeHeadlessClient
from meta_filter.proposer import (
    Proposer,
    ProposerResult,
    _write_rationale_md,
    render_edit_discipline,
)
from meta_filter.score_tool import ScoreToolAccess

logger = logging.getLogger(__name__)

# Meta-Harness grants Read/Glob/Grep/Agent/Write/Edit/Bash. We drop Agent —
# sub-agents multiply cost unpredictably and nothing here needs them.
DEFAULT_AGENT_TOOLS: tuple[str, ...] = ("Read", "Glob", "Grep", "Bash", "Write", "Edit")

_LAYOUT_NOTE = """\
You are working inside a meta-optimization run directory. Layout:

  config.yaml                       run configuration
  summary.csv                       one row per completed iteration
  iterations/<NNN>/
      current_filter/               the filter this iteration started from
      candidates/<id>/              filters proposed this iteration
      candidates/<NNN>_baseline/    the loop's own copy of current_filter —
                                    never delete, move or edit it; the loop
                                    reads it after you finish
      candidates/<id>/precheck.json what that candidate scores on the pairs
                                    a judge has ALREADY adjudicated, using
                                    cached rubric cells. No judge was called
                                    to produce it. Present from the revision
                                    leg onward.
      proposer/prompt.txt           what the proposer was asked
      proposer/response.json        what it answered
      solver_outputs/               per-trajectory decisions, with reasons
      critic_outputs/               rows the JUDGE placed on the other side of
                                    a pair from this candidate, with the
                                    judge's own rationale. Nothing in here is
                                    the filter's own verdict re-labelled.
      boundary_flags/               the filter grading ITSELF — decisions it
                                    made on thin margins. No judge saw these.
      ranking.json                  per-candidate concordance, distinct score
                                    levels, judge TIE count, criterion stats
      ranking_pairs.jsonl           the COMPLETE judged pair set for this
                                    pool — every pair, the judge's order and
                                    both of its rationales, and every
                                    candidate's result on each pair. Read
                                    this before diagnosing; `critic_outputs/`
                                    holds the capped, evidence-rich subset of
                                    the same pairs.
      v2_critics/rubric_critic.json rubric quality report (DIAGNOSTIC — this
                                    number carries weight 0 in the score)
      v2_critics/rule_critic.json   per-rule precision / redundancy stats
                                    (DIAGNOSTIC — also weight 0)
      selection.json                which candidate won, and why
      record.json                   the iteration's metrics

Earlier iterations of THIS run are all present. Read whatever helps.

Stay inside this directory. Other optimization runs are executing right now
in sibling directories; their artifacts describe a different filter lineage
and a different trajectory pool, and writing there corrupts their results.
Never read or write a path containing `..`.
"""


def _investigation_block(has_history: bool, current_filter_rel: str) -> str:
    """Step 1 of the instructions.

    `current_filter_rel` is named in BOTH branches on purpose. A proposer not
    told what it is editing writes each candidate from scratch, sharing only
    a fraction of its lines with the filter it was supposed to improve, so
    nothing accumulates across generations and the scorer cache never hits.
    """

    if not has_history:
        return f"""\
1. This is the first iteration: there are no `solver_outputs/`,
   `critic_outputs/`, or past `proposer/` artifacts yet, because nothing has
   run. Do not go looking for them — the evidence above is all there is.
   Read `{current_filter_rel}/` — that is the filter you are editing.
"""
    pairs_file = """
     - `iterations/*/ranking_pairs.jsonl` — the whole judged pair set, one
       line per pair: the judge's order, both rationales, and what every
       candidate scored it. The prompt above shows a handful of pairs; this
       is all of them, and it is the only place your own filter's answers sit
       next to the judge's."""
    return f"""\
1. Investigate. The evidence above is a summary, not the whole record. Read
   the actual artifacts before deciding what is wrong with the current filter.
   Start with `{current_filter_rel}/` — that is the filter you are editing,
   and every candidate you write begins as a copy of it.
   Some things worth checking that the summary cannot show you:
     - `iterations/*/solver_outputs/` — the per-trajectory reasons behind
       decisions, including the ones nobody flagged.
     - `iterations/*/critic_outputs/` — the pairs the judge ordered against
       the filter, and its stated rationale for each. This is the only
       place a second opinion speaks; everything else in the run tree is
       the filter's own output or arithmetic over it.
     - `iterations/*/ranking.json` — each candidate's concordance, its
       distinct score levels, and how many judged pairs came back TIE.{pairs_file}
     - `iterations/*/proposer/response.json` — what earlier iterations
       already tried, so you do not re-propose a rejected idea.
     - `iterations/*/selection.json` — why a candidate won or lost.
"""


_RANKING_MISSION = """\

## RANKING MODE — the filter is a scorer, not a gate

This run evolves a CONTINUOUS SCORER. `decide()` must return, alongside the
usual fields, `"score": <float 0-100>` — higher means better SFT training
data. Every row gets a score; there is no "obvious accept" shortcut, because
every score moves the row's position in the ordering. Keep emitting a
`decision` too (it is bookkeeping), but the objective reads the score.

{selection_measures}

HARD CONTRACT — violating any of these discards the candidate before it is
even scored:
  - `policy.py` MUST export a non-empty `RUBRIC_CRITERIA` list. The scorer
    reads THIS list to know which criteria to score; if it is missing or
    empty, nothing gets scored, every row collapses to one constant
    fallback score, and the candidate is thrown away.
    The way this usually happens: you rewrite `policy.py`, you write
    `RUBRIC_WEIGHTS` with the criteria as its keys, and you never write
    `RUBRIC_CRITERIA`. The names are all there in
    the weights dict and the file reads fine — but the scorer does not read
    that dict, and the candidate is lost. `RUBRIC_CRITERIA` and
    `RUBRIC_WEIGHTS` hold the same criterion names and BOTH have to be in
    the file; check for both before you finish.
  - Every name in `RUBRIC_CRITERIA` must match a criterion section in your
    `rubric.md`, and every criterion your `decide()` reads must be in the
    list — a mismatch silently scores as "missing", which is scored as
    absence of merit, not neutrality.
  - `decide()` MUST return `"score": <float>` on every row.

Design consequences:
  - A scorer with a handful of distinct output values cannot rank; below the
    degeneracy floor its concordance is zeroed AND a fatal score penalty is
    applied — a filter that cannot express an ordering cannot win. Compose
    enough independent signals (rule features, rubric criteria, calibrated
    offsets) that scores genuinely spread.
  - The distribution shape you produce is reported back as a DIAGNOSTIC
    (histogram, entropy). Do not chase a
    shape — a bimodal pool is honestly bimodal; noise injected to fake a
    spread is ordering noise and loses pairs.
  - Rank-lift telemetry replaces flip counts: it measures how much the
    ordering moves when one criterion or rule is ablated. Near-zero means
    the item does not move the ordering.
  - Criterion-health telemetry reports, per criterion, where its scores
    actually sit and how correlated the axes are. Two failure modes it
    exposes are cheap wins: a DEAD axis (nearly all rows on one anchor)
    orders nothing and only burns scorer calls — repair what it reads or
    remove it; and a REDUNDANT pair (high |rho| between two criteria whose
    prose differs) re-measures one axis twice, which double-weights it in
    any aggregate and imports correlated scorer noise.
{redundancy_note}
  - Write each rubric criterion SELF-CONTAINED (its section readable without
    the rest of the rubric): scoring is per criterion, and a criterion that
    leans on another's text is scored against words the scorer is never shown.
  - The loop pays for scoring; you do not. Do not optimise for cache hits.
    A rubric edit costs you nothing and is evaluated exactly like a rules
    edit, by the precheck and then by the full pass; unchanged criteria keep
    their cached cells and changed ones are scored fresh, which is
    bookkeeping the precheck reports, not a reason to leave a criterion you
    have a hypothesis about untouched.
{score_tool}"""


# Appended to the mission when the optimizer has armed the criterion-scoring
# tool (`meta_filter.score_tool`). The command is spelled out verbatim with
# its absolute paths: the agent's cwd is the run directory and the tool lives
# in the repository's virtualenv. Rendered only when a `ScoreToolAccess` is
# given.
#
# The shape of this section is a finding, not a preference. Given one
# runnable command with `--rows` left as a placeholder, a proposer copies
# that example verbatim and stops far below the cap. So the worked commands
# are a SEQUENCE, `--rows` gets a real example in the `traj_...` spelling the
# diagnostics print, and the cap is stated in calls rather than in dollars
# the agent has no exchange rate for.
_SCORE_TOOL_SECTION = """\

## Scoring a criterion before you commit to it

You can score a criterion you are drafting on a handful of TRAIN rows under
this run's own scorer model and prompt, the way you replay a rule against
`solver_outputs/`. It is the same draft-measure-revise loop you already run
on rules, and like that one it takes several calls, not one. Write the
rubric into a candidate directory first (with `RUBRIC_CRITERIA` in its
policy.py naming the criterion), then:

1. Score ten rows on the criterion you are least sure the scorer will read
   the way you meant it:

       {command} \\
           --rubric <candidate dir>/rubric.md --criterion "<criterion name>" --sample 10

   Per row you get the new score, the scorer's evidence line, and the
   criterion's score under the current filter where one exists.

2. Read every evidence line. It is what the scorer believed it was
   answering. A line that quotes something your anchors never mention, or
   the same anchor on two rows you would separate, is your text failing to
   say what you meant — not the row being hard.

3. Rewrite the anchors and re-score THE SAME ROWS, so the only thing that
   changed is your wording. Pass the ids the last call reported back with
   `--rows` (at most 20 per call):

       {command} \\
           --rubric <candidate dir>/rubric.md --criterion "<criterion name>" \\
           --rows traj_21c3fa9a,traj_0115cdc7,traj_dd44501a

   A row that did not move is a row your edit did not reach. A fresh
   `--sample` here would tell you nothing: different rows, so any change
   could be the rows rather than the edit.

4. If the picture is still unclear, widen — up to 20 rows in one call —
   before you conclude the criterion works.

Trajectories go by the same `traj_...` names here as everywhere else in
this run: that is what `--rows` takes and what the output reports back. An
id this run does not recognise is refused as unrecognised, so a typo is
something you can fix and retry. The other refusal is a real boundary —
rows outside the train pool are the exam, and no id reaches them.

Where the judge has already ordered pairs among the rows you asked for, the
output also reports how many of them the new criterion alone orders the
judge's way, against how many the old one does.

Budget: ${budget:.2f} per leg (one leg = this call), and a ten-row call
costs about $0.05 — so the cap is roughly {ten_row_calls} such calls. One
call leaves most of that unspent; scoring every criterion on every row is
not on offer. Every output ends with `budget: spent $X of $Y this leg,
remaining $Z`, and a call that would pass the cap is refused before it
spends anything. Cells you buy here are cached for the full pass, so
nothing is paid for twice.

## Checking a rule before you commit to it

A rule is pure Python over the trajectory, so checking one costs NOTHING —
no LLM call, no budget, no cap. A candidate whose `rules.py` you have not
run is not ready to write: the run itself will not tell you whether the rule
fired, and an iteration is the unit you pay to find out.

1. Write the draft rule into the candidate directory, then run every rule in
   the file against twenty train rows:

       {rule_command} \
           --rules <candidate dir>/rules.py --sample 20

2. Read `fires: k/20` first, because both ends of it are failures:

   * `0/20 — DEAD`. The commonest cause is reading a `metadata` key these
     rows do not carry. Nothing else in the run reports this: a rule that
     never fires has no scores to be constant and no rank lift, so it reads
     as a no-op that cost you an iteration. Open a row and see which keys
     are actually there.
   * `20/20 — SATURATED`, or anything near it. A rule that fires on nearly
     every row separates nearly no pairs, whatever its verdict says.
   * A key that EXISTS but holds the same value on every row is the same
     dead end as an absent one. Twenty rows show you that too.

3. In between is where a rule can order something, so read the pair credit:
   how many judged pairs it orders the judge's way, out of those where
   exactly one side fires. Near 50% means the rule fires on a real
   distinction that is not the one the judge is making — it is not yet worth
   its lines. Few pairs, or none, means this sample cannot tell you; widen
   it before concluding either way.

4. Fix the rule and re-run it on THE SAME rows, so the only thing that
   changed is your code. Step 1 printed the ids it used — pass them back:

       {rule_command} \
           --rules <candidate dir>/rules.py --rows <ids step 1 printed>

   A row that did not move is a row your edit did not reach. A fresh
   `--sample` here would tell you nothing: different rows, so any change
   could be the rows rather than the edit.

Do this while you are writing the candidate, not after: a candidate whose
rules.py you have not run is one you cannot say anything about, and once the
iteration starts the only report you get is silence.

"""


def _render_score_tool_section(access: ScoreToolAccess) -> str:
    """The tool section, filled in for one leg's access.

    Both legs render the same text: the revision leg is where a precheck has
    already named a concrete regression — the one place pre-testing a
    rewritten criterion pays most.
    """

    return _SCORE_TOOL_SECTION.format(
        command=access.command,
        budget=access.budget_usd,
        ten_row_calls=access.affordable_ten_row_calls,
        # The rule checker takes no budget and needs no arming, so it has no
        # access object of its own; it is the same interpreter and the same
        # run, and deriving the command here keeps one source of truth for
        # both. It rides on this section because that is where the proposer
        # is already told how to measure something before proposing it.
        rule_command=access.command.replace("meta_filter.score_tool", "meta_filter.rule_tool"),
    )


# What the mission says selection measures: the train pairs are evidence and
# a hidden pair set is the exam.

_SELECTION_MEASURES = """\
What selection measures: pairwise CONCORDANCE on a FIXED set of trajectory
pairs you cannot see — rows outside this pool, judged once at the start of
the run, never shown here. A candidate replaces the current filter only when
it orders significantly more of those pairs correctly than the current
filter does (paired test; b = pairs it gets right that the current filter
gets wrong, c = the reverse). The selector's line in
`iterations/*/selection.json` reports b, c, z and p for every candidate.

The train-pair concordance shown here, and the misordered pairs, are the
evidence you optimise against: pairs every candidate orders the same way are
pseudo-truth for free, and a decile-stratified sample of disputed pairs is
adjudicated by a judge. Winning them is necessary but not sufficient — a
change that only fixes the specific pairs listed and nothing general will
not move the hidden set. Order the WHOLE distribution, not just near some
cut: the eventual training set is "top N% of the pool by this score", and
where that cut lands is decided downstream, not by you — so the ordering
must be trustworthy among the strong rows, among the weak rows, and in the
middle."""


# What the rows carry (`meta_filter.metadata_free`). It describes what the
# rows DO carry and names nothing else: naming the outcome fields to say they
# are missing would re-introduce, by name, exactly what the strip took out,
# and a proposer told a field exists writes criteria that read it.
_DATA_DESCRIPTION = """\
     - **This run carries no outcome metadata.** A row holds the task, the
       transcript and the submitted diff, plus process counts where the pool
       records them — no grading outcome, no test result, no reference fix.
       Judge from what the agent did and what it shipped.
     - Do not substitute the agent's own claims for a verdict nobody gave:
       that a trajectory *says* its tests passed is a thing it did, not
       evidence that they did."""

# The metadata keys a stripped row can carry.
_METADATA_KEYS = "`source`, `api_calls`, `n_messages`, `dataset`,\n       `license`."

_AXIS_QUERY = """\
     - *Query / PR quality*: a clean trajectory on a bad task is still bad
       training data. The rubric and rules may judge the task itself — is the
       problem statement specific enough to be solvable, does it name a
       concrete defect and a way to tell it is fixed."""


# Composition and confirmation. There is no accept set for the filter to
# compose — selection reads an ordering, and the training set is a top-k cut
# of it taken downstream — so the argument is about the top of the ordering,
# not about a wrong ACCEPT: a proposer told to protect an accept path keeps
# doing so while every row comes back REVIEW.

_AXIS_COMPOSITION = """\
     - *Top-of-ordering composition*: the training set is a top-N% cut of
       YOUR ordering, and where the cut lands is decided downstream. So
       whatever your score piles at the top becomes the training
       distribution. A scorer that wins pairs can still stack the high end
       with easy, homogeneous queries (a lineage can drift
       onto the easiest instances in the pool). Prefer designs whose high end
       preserves query difficulty and repo diversity; the probe-trend
       telemetry shows the current drift."""


_CONFIRMATION = """\
   **Confirmation on the high side is a design invariant.** The two
   directions of an ordering error are not symmetric: a row you rank too LOW
   costs one row out of a surplus pool, while a row you rank too HIGH takes
   its place inside the eventual top-N% cut. So any term that can lift a row
   into the top of your ordering must be CONFIRMED by rubric scores before it
   stands — countable approximations of semantic judgments (substring
   containment, a single concreteness regex, region-bucket overlap) may
   NOMINATE a lift, never finalize one. State the LLM cost the confirmation
   pays; buying certainty on the side that ships data is what the budget is
   for."""


# The consequence half of the REDUNDANT-pair finding above: which score pays
# for it. An agent told the wrong one either optimizes a term selection
# ignores or ignores a term selection charges.

_REDUNDANCY_NOTE = """\
    Redundancy (|rho| above the run's threshold between two criteria) is
    measured and shown to you but does NOT enter this iteration's selection;
    it is charged to the final filter's reported score, so avoid it where you
    can without giving up ordering power — differentiate the axes (make each
    read evidence the other cannot) or merge them. Do not evade the measure
    by adding noise: noise loses pairs directly."""


def _agentic_instructions(
    rel_candidates_dir: str,
    iteration: int,
    n: int,
    current_filter_rel: str,
    *,
    parent_counts: dict[str, int],
    initial_counts: dict[str, int] | None = None,
    candidates_scored: int | None = None,
    score_tool: ScoreToolAccess | None = None,
    frozen: tuple[str, ...] = (),
) -> str:
    # The ZERO LIFT finding is one `criterion_lift.render_ranking` emits; the
    # removal it names is carried by every candidate rather than tested as a
    # mechanism of its own.
    zero_lift_carve_out = (
        "\n"
        "   The one carve-out is a removal the observations name under ZERO LIFT.\n"
        "   That item is MEASURED to change no pair's ordering, so carrying its\n"
        "   removal adds no second idea for the gate to test — fold it into every\n"
        "   candidate instead of spending a slot on it. A candidate whose only change\n"
        "   is such a removal ties its parent by construction and therefore loses to\n"
        "   any sibling that improves anything.\n"
    )
    ranking_block = (
        _RANKING_MISSION.replace("{redundancy_note}", _REDUNDANCY_NOTE)
        .replace("{selection_measures}", _SELECTION_MEASURES)
        .replace(
            "{score_tool}",
            _render_score_tool_section(score_tool) if score_tool is not None else "",
        )
    )
    # The caps are printed against the filter under edit (the soft caps) and
    # the run's initial filter (the hard cap and the floor).
    discipline = "\n" + render_edit_discipline(
        n_candidates=n,
        parent_counts=parent_counts,
        initial_counts=initial_counts,
        candidates_scored=candidates_scored,
        hypothesis_source=(
            f"the candidate's `hypothesis` in `{rel_candidates_dir}/diagnosis.json` (step 3)"
        ),
        frozen=frozen,
        score_tool=score_tool is not None,
    )
    # The rule-check field and its note are rendered only when the rule
    # checker is armed (the same condition as its prompt section): a run
    # without it must not be told to report a count it cannot measure.
    armed = score_tool is not None
    rule_check_field = _RULE_CHECK_FIELD if armed else ""
    rule_check_note = _RULE_CHECK_NOTE if armed else ""
    precheck_step = _PRECHECK_STEP.replace(
        "{redundancy_terms}",
        "the complexity term, the redundancy figure it is only reported with",
    )
    if armed:
        # The step says what the precheck measures, and that is false for
        # rules: a rule that fires on no row moves none of its numbers.
        precheck_step = precheck_step.replace(
            "the rows those pairs name, not the whole training pool.\n",
            "the rows those pairs name, not the whole training pool.\n" + _PRECHECK_RULE_CAVEAT,
        )
    data_labels = _DATA_DESCRIPTION
    axis_query = _AXIS_QUERY
    axis_composition = _AXIS_COMPOSITION
    confirmation = _CONFIRMATION
    metadata_keys = _METADATA_KEYS
    return (
        ranking_block
        + f"""\

## How to answer

Do not reply with a JSON blob of candidates. Instead, investigate first and
then write your candidates to disk.

{_investigation_block(iteration > 0, current_filter_rel)}
2. Write exactly {n} candidate directories under `{rel_candidates_dir}`,
   named `iter{iteration:03d}_00_<slug>`, `iter{iteration:03d}_01_<slug>`, …
   where `<slug>` is a short lowercase-with-underscores name for the idea.

   **Start each candidate as a copy of `{current_filter_rel}/`, then edit the
   copy** — that keeps the scorer cache warm and preserves what earlier
   iterations paid for. Each directory ends up with all three files:
     - `rubric.md`
     - `rules.py`   (top-level `rule_<name>(trajectory) -> dict | None`)
     - `policy.py`  (exposes `decide(...)`)
   `policy.py` may do `from rules import <helper>` to reuse rules.py helpers.

   **Good candidates change a mechanism. Bad candidates just tune numbers.**
   A mechanism change replaces HOW a decision is made: a new rubric criterion
   with the policy branch that reads it, a different evidence source, a
   restructured cascade, a merged or split rule. Re-read each candidate after
   writing it and ask: if only constants differ from the current filter, is
   there a real, evidence-backed reason this exact number generalizes beyond
   the 200 training trajectories? Thresholds carved between two specific
   training records are memorization, and they lose held-out agreement. If
   the honest answer is "it's a tune", REWRITE that candidate around a
   genuinely different mechanism instead. A full rewrite of a file is allowed
   when the hypothesis calls for it; what is not allowed is a rewrite that
   silently discards behavior the hypothesis never questioned.

   **Escalate when the loop telemetry says the search is stagnant.** If the
   telemetry above shows the rubric unchanged for many iterations, novelty
   persistently low, or rubric-editing candidates always losing, then small
   edits are being outcompeted by the incumbent's cached advantage — at least
   one candidate MUST change the decision mechanism (rubric criteria set or
   policy structure), not thresholds. Prototype first: with Bash you can run
   `python` against real solver outputs in the run directory to sanity-check
   a mechanism before committing to it. Read past `proposer/response.json`
   files so you do not re-explore an axis that already lost for a stated
   reason.

   **One hypothesis per candidate.** A candidate tests exactly one idea about
   what is wrong. If you catch yourself writing "and also", that is a second
   candidate. The hypothesis may touch more than one file when the change is
   genuinely one idea — a new criterion is inert without the policy branch
   that reads it — but you must be able to state the causal link in a single
   sentence. If you cannot, it is two changes.
{zero_lift_carve_out}
   Candidates should test genuinely different hypotheses from each other —
   three variations on one threshold wastes the iteration.
{discipline}
## What the data gives you

   A trajectory is four fields — `trajectory_id`, `task`, `turns`, `metadata`:
     - `task`: the issue / PR description the agent was given, as text.
     - `turns`: the full agent transcript. A LIST of turns, each carrying
       `.role`, `.action` and `.content`. There is no `messages` field.
     - `metadata.submission`: the diff the agent shipped, and
       `metadata.submission_bytes` its size. There is no `model_patch`.
{data_labels}
     - `metadata` carries these WHEN THE POOL HAS THEM, and which keys exist
       varies by pool: {metadata_keys}
       Look at a row before relying on one.
   A rule that reads a field these rows do not carry returns None on every
   row and NOTHING in this run will tell you: it has no scores to be constant
   and no rank lift to report, so it simply never fires and your candidate
   reads as a no-op it took you an iteration to learn nothing from. The
   accessors in the current `rules.py` are known to work on this pool — copy
   their shape rather than this list.
   **A fired rule reaches `policy.py` under its PYTHON FUNCTION NAME.** The
   runner overwrites whatever label the rule passes to `_fire` with
   `rule.__name__`, so a rule defined as `def rule_no_tests(...)` arrives at
   `decide()` as `{{"name": "rule_no_tests", ...}}` even if it declared itself
   `"no_tests"`. Key every price table and every `if name == ...` branch by
   the FUNCTION name, prefix included. Getting this wrong raises nothing:
   `dict.get(name, fallback)` returns the fallback, so the rule quietly does
   the wrong thing — or, when the fallback is 0, nothing at all while still
   firing. If the wiring readout in your diagnosis says RULE WIRING BROKEN,
   that is this mistake; fix the key rather than deleting the rule.

   The run directory may also contain `reference/` — curated notes on how
   published SWE-data pipelines filter SFT data. Read it once before your
   first diagnosis; it is a menu of mechanisms, not a mandate.

   **Two evaluation axes beyond the trajectory itself are in scope:**
{axis_query}
{axis_composition}

   **What the rubric is FOR — two roles, stated so neither atrophies:**
     1. *Judge what code cannot count.* Reasoning coherence, whether an
        alternative fix is defensible, whether a task is genuinely
        learnable — anything whose honest form is a reading, not a regex.
     2. *Audit the rules' proxies.* Every countable rule is an
        approximation of a semantic claim, and the code cannot tell you
        when the approximation and the claim come apart — a filename
        substring match is not "the issue names the defect", a shared
        60-line diff window is not "engages the same bug". When a proxy
        NOMINATES a decision (especially a keep), a rubric criterion
        should ask the original semantic question the proxy stands in
        for, on that row. This is not double-counting: re-counting the
        proxy's number is double-counting; judging the claim behind it is
        the check the number cannot do. The criterion-lift telemetry will
        show you which audit questions actually catch divergence.

{confirmation}



3. Write `{rel_candidates_dir}/diagnosis.json`:
   {{"diagnosis": "<what you concluded is wrong, and the evidence you read
   that led you there>", "candidates": [{{"name": "<slug>", "hypothesis":
   "<the single idea this candidate tests, in one sentence>", "notes": "<what
   changed, in which files, and why those edits are one idea>"{rule_check_field}}}, ...]}}
{rule_check_note}
Verify each candidate parses before you finish — a candidate that fails to
import is discarded and the iteration is wasted.
{precheck_step}"""
    )


# Appended to leg 1's instructions when the loop is going to precheck. Leg 1
# must know a revision is coming, or it will treat this call as its last word
# and the revision leg arrives to find nothing worth revising; it must also
# know the revision is the ONLY one, or it will ship a probe and plan a fix
# that never gets a round.
# Appended to the diagnosis.json contract when the rule checker is armed.
# `rule_check` is the proposer's own claim that it measured a rule before
# adding it; `precheck.py` re-measures and compares, so the claim is
# checkable rather than paperwork.
# The `diagnosis.json` field itself, spliced into the candidate object.
# Empty when the checker is not armed, which keeps the older prompt
# snapshots byte-exact.
_RULE_CHECK_FIELD = (
    ', "rule_check":\n'
    '   "<the firing count you measured for each rule this candidate adds\n'
    "   or edits, e.g. rule_tiny_patch 3/20, rule_churn 7/20 — or the single\n"
    '   word none if this candidate keeps no rules at all>"'
)

_RULE_CHECK_NOTE = """
   `rule_check` is not paperwork. A rule you have not run is one you cannot
   say anything about: it may read a field these rows do not carry and fire
   on none of them, and the precheck below will NOT tell you — a rule that
   never fires and a rule that fires without helping produce the same
   concordance. Run it, read the count, and add the rule only once the count
   says it separates something. The loop measures the same rules itself and
   compares, so a number you did not measure is worse than none.
"""

# Inserted into `_PRECHECK_STEP` under the same condition, immediately after
# the sentence that says what the precheck measures.
_PRECHECK_RULE_CAVEAT = """
**It does not tell you whether a rule fires.** What it reports is
concordance, redundancy and the resulting score, and a rule that fired on no
row at all moves none of them — exactly like a rule that fired and did not
help. If you are adding a rule, its firing count is yours to measure before
you write the candidate, not something this step will hand back.
"""

_PRECHECK_STEP = """
## What happens next — one precheck, then one revision

When this call returns, the loop runs each candidate you wrote against the
truth pairs a judge has ALREADY adjudicated in earlier iterations, reusing
cached rubric cells for every criterion whose text you did not change, and
writes the result to `candidates/<id>/precheck.json`: concordance on those
pairs, {redundancy_terms}, the resulting score under the
formula the precheck reports, and the list of pairs the candidate orders
against the judge.

No judge is called for it — no new verdict exists, and nothing you write can
buy supervision. It is measured on pairs bought BEFORE this iteration, over
the rows those pairs name, not the whole training pool.

You then get exactly ONE more call, in which you may edit the candidates.
After that the loop scores them for real. So write real candidates now: the
revision is for repairing what the precheck exposes, not a second chance to
start.
"""


def _revision_instructions(
    rel_candidates_dir: str,
    iteration: int,
    precheck_block: str,
    score_tool: ScoreToolAccess | None = None,
    zero_lift_note: str = "",
) -> str:
    """Leg 2, the revision. Stated as final because it is.

    The bound is the whole design. A proposer that believes it can iterate
    will hold back — it will ship a probe, read the precheck, and plan to fix
    it next round — and there is no next round: the loop scores whatever
    stands here and pays a full solver pass and a judging round for it. So
    the instruction says the words "this is the only revision" rather than
    implying it from the absence of a further prompt.

    `score_tool` renders the whole tool section here, not a reminder of it.
    This leg is a FRESH process — see `revise` — so nothing carries over
    from leg 1's prompt. The section sits above "Leaving a candidate exactly
    as it is, is a legitimate answer", because the revision is where a
    precheck has already named a concrete regression, which is exactly where
    testing a rewritten criterion before shipping it pays most.
    """

    tool_section = ""
    if score_tool is not None:
        tool_section = _render_score_tool_section(score_tool)
    # Leg 2 is where candidates become final, and it is a fresh process: the
    # ZERO LIFT finding reached leg 1 only. A removal folded into every
    # candidate then shows up here as a shrunken filter with no upside on the
    # old pairs, and the obvious repair — put the criterion back — silently
    # undoes it on the only leg that ships. Same fix as `score_tool`: carry
    # the sentence, not a reminder of it.
    zero_lift_section = ""
    if zero_lift_note:
        zero_lift_section = (
            "\nStill true from the diagnosis leg:\n"
            + zero_lift_note.strip("\n")
            + "\nA criterion or rule named there stays removed. Restoring one is not a "
            "repair — it reverses a measured finding, and the precheck number you are "
            "looking at cannot see the scorer calls it costs.\n"
        )
    return f"""\
{precheck_block}

## Revise — this is the ONLY revision round

You may now edit the candidate directories under `{rel_candidates_dir}` in
place. Each candidate's own numbers are in
`{rel_candidates_dir}/iter{iteration:03d}_*/precheck.json`, including the
full list of pairs it orders against the judge; read them.

This is the last call you get this iteration. When it returns, the loop
scores these candidates FOR REAL — a full solver pass over the training
pool, then a round of judging — and selects a winner from whatever is on
disk at that moment. There is no third leg and no further precheck.
{tool_section}{zero_lift_section}
What to do with the time:
  - A candidate the precheck marked NOT SCOREABLE is already lost: it never
    reaches the gate, so the iteration goes in a candidate short. Repair it
    from the reason given, or replace it outright with a different
    mechanism. Do not leave it — an unscoreable directory is a wasted slot,
    not a neutral one, and this is the only leg left to fix it in.
  - A candidate scoring below the parent on the pairs it was checked against
    is, on the evidence you have, a regression. Either find and fix what it
    broke or replace its mechanism — shipping it unchanged spends a
    candidate slot on a known loss. If what you change is a criterion, score
    it before you ship it: this leg has its own budget and the precheck has
    just told you which rows to look at.
  - Read the misordered pairs. They name specific trajectories; the
    transcripts are in the run tree and the judge's rationale for each pair
    is in `iterations/*/critic_outputs/`. That directory is capped — for the
    complete set, with every candidate's result on every judged pair and both
    of the judge's rationales, read `iterations/*/ranking_pairs.jsonl`.
  - `unscored` in the misorder list is a ranking-contract violation: your
    `decide()` returned no `score` for that row. Fix it — those pairs are
    charged to the candidate, not forgiven.
  - Do not chase the precheck number itself. It is measured on pairs bought
    in EARLIER iterations, over the rows those pairs name; this iteration
    buys new verdicts after you finish, and a candidate tuned onto the old
    pairs is memorizing an exam it has already seen. Fix mechanisms the
    misorders expose, not individual pairs.
  - Leaving a candidate exactly as it is, is a legitimate answer. Say so in
    `{rel_candidates_dir}/diagnosis.json` if you do.

Keep the same directory names — the loop tracks candidates by them. Keep all
three files loadable; a candidate that stops importing is discarded, and
this time nothing will re-run to tell you.
"""


def _retry_instructions(rel_candidates_dir: str, iteration: int, n: int, errors: list[str]) -> str:
    """Leg 2 when leg 1 wrote nothing loadable — a retry, not a revision.

    There is nothing to precheck, so this leg spends the iteration's second
    and last call on getting any valid candidate at all.
    """

    detail = "\n".join(f"  - {e}" for e in errors[:10]) or "  - (no candidate directory appeared)"
    return f"""\
## Your candidates did not load — one retry

Nothing you wrote under `{rel_candidates_dir}` could be imported, so this
iteration currently has no candidate to score. What went wrong:

{detail}

Write exactly {n} candidate directories named
`iter{iteration:03d}_00_<slug>`, `iter{iteration:03d}_01_<slug>`, … each a
copy of the current filter with your edits applied, each containing
`rubric.md`, `rules.py` and `policy.py`. Import-check every one before you
finish — run `python -c "import policy"` from inside the directory if that
helps.

This is the last call this iteration. Prefer a small, certainly-working edit
over an ambitious one that might not import.
"""


@dataclass
class AgenticProposer(Proposer):
    """Proposer that reads the run directory instead of only its prompt."""

    client: ClaudeCodeHeadlessClient = field(default=None)  # type: ignore[assignment]
    run_dir: Path | None = None
    allowed_tools: tuple[str, ...] = DEFAULT_AGENT_TOOLS
    agentic: ClassVar[bool] = True
    # Directory names that were already under `candidates/` when this
    # iteration's first leg started — the baseline copy, and on a resumed run
    # anything an earlier attempt left behind. Recorded by `propose` so
    # `revise` re-scans the same way and does not adopt the baseline as a
    # candidate. Reset every `propose`, so it never carries across iterations.
    _preexisting: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def propose(self, **kwargs) -> ProposerResult:
        candidates_root = Path(kwargs["candidates_root"])
        candidates_root.mkdir(parents=True, exist_ok=True)
        iteration = kwargs["iteration"]

        run_dir = Path(self.run_dir) if self.run_dir else candidates_root.parents[2]
        rel_candidates = candidates_root.resolve().relative_to(run_dir.resolve())

        kwargs["candidates_root"] = candidates_root
        prompt = self._build_prompt(**{k: v for k, v in kwargs.items() if k != "candidates_root"})
        # The filter under edit lives beside the candidates, one level up.
        current_filter_rel = f"iterations/{iteration:03d}/current_filter"
        prompt += _agentic_instructions(
            str(rel_candidates),
            iteration,
            self.n_candidates,
            current_filter_rel,
            # The filter under edit is the per-iteration baseline the soft
            # caps are measured against; the run's initial filter (set by
            # the optimizer) is what the hard cap and the floor read.
            parent_counts=measure_filter_complexity(kwargs["loaded_filter"].directory)[0],
            initial_counts=self.initial_complexity,
            candidates_scored=self.candidates_scored,
            score_tool=self.score_tool,
            frozen=tuple(self.frozen_files or ()),
        )

        # Anything already here (the baseline copy) is not a proposal.
        self._preexisting = frozenset(p.name for p in candidates_root.iterdir() if p.is_dir())

        response = self._call_agent(run_dir, prompt, iteration)
        candidate_dirs, parse_errors = self._collect(candidates_root)

        if not candidate_dirs:
            parse_errors.append("agent wrote no loadable candidate directories")
        logger.info(
            "agentic proposer iter %d: %d candidates, %d rejected",
            iteration,
            len(candidate_dirs),
            len(parse_errors),
        )
        return ProposerResult(
            prompt=prompt,
            response=response,
            candidate_dirs=candidate_dirs,
            parse_errors=parse_errors,
            legs=1,
        )

    def revise(
        self,
        *,
        result: ProposerResult,
        precheck_block: str,
        candidates_root,
        iteration: int,
    ) -> ProposerResult:
        """Leg 2: one bounded call in which the proposer may edit candidates.

        Fresh call rather than a session resume. Neither harness exposes
        "continue this thread with a new message" as a public API —
        `ClaudeCodeHeadlessClient.generate` starts a session per call, and
        `CodexCliClient` resumes a thread only to finish a leg its own timeout
        killed. Restarting is also what makes the two harnesses behave
        identically here. Nothing is lost by it: the workspace is the same directory, the
        candidates are on disk where leg 1 left them, and each one's
        `precheck.json` sits beside it.

        Returns a result whose `candidate_dirs` are re-scanned from disk, so
        a candidate the revision broke is dropped here rather than at the
        solver, and one it newly added is picked up.
        """

        candidates_root = Path(candidates_root)
        run_dir = Path(self.run_dir) if self.run_dir else candidates_root.parents[2]
        rel_candidates = str(candidates_root.resolve().relative_to(run_dir.resolve()))

        if result.candidate_dirs:
            prompt = _revision_instructions(
                rel_candidates,
                iteration,
                precheck_block,
                score_tool=self.score_tool,
                zero_lift_note=self.zero_lift_note or "",
            )
        else:
            prompt = _retry_instructions(
                rel_candidates, iteration, self.n_candidates, result.parse_errors
            )

        response = self._call_agent(run_dir, prompt, iteration)
        candidate_dirs, parse_errors = self._collect(candidates_root)
        if not candidate_dirs:
            parse_errors.append("agent wrote no loadable candidate directories")
        logger.info(
            "agentic proposer iter %d leg 2 (%s): %d candidates, %d rejected",
            iteration,
            "revision" if result.candidate_dirs else "retry",
            len(candidate_dirs),
            len(parse_errors),
        )
        return ProposerResult(
            prompt=f"{result.prompt}\n\n{prompt}",
            response=response,
            candidate_dirs=candidate_dirs,
            parse_errors=parse_errors,
            legs=result.legs + 1,
        )

    # ------------------------------------------------------------- internals

    def _call_agent(self, run_dir: Path, prompt: str, iteration: int):
        """One agent call in the run directory, with the guard audited."""

        agent = self.client.with_workspace(run_dir)
        agent = _with_tools(agent, self.allowed_tools, score_tool=getattr(self, "score_tool", None))
        audit_path = agent.audit_path
        audit_offset = audit_path.stat().st_size if audit_path and audit_path.exists() else 0
        response = agent.generate(prompt, system=self.system_prompt)
        escapes = _guard_denials(audit_path, audit_offset)
        if escapes:
            logger.warning(
                "agentic proposer iter %d: guard blocked %d out-of-bounds tool call(s); first: %s",
                iteration,
                len(escapes),
                escapes[0],
            )
        return response

    def _collect(self, candidates_root: Path) -> tuple[list[Path], list[str]]:
        """Every loadable candidate directory the agent has written so far.

        `_baseline` is excluded by name as well as by `_preexisting`: the
        baseline is a copy of the filter under edit, and adopting it as a
        candidate would enter the incumbent twice.
        """

        diagnosis, notes_by_slug, hypothesis_by_slug = _read_diagnosis(candidates_root)
        candidate_dirs: list[Path] = []
        parse_errors: list[str] = []
        for target in sorted(candidates_root.iterdir()):
            if not target.is_dir() or target.name in self._preexisting:
                continue
            if target.name.endswith("_baseline"):
                continue
            try:
                load_filter(target)
            except Exception as exc:
                parse_errors.append(f"candidate {target.name}: {exc}")
                continue
            _write_rationale_md(
                target,
                {
                    "name": target.name,
                    "notes": _match_notes(target.name, notes_by_slug),
                    # Becomes the one `Hypothesis:` line the one-mechanism
                    # check counts.
                    "hypothesis": _match_notes(target.name, hypothesis_by_slug),
                },
                diagnosis,
            )
            candidate_dirs.append(target)
        return candidate_dirs, parse_errors


def _with_tools(
    client: ClaudeCodeHeadlessClient,
    tools: tuple[str, ...],
    *,
    score_tool: ScoreToolAccess | None = None,
) -> ClaudeCodeHeadlessClient:
    """The client with its tools, and the sandbox holes the scoring tool needs.

    Each harness confines the agent differently, so each needs a different
    hole and only the field it actually has is set: the Claude Code guard
    blocks a Bash command naming an absolute path outside the run directory,
    so it is told the tool's interpreter (`guard_allow`); the Codex sandbox
    cuts the network, so it is given `network_access`. Neither harness is
    given anything new to WRITE: the tool stages its cells inside the run
    directory and the optimizer merges them, because the run's scorer cache
    also holds the hidden selection rows' cells and a writable cache would
    let the workspace forge the gate's inputs. A client that has neither
    field (a test double, the static path) is left alone.
    """

    from meta_filter.llm.cost_meter import replace_client, unwrap

    changes: dict[str, object] = {"allowed_tools": tools, "append_system_prompt": _LAYOUT_NOTE}
    if score_tool is not None:
        inner = unwrap(client)
        if hasattr(inner, "guard_allow"):
            changes["guard_allow"] = (score_tool.executable,)
        if hasattr(inner, "network_access"):
            changes["network_access"] = True
    return replace_client(client, **changes)


def _guard_denials(audit_path: Path | None, offset: int) -> list[str]:
    """Out-of-bounds tool calls the guard blocked during the last agent call."""

    if audit_path is None or not audit_path.exists():
        return []
    with audit_path.open(encoding="utf-8") as fh:
        fh.seek(offset)
        lines = fh.readlines()
    out = []
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("decision") == "deny":
            out.append(f"{rec.get('tool')}: {'; '.join(rec.get('violations') or [])}")
    return out


def _read_diagnosis(
    candidates_root: Path,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """`(diagnosis, notes by slug, hypothesis by slug)` from `diagnosis.json`."""

    path = candidates_root / "diagnosis.json"
    if not path.exists():
        return [], {}, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], {}, {}
    candidates = [c for c in payload.get("candidates", []) if isinstance(c, dict)]
    notes = {str(c.get("name", "")): str(c.get("notes", "")) for c in candidates}
    hypotheses = {str(c.get("name", "")): str(c.get("hypothesis") or "") for c in candidates}
    raw = payload.get("diagnosis", "")
    diagnosis = [str(d) for d in raw] if isinstance(raw, list) else [str(raw)] if raw else []
    return diagnosis, notes, hypotheses


def _match_notes(candidate_id: str, notes_by_slug: dict[str, str]) -> str:
    for slug, note in notes_by_slug.items():
        if slug and candidate_id.endswith(slug):
            return note
    return ""
