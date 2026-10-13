"""Solver — applies a loaded filter to trajectories.

The solver is deliberately shallow:

    for each trajectory:
        1. run every deterministic `rule_*` from `rules.py`
        2. ask a rubric scorer for one score per criterion in policy.py
        3. call policy.decide(traj, rubric_scores, triggered_rules)
        4. wrap the result into a FilterDecision

Lazy rubric scoring (on by default, `lazy_scoring=False` disables it): before
step 2 the solver probes `decide()` with adversarial score dicts — every
criterion at +2, every criterion at -2, and no scores at all. If all three
probes agree on the decision and none raises, the row's decision cannot depend
on rubric scores (the fired rules decided it), so the LLM scoring step is
skipped and the probe's own payload becomes the record. Rules still run
exactly once per row; the probe reuses their output.

The rubric scorer is pluggable. The default is a deterministic heuristic
scorer that never calls an LLM (so the offline demo is reproducible). An
LLM-based scorer can be dropped in later.

The solver never persists hidden chain-of-thought: only concise
`reasoning_summary` strings + short evidence bullets are stored.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from meta_filter.core.filter_loader import LoadedFilter
from meta_filter.core.schemas import Decision, FilterDecision, Trajectory

logger = logging.getLogger(__name__)

# What fraction of a pool may be unscoreable before the result stops meaning
# anything. One refused trajectory is a quirk of the data — a test fixture
# holding a literal private key can trip a model's safety refusal on one row
# of two hundred — and since every candidate is refused the same trajectory it
# is a constant offset that cannot change which candidate wins.
#
# A large fraction is a different event entirely, and the case worth guarding
# is the one nobody would predict: the proposer writes the rubric, so a
# candidate whose rubric happens to trip the safeguard has *every* call
# refused. That candidate would score terribly for a reason that has nothing
# to do with its quality, and nothing in the metrics would say so.
#
# 5% is where the distortion starts to matter: `lambda_review` is 2.0, so 5%
# unscoreable moves the score by 0.10 — larger than the gaps that usually
# separate candidates. Past half the pool the number is not distorted, it is
# meaningless, and continuing would feed it into selection.
UNSCOREABLE_WARN_FRACTION = 0.05
UNSCOREABLE_FATAL_FRACTION = 0.5


@dataclass
class ScorerCost:
    """Per-call cost accounting so the Validator can bill filter cost."""

    llm_calls: int = 0
    llm_tokens: int = 0


class RubricScorer(Protocol):
    """Anything that maps (trajectory, criteria, rubric_md) -> (scores, evidence, cost).

    `rubric_md` is the full rubric text as read from the filter directory —
    heuristic scorers may ignore it, but an LLM scorer needs it to actually
    honor the rubric that the proposer is editing.
    """

    def score(
        self,
        trajectory: Trajectory,
        criteria: Sequence[str],
        rubric_md: str,
    ) -> tuple[dict[str, float], list[str], ScorerCost]: ...


# ---------------------------------------------------------------------------
# Default deterministic heuristic scorer
# ---------------------------------------------------------------------------


_SEARCH_ACTIONS = {"grep", "rg", "ag", "find", "search"}
_READ_ACTIONS = {"cat", "head", "tail", "read", "view", "less", "more", "open"}
_EDIT_ACTIONS = {
    "edit",
    "apply_patch",
    "patch",
    "sed",
    "awk",
    "write",
    "create",
    "insert",
    "str_replace",
}
_TEST_ACTIONS = {"pytest", "test", "run_tests", "python", "unittest", "tox", "nose"}
_BUILD_ACTIONS = {"build", "make", "cmake", "cargo", "npm", "uv"}
_LINT_ACTIONS = {"lint", "ruff", "flake8", "pylint", "mypy", "eslint"}
_VERIFY_ACTIONS = _TEST_ACTIONS | _BUILD_ACTIONS | _LINT_ACTIONS
_CONTENT_ACTIONS = _SEARCH_ACTIONS | _READ_ACTIONS | _EDIT_ACTIONS | _VERIFY_ACTIONS


def _norm_action(t) -> str:
    return (getattr(t, "action", None) or t.role or "").lower()


@dataclass
class HeuristicRubricScorer:
    """A cheap, deterministic rubric scorer wired for SWE agent trajectories.

    The scorer is domain-aware: navigation actions (`cd`, `ls`, `pwd`) are
    ignored when we look for "action repetition" so long shell sessions are
    not incorrectly penalized. Verification recognizes `pytest`, `python -m
    pytest`, and common build/lint tools. Edit recognizes `apply_patch`,
    `sed`, `awk`, etc.

    Book-keeping fields are set to zero because the heuristic doesn't call
    an LLM. When you swap this class out for an LLM-based scorer, populate
    `llm_calls_per_call` / `llm_tokens_per_call` and Validator will bill it.
    """

    def score(
        self,
        trajectory: Trajectory,
        criteria: Sequence[str],
        rubric_md: str = "",
    ) -> tuple[dict[str, float], list[str], ScorerCost]:
        # Heuristic scorer ignores rubric_md; the LLM scorer honors it.
        _ = rubric_md
        turns = trajectory.turns
        actions = [_norm_action(t) for t in turns]
        contents = [(t.content or "").lower() for t in turns]

        content_actions = [a for a in actions if a in _CONTENT_ACTIONS]
        search_actions = [a for a in actions if a in _SEARCH_ACTIONS]
        edit_indices = [i for i, a in enumerate(actions) if a in _EDIT_ACTIONS]
        verify_indices = [i for i, a in enumerate(actions) if a in _VERIFY_ACTIONS]
        error_indices = [
            i
            for i, c in enumerate(contents)
            if "traceback" in c
            or " error" in c
            or c.startswith("error")
            or "exit code 1" in c
            or "exit status 1" in c
            or "returncode>1" in c
        ]

        scores: dict[str, float] = {c: 0.0 for c in criteria}
        evidence: list[str] = []

        # --- Task Grounding ---------------------------------------------------
        if "Task Grounding" in scores:
            has_task = len((trajectory.task or "").strip()) > 20
            first_five_content = [a for a in actions if a in _CONTENT_ACTIONS][:5]
            first_five_are_recon = sum(
                1 for a in first_five_content if a in _SEARCH_ACTIONS | _READ_ACTIONS
            )
            edit_before_recon = any(
                i
                for i in edit_indices
                if all(a not in _SEARCH_ACTIONS | _READ_ACTIONS for a in actions[:i])
            )
            if has_task and first_five_are_recon >= 3:
                scores["Task Grounding"] = 2.0
                evidence.append("task_grounding: first content actions are recon")
            elif has_task and first_five_are_recon >= 1 and not edit_before_recon:
                scores["Task Grounding"] = 1.0
            elif edit_before_recon:
                scores["Task Grounding"] = -1.0
                evidence.append("task_grounding: edit before any recon")
            elif not has_task:
                scores["Task Grounding"] = 0.0
            else:
                scores["Task Grounding"] = 0.0

        # --- Localization Efficiency ------------------------------------------
        if "Localization Efficiency" in scores:
            n_content = len(content_actions)
            if not n_content:
                scores["Localization Efficiency"] = -1.0
            else:
                # Search-heavy without any edit is a red flag.
                if search_actions and not edit_indices:
                    scores["Localization Efficiency"] = -2.0
                    evidence.append("localization: many searches, never reached an edit")
                else:
                    # Count how many unique tokens appear in search calls: we
                    # do not have full args here (content preview truncates),
                    # so approximate by looking at the number of distinct
                    # content-turn snippets that follow each search.
                    search_ratio = len(search_actions) / n_content
                    if 0.15 <= search_ratio <= 0.6 and edit_indices:
                        scores["Localization Efficiency"] = 1.0
                    elif search_ratio > 0.6:
                        scores["Localization Efficiency"] = -1.0
                        evidence.append(f"localization: search dominates ({search_ratio:.0%})")
                    else:
                        scores["Localization Efficiency"] = 0.0

        # --- Edit Justification ------------------------------------------------
        if "Edit Justification" in scores:
            n_edits = len(edit_indices)
            submission_bytes = int(trajectory.metadata.get("submission_bytes", 0) or 0)
            if n_edits == 0:
                scores["Edit Justification"] = -2.0
                evidence.append("edit_justification: no edit action observed")
            elif submission_bytes and submission_bytes < 200:
                scores["Edit Justification"] = -2.0
                evidence.append(f"edit_justification: submission tiny ({submission_bytes} B)")
            elif n_edits > 12:
                scores["Edit Justification"] = -1.0
                evidence.append(f"edit_justification: {n_edits} edit ops (thrash?)")
            elif 1 <= n_edits <= 6:
                scores["Edit Justification"] = 1.0
            else:
                scores["Edit Justification"] = 0.0

        # --- Verification Completeness ----------------------------------------
        if "Verification Completeness" in scores:
            claimed_done = any(
                phrase in c
                for c in contents
                for phrase in ("task complete", "all done", "submitted", "finished")
            )
            if not verify_indices:
                scores["Verification Completeness"] = -2.0 if claimed_done else -1.0
                if claimed_done:
                    evidence.append("verification: completion claimed with no test invocation")
            else:
                last_edit = max(edit_indices) if edit_indices else -1
                verify_after_edit = any(v > last_edit for v in verify_indices)
                if verify_after_edit and claimed_done:
                    scores["Verification Completeness"] = 2.0
                elif verify_after_edit:
                    scores["Verification Completeness"] = 1.0
                else:
                    scores["Verification Completeness"] = 0.0
                    evidence.append("verification: tests ran but not after final edit")

        # --- Error Recovery ---------------------------------------------------
        if "Error Recovery" in scores:
            if not error_indices:
                scores["Error Recovery"] = 1.0
            else:
                # Look at whether the very next action after an error is the
                # same command word (= naive retry) or something different.
                retries = 0
                for i in error_indices:
                    if i + 1 < len(actions) and i - 1 >= 0:
                        prev_action = actions[i - 1]
                        next_action = actions[i + 1]
                        if prev_action == next_action and prev_action:
                            retries += 1
                late_errors = sum(1 for i in error_indices if i > 2 * len(actions) // 3)
                if retries >= 3 or late_errors >= 3:
                    scores["Error Recovery"] = -2.0
                    evidence.append(
                        f"error_recovery: {retries} naive retries, {late_errors} late errors"
                    )
                elif retries >= 1:
                    scores["Error Recovery"] = -1.0
                else:
                    scores["Error Recovery"] = 0.0

        return scores, evidence, ScorerCost(llm_calls=0, llm_tokens=0)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


# Adversarial probe extremes: the scale endpoints every rubric uses, so a
# policy that reads scores at all sees the widest swing a real scorer could
# ever produce.
_PROBE_HIGH = 2.0
_PROBE_LOW = -2.0

_LAZY_MARKER = "lazy scoring: decision independent of rubric scores; scoring skipped"


def _probe_rubric_independence(
    policy_module,
    trajectory: Trajectory,
    criteria: Sequence[str],
    triggered: list[dict],
) -> dict | None:
    """Return decide()'s payload iff the decision cannot depend on rubric scores.

    Calls `decide()` with all-criteria +2, all-criteria -2, and an empty score
    dict. If any call raises (contained the same way the solver contains rule
    failures) or the three decisions differ, returns None and the caller
    scores normally. The returned payload is the empty-dict probe's, which
    matches the `rubric_scores={}` stored on the skipped row's record.
    """

    probes = (
        {c: _PROBE_HIGH for c in criteria},
        {c: _PROBE_LOW for c in criteria},
        {},
    )
    payloads: list[dict] = []
    for probe_scores in probes:
        try:
            # Deep-copied rules so a mutating policy cannot let a probe bleed
            # into the real decide() call that may follow.
            payload = policy_module.decide(trajectory, probe_scores, copy.deepcopy(triggered))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        payloads.append(payload)
    decisions = {str(p.get("decision", "REVIEW")).upper() for p in payloads}
    if len(decisions) != 1:
        return None
    return payloads[-1]


@dataclass
class Solver:
    """Applies a filter policy to a list of trajectories.

    `lazy_scoring` (default on) skips LLM rubric scoring for rows whose
    decision is provably score-independent — see the module docstring. Skipped
    rows carry `rubric_scores={}`, zero LLM cost, and a marker evidence line.
    """

    rubric_scorer: RubricScorer = field(default_factory=HeuristicRubricScorer)
    lazy_scoring: bool = True

    def run(
        self,
        loaded_filter: LoadedFilter,
        trajectories: Sequence[Trajectory],
    ) -> list[FilterDecision]:
        decisions: list[FilterDecision] = []
        unscoreable: list[str] = []
        criteria = getattr(loaded_filter.policy_module, "RUBRIC_CRITERIA", [])

        # Pass 1: rules run exactly once per row; their output feeds both the
        # lazy-scoring probe and the final decide().
        prepared: list[tuple[Trajectory, list[dict], dict | None]] = []
        for traj in trajectories:
            triggered = []
            for rule in loaded_filter.rule_functions:
                try:
                    fired = rule(traj)
                except Exception as exc:  # pragma: no cover - defensive
                    fired = {
                        "name": rule.__name__,
                        "verdict": "REVIEW",
                        "evidence": [f"rule raised {type(exc).__name__}: {exc}"],
                    }
                if fired:
                    # Prefer the function name so the trigger label matches the
                    # rules.py source-of-truth rather than a free-form label the
                    # rule chose to emit.
                    fired = {**fired, "name": rule.__name__}
                    triggered.append(fired)

            lazy_payload = None
            if self.lazy_scoring:
                lazy_payload = _probe_rubric_independence(
                    loaded_filter.policy_module, traj, criteria, triggered
                )
            prepared.append((traj, triggered, lazy_payload))

        to_score = [traj for traj, _, lazy_payload in prepared if lazy_payload is None]
        if self.lazy_scoring:
            logger.info(
                "lazy scoring: %d of %d rows decided without the rubric",
                len(prepared) - len(to_score),
                len(prepared),
            )
            # Warm the cache in parallel for exactly the rows that still need
            # a score. Best-effort: per-row failures resurface in the
            # sequential loop below, where they are contained row-by-row.
            prefetch = getattr(self.rubric_scorer, "prefetch_parallel", None)
            if to_score and callable(prefetch):
                try:
                    prefetch(list(to_score), criteria, loaded_filter.rubric_md)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "scorer prefetch failed (%s) — falling back to sequential scoring",
                        type(exc).__name__,
                    )

        for traj, triggered, lazy_payload in prepared:
            if lazy_payload is not None:
                decisions.append(
                    self._build_decision(
                        traj,
                        lazy_payload,
                        rubric_scores={},
                        rubric_evidence=[_LAZY_MARKER],
                        triggered=triggered,
                        cost=ScorerCost(),
                    )
                )
                continue

            try:
                rubric_scores, rubric_evidence, cost = self.rubric_scorer.score(
                    traj, criteria, loaded_filter.rubric_md
                )
            # One trajectory the model will not score must not end a
            # twenty-iteration run — the rule loop above already treats its own
            # failures this way. `bridgecrewio__checkov-755::att0` is a checkov
            # test fixture containing a literal private key, and Sonnet 5's
            # cyber safeguards refuse it (`stop_reason: refusal`) every time, so
            # retrying cannot help. Prefetch was already swallowing the same
            # error, which meant this sequential pass was not catching a new
            # problem, only deferring the crash to a less obvious place.
            #
            # REVIEW rather than a default score: the filter genuinely could not
            # evaluate this trajectory, and scoring it zero would make an
            # unreadable input look like a bad one.
            except Exception as exc:
                logger.warning(
                    "scorer could not score %s (%s) — routing to review",
                    traj.trajectory_id,
                    type(exc).__name__,
                )
                unscoreable.append(traj.trajectory_id)
                decisions.append(
                    FilterDecision(
                        trajectory_id=traj.trajectory_id,
                        decision=Decision.REVIEW,
                        confidence=0.0,
                        rubric_scores={},
                        triggered_rules=[r.get("name", "?") for r in triggered],
                        evidence=[f"scorer unavailable: {type(exc).__name__}: {exc}"[:300]],
                        reasoning_summary="the rubric scorer could not score this trajectory",
                        llm_calls=0,
                        llm_tokens=0,
                        # Marks the row as unscoreable rather than merely
                        # unscored, which is what lets ranking drop it from
                        # everyone's exam instead of charging it to this
                        # candidate. True for a refusal as well as for an
                        # unreadable answer: in both cases no model judged
                        # the row, and neither is the filter's doing.
                        scorer_failed=True,
                    )
                )
                continue
            payload = loaded_filter.policy_module.decide(traj, rubric_scores, triggered)
            decisions.append(
                self._build_decision(
                    traj,
                    payload,
                    rubric_scores=rubric_scores,
                    rubric_evidence=rubric_evidence,
                    triggered=triggered,
                    cost=cost,
                )
            )

        # The unscoreable fraction is over rows that actually needed scoring:
        # a lazily-decided row could not have been refused, so counting it in
        # the denominator would dilute a real scoring outage.
        self._report_unscoreable(unscoreable, len(to_score))
        return decisions

    @staticmethod
    def _build_decision(
        traj: Trajectory,
        payload: dict,
        *,
        rubric_scores: dict[str, float],
        rubric_evidence: list[str],
        triggered: list[dict],
        cost: ScorerCost,
    ) -> FilterDecision:
        """Wrap a decide() payload into a FilterDecision record."""

        decision_str = str(payload.get("decision", "REVIEW")).upper()
        try:
            decision_enum = Decision(decision_str)
        except ValueError:
            decision_enum = Decision.REVIEW

        evidence = list(payload.get("evidence", []) or []) + rubric_evidence
        raw_score = payload.get("score")
        try:
            rank_score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            rank_score = None
        return FilterDecision(
            trajectory_id=traj.trajectory_id,
            decision=decision_enum,
            confidence=float(payload.get("confidence", 0.0)),
            rank_score=rank_score,
            rubric_scores=rubric_scores,
            triggered_rules=[r.get("name", "?") for r in triggered],
            evidence=evidence,
            reasoning_summary=str(payload.get("reasoning_summary", "")),
            llm_calls=cost.llm_calls,
            llm_tokens=cost.llm_tokens,
        )

    @staticmethod
    def _report_unscoreable(unscoreable: list[str], total: int) -> None:
        """Say how much of the pool could not be scored, and stop if it is most of it."""

        if not unscoreable or not total:
            return
        fraction = len(unscoreable) / total
        if fraction >= UNSCOREABLE_FATAL_FRACTION:
            raise RuntimeError(
                f"{len(unscoreable)} of {total} trajectories ({fraction:.0%}) could not "
                f"be scored — this filter's metrics do not describe its quality. "
                f"First few: {', '.join(unscoreable[:5])}"
            )
        level = logging.ERROR if fraction >= UNSCOREABLE_WARN_FRACTION else logging.INFO
        logger.log(
            level,
            "%d of %d trajectories (%.1f%%) were unscoreable and went to review: %s",
            len(unscoreable),
            total,
            fraction * 100,
            ", ".join(unscoreable[:5]),
        )
