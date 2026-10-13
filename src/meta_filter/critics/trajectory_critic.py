"""LLM-based CriticBackend.

Replaces the gold-label look-up with a real judgment: the backend prompts a
strong LLM with a compact view of the trajectory and asks for
ACCEPT / REJECT / ABSTAIN. Judgments are filesystem-cached by trajectory_id
so repeated iterations do not re-query the model.

The backend also supports **disagreement-based labeling**: the optimizer can
prefetch judgments only for trajectories where candidate filters actually
disagree, saving labeling budget for information-rich cases.

Two kinds of critic live in this package, and they are not peers:

  * The **trajectory critic** (this module) rules on one TRAJECTORY at a time
    — is this worth keeping as training data. Its verdicts stand in for the
    gold labels the project does not have, which is why every `critic_*`
    metric (`critic_agreement`, `filter_accept_critic_reject`, …) is measured
    against them. It is the yardstick.
  * The **filter critics** (`rubric_critic`, `rule_critic`) review the
    FILTER's design once per iteration and never see a trajectory. Their
    output is a diagnostic (weight 0 in the score), not a yardstick.

Never call this one's output ground truth: it is a pseudo gold opinion, and a
filter that agrees with it is agreeing with a second reader, not with facts.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from meta_filter.core.rendering import render_submission, render_turns, view_signature
from meta_filter.core.schemas import Decision, FilterDecision, Trajectory
from meta_filter.llm.base import LLMClient
from meta_filter.visibility.view_policy import pair_view_signature

logger = logging.getLogger(__name__)

# The pseudo gold judge is the expensive opinion the whole loop is distilling,
# so it gets the widest view we can afford.
JUDGE_TOOL_BUDGET = 60_000

_VALID_MISFIRE = ("rubric", "rules", "policy", "none")


@dataclass(frozen=True)
class TrajectoryVerdict:
    """Everything the judge said about one trajectory.

    `decision is None` covers both cache misses and the ABSTAIN → None mapping;
    callers should use `TrajectoryCritic.verdict_for` which returns None
    when nothing is cached at all and a TrajectoryVerdict otherwise.
    """

    trajectory_id: str
    decision: Decision | None
    rationale: str = ""
    closest_candidate: str = "none"
    misfire: str = "none"
    misfire_note: str = ""


JUDGE_SYSTEM = (
    "You are a data-quality auditor for a project that trains SWE (software "
    "engineering) coding agents. The team is filtering agent rollouts to keep "
    "only trajectories that a general-purpose LLM could learn good SWE "
    "behavior from. We do NOT have gold labels — you are the second opinion "
    "on the filters' own reasoning. Each candidate filter uses a shared "
    "rubric prompt (rubric.md) and a set of deterministic checks (rules.py). "
    "Your job is to (a) decide whether the trajectory should be kept as "
    "training data, and (b) if the filters got it wrong, name which layer was "
    "fooled: the rubric prompt, the deterministic rules, or the aggregation "
    "policy. Output exactly one JSON object; no prose outside it."
)

JUDGE_PROMPT = """\
Audit the following SWE agent trajectory.

Task:
{task}

Trajectory metadata:
{metadata}

Compact action histogram (top actions; `tool_result` rows are shell responses):
{histogram}

Full transcript. Agent reasoning and commands are verbatim; long tool output
is elided in the middle, keeping the head (what ran) and the tail (how it
ended):
{turns_block}

Final submitted patch:
{submission}
{candidate_block}

Judge the trajectory as a senior SWE reviewer:
- Did the agent understand the task and address the root cause (not just a
  symptom)?
- Is the submitted patch minimal, focused, and likely correct — or does it
  thrash, hallucinate files, or ship half-finished work?
- Would a strong SWE model actually learn good behavior from imitating this
  trajectory?

If candidate filter verdicts are shown, treat them as opinions to weigh, NOT
as authoritative — you may disagree with all of them. When you disagree with
the closest candidate, identify which layer misfired:
- "rubric"  — the rubric prompt named the wrong signals, or an anchor is vague
- "rules"   — a deterministic check over- or under-fired on this trajectory
- "policy"  — the per-criterion scores were fine but the aggregation flipped
              the verdict (e.g. threshold too strict / lenient)
- "none"    — filters agreed with you, no misfire

Return exactly this JSON (no other text):
{{"decision": "ACCEPT" | "REJECT" | "ABSTAIN",
  "rationale": "one sentence",
  "closest_candidate": "filter_id or 'none'",
  "misfire": "rubric" | "rules" | "policy" | "none",
  "misfire_note": "short explanation of which layer was fooled, or empty"}}"""


def _action_histogram(traj: Trajectory) -> dict[str, int]:
    from collections import Counter

    c: Counter[str] = Counter()
    for t in traj.turns:
        key = (t.action or t.role or "").strip()
        if key:
            c[key] += 1
    return dict(c.most_common(10))


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", name)[:200]


@dataclass
class TrajectoryCritic:
    """Pseudo gold judge — a strong LLM asked for a second opinion, cached
    on disk. Its verdicts are opinions, not labels."""

    client: LLMClient
    cache_dir: Path = Path("data/judge_cache")
    model_tag: str = ""
    abstain_maps_to: Decision | None = None
    _mem_cache: dict[str, Decision | None] = field(default_factory=dict)
    # Full TrajectoryVerdict per trajectory_id (populated whenever cache is read
    # or written). Kept alongside `_mem_cache` so existing Decision-only
    # callers stay fast while misfire-aware callers can pull the details.
    _verdict_cache: dict[str, TrajectoryVerdict] = field(default_factory=dict)
    # trajectory_id -> grading-label view signature, empty views omitted.
    _label_views: dict[str, str] = field(default_factory=dict)
    _new_calls: int = 0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.model_tag:
            self.model_tag = getattr(self.client, "model", "unknown")

    # ------------------------------------------------------------- protocol

    def critic_verdict(
        self,
        trajectory: Trajectory,
        candidate_decisions: list[tuple[str, FilterDecision]] | None = None,
    ) -> Decision | None:
        self.register_label_views([trajectory])
        traj_id = trajectory.trajectory_id
        cached = self._lookup_cached(traj_id)
        if cached is not _MISSING:
            return cached
        return self._query_and_cache(trajectory, candidate_decisions)

    # -------------------------------------------- disagreement-based prefetch

    def prefetch(
        self,
        trajectories: list[Trajectory],
        candidate_context: dict[str, list[tuple[str, FilterDecision]]] | None = None,
    ) -> int:
        """Ensure a judge verdict exists for every trajectory in the list.
        Returns the number of *new* LLM calls made."""

        self.register_label_views(trajectories)
        made = 0
        candidate_context = candidate_context or {}
        for t in trajectories:
            if self._lookup_cached(t.trajectory_id) is _MISSING:
                self._query_and_cache(t, candidate_context.get(t.trajectory_id))
                made += 1
        return made

    def prefetch_parallel(
        self,
        trajectories: list[Trajectory],
        *,
        max_workers: int = 8,
        candidate_context: dict[str, list[tuple[str, FilterDecision]]] | None = None,
    ) -> int:
        """Same as `prefetch` but runs missing judgments concurrently."""

        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.register_label_views(trajectories)
        pending = [t for t in trajectories if self._lookup_cached(t.trajectory_id) is _MISSING]
        if not pending:
            return 0
        candidate_context = candidate_context or {}
        logger.info(
            "LLM judge prefetch: %d new + %d cached (parallel workers=%d)",
            len(pending),
            len(trajectories) - len(pending),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self._query_and_cache,
                    t,
                    candidate_context.get(t.trajectory_id),
                ): t
                for t in pending
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "judge failed for %s: %s",
                        futures[fut].trajectory_id,
                        exc,
                    )
        return len(pending)

    # -------------------------------------------------- trajectory curation

    def select_for_proposer(
        self,
        *,
        false_negative_ids: list[str],
        false_positive_ids: list[str],
        audit_mismatch_ids: list[str],
        candidate_trajectory_ids: list[str],
        seed: int,
        budget: int = 6,
    ) -> list[str]:
        """Return up to `budget` trajectory ids for the proposer's evidence pack.

        Bucket layout:
            recent FN (baseline)              up to 2
            recent FP (baseline)              up to 2
            consensus-audit mismatch          up to 1
            judge-chosen instructive          up to 1

        When a bucket is empty, its slot is redistributed to the largest
        remaining bucket. Selection is deterministic under `seed`.

        `candidate_trajectory_ids` is the pool from which the "instructive"
        case is chosen — typically the current iteration's `active_train` ids.
        Only cached judgments with `misfire != "none"` qualify.
        Tie-break inside the instructive bucket prefers misfire tier
        rubric > rules > policy so the proposer sees rubric-critical
        evidence first.
        """

        rng = random.Random(seed)

        fn_pool = _dedup_ids(false_negative_ids)
        fp_pool = _dedup_ids(false_positive_ids)
        audit_pool = _dedup_ids(audit_mismatch_ids)
        instructive_pool = self._instructive_pool(candidate_trajectory_ids, rng)

        # Each caller-supplied pool is already ordered by "how much we want it":
        #   FN / FP / audit — caller passes most-recent first.
        #   instructive     — rubric > rules > policy (tier order).
        # So the picker takes head-of-list rather than random-sampling; this
        # preserves both "recent first" and "rubric first" without needing an
        # extra tie-break rule.
        already: set[str] = set()

        def _take(pool: list[str], cap: int) -> list[str]:
            picked: list[str] = []
            for tid in pool:
                if len(picked) >= cap:
                    break
                if tid in already:
                    continue
                picked.append(tid)
                already.add(tid)
            return picked

        caps = [
            ("fn", fn_pool, 2),
            ("fp", fp_pool, 2),
            ("audit", audit_pool, 1),
            ("instructive", instructive_pool, 1),
        ]

        selected: list[str] = []
        for _, pool, cap in caps:
            if len(selected) >= budget:
                break
            room = min(cap, budget - len(selected))
            selected += _take(pool, room)

        # Backfill: give leftover budget to whichever remaining bucket has the
        # most unused items. Deterministic because pool order and bucket order
        # are both fixed.
        while len(selected) < budget:
            remaining = [(name, [x for x in pool if x not in already]) for name, pool, _ in caps]
            remaining = [(n, p) for n, p in remaining if p]
            if not remaining:
                break
            remaining.sort(key=lambda np: (-len(np[1]), np[0]))
            _, largest = remaining[0]
            selected.append(largest[0])
            already.add(largest[0])

        return selected

    def _instructive_pool(
        self,
        candidate_trajectory_ids: list[str],
        rng: random.Random,
    ) -> list[str]:
        """Cached trajectories the judge flagged as filter misfires.

        Ranked by misfire tier (rubric > rules > policy) then shuffled inside
        each tier with the caller's RNG so callers stay deterministic under
        seed.
        """

        tiers: dict[str, list[str]] = {"rubric": [], "rules": [], "policy": []}
        for tid in _dedup_ids(candidate_trajectory_ids):
            verdict = self.verdict_for(tid)
            if verdict is None or verdict.misfire not in tiers:
                continue
            tiers[verdict.misfire].append(tid)
        ordered: list[str] = []
        for tier_name in ("rubric", "rules", "policy"):
            bucket = tiers[tier_name]
            if not bucket:
                continue
            # rng.sample gives a deterministic shuffle without mutating the
            # source list; use it inside each tier so the tier ranking is
            # preserved but ties are broken deterministically.
            ordered += rng.sample(bucket, len(bucket))
        return ordered

    # ---------------------------------------------------------------- internal

    def register_label_views(self, trajectories: Iterable[Trajectory]) -> None:
        """Record each trajectory's grading-label view before any cache lookup.

        Called from every entry point that receives `Trajectory` objects, so
        the id-keyed lookups below can reach the view without threading it
        through six signatures. An id never registered falls back to the
        view-free filename, which is correct for an all-fields-present row and is
        why the population points are the *first* thing each entry point does.
        """

        for traj in trajectories:
            sig = pair_view_signature(traj)
            if sig:
                self._label_views[traj.trajectory_id] = sig

    def cache_path_for(self, traj_id: str) -> Path:
        """Cache file for a trajectory, scoped to the judge and what it read.

        The model is in the name because verdicts from two backends are not
        interchangeable, and the view signature is in it because a verdict
        formed on a 2%-of-trajectory excerpt is not the same opinion as one
        formed on the full transcript.

        The grading-label view is in it for the third instance of that same
        reason. `critics.agentic._trajectory_payload` hands this judge
        `dict(trajectory.metadata)` — the whole dict, no allow-list — so its
        verdicts are the most view-sensitive in the project. Without the view
        in the key, verdicts on a label-stripped pool whose trajectory ids
        match a labelled pool's would land under labelled-looking names and
        could not be told apart. Empty for the all-present case, so every
        verdict already bought keeps its filename.
        """

        view = view_signature(JUDGE_TOOL_BUDGET)
        stem = f"{_safe_name(traj_id)}__{_safe_name(self.model_tag)}__{view}"
        lv = self._label_views.get(traj_id)
        if lv is not None:
            return self.cache_dir / f"{stem}__lv_{lv}.json"

        # No view registered: this reader was handed an id and never the
        # trajectory, which `verdict_for` legitimately is — offline scoring
        # and `--resume` both do it. Writing is always registered, so an
        # unregistered read must not guess a view; it accepts the one verdict
        # on file for this id and refuses to choose between several.
        unviewed = self.cache_dir / f"{stem}.json"
        if unviewed.exists():
            return unviewed
        matches = sorted(self.cache_dir.glob(f"{stem}__lv_*.json"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning(
                "%s has verdicts under %d different label views and this "
                "reader was given no trajectory to disambiguate; treating it "
                "as absent rather than picking one",
                traj_id,
                len(matches),
            )
        return unviewed

    def _lookup_cached(self, traj_id: str):
        if traj_id in self._mem_cache:
            return self._mem_cache[traj_id]
        cache_path = self.cache_path_for(traj_id)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("bad cache entry for %s, will re-query", traj_id)
                return _MISSING
            decision = _decision_from_cache(data)
            self._mem_cache[traj_id] = decision
            self._verdict_cache[traj_id] = TrajectoryVerdict(
                trajectory_id=traj_id,
                decision=decision,
                rationale=str(data.get("rationale") or ""),
                closest_candidate=str(data.get("closest_candidate") or "none"),
                misfire=_coerce_misfire(data.get("misfire")),
                misfire_note=str(data.get("misfire_note") or ""),
            )
            return decision
        return _MISSING

    def verdict_for(self, traj_id: str) -> TrajectoryVerdict | None:
        """Full judge verdict for a trajectory, or None if not cached."""

        if traj_id in self._verdict_cache:
            return self._verdict_cache[traj_id]
        cached = self._lookup_cached(traj_id)
        if cached is _MISSING:
            return None
        return self._verdict_cache.get(traj_id)

    def _query_and_cache(
        self,
        trajectory: Trajectory,
        candidate_decisions: list[tuple[str, FilterDecision]] | None = None,
    ) -> Decision | None:
        verdict, raw = self._query(trajectory, candidate_decisions)
        cache_path = self.cache_path_for(trajectory.trajectory_id)
        cache_path.write_text(
            json.dumps(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "decision": verdict.decision.value if verdict.decision else None,
                    "rationale": verdict.rationale,
                    "closest_candidate": verdict.closest_candidate,
                    "misfire": verdict.misfire,
                    "misfire_note": verdict.misfire_note,
                    "model": self.model_tag,
                    "raw": raw,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._mem_cache[trajectory.trajectory_id] = verdict.decision
        self._verdict_cache[trajectory.trajectory_id] = verdict
        self._new_calls += 1
        return verdict.decision

    def _query(
        self,
        trajectory: Trajectory,
        candidate_decisions: list[tuple[str, FilterDecision]] | None,
    ) -> tuple[TrajectoryVerdict, str]:
        prompt = JUDGE_PROMPT.format(
            task=(trajectory.task or "(no task provided)")[:1000],
            metadata=json.dumps(
                {
                    k: v
                    for k, v in trajectory.metadata.items()
                    if k in ("api_calls", "n_messages", "submission_bytes", "source")
                },
                indent=2,
            ),
            histogram=_action_histogram(trajectory),
            turns_block=render_turns(trajectory, tool_output_budget=JUDGE_TOOL_BUDGET),
            submission=render_submission(trajectory),
            candidate_block=_format_candidate_block(candidate_decisions),
        )
        raw = self._ask(prompt, trajectory)
        parsed = _parse_judge_response(raw)
        decision = parsed["decision"]
        if decision == "ABSTAIN":
            decision = self.abstain_maps_to
        verdict = TrajectoryVerdict(
            trajectory_id=trajectory.trajectory_id,
            decision=decision,
            rationale=parsed["rationale"],
            closest_candidate=parsed["closest_candidate"],
            misfire=parsed["misfire"],
            misfire_note=parsed["misfire_note"],
        )
        return verdict, raw

    def _ask(self, prompt: str, trajectory: Trajectory) -> str:
        """The single LLM call. Overridden by the agentic backend, which
        gives the model tools and the full trajectory on disk instead."""

        _ = trajectory
        return self.client.generate(prompt, system=JUDGE_SYSTEM, temperature=0.0).text


def _format_candidate_block(
    candidate_decisions: list[tuple[str, FilterDecision]] | None,
) -> str:
    if not candidate_decisions:
        return ""
    lines = ["", "Candidate filter verdicts (opinions, not gold):"]
    for cand_id, d in candidate_decisions:
        lines.append(
            f"- filter_id={cand_id}  decision={d.decision.value}  confidence={d.confidence:.2f}"
        )
        if d.rubric_scores:
            lines.append(f"    rubric_scores: {d.rubric_scores}")
        if d.triggered_rules:
            lines.append(f"    triggered_rules: {d.triggered_rules}")
        if d.reasoning_summary:
            lines.append(f"    reasoning: {d.reasoning_summary[:250]}")
    return "\n".join(lines)


# --------------------------------------------------------------------- helpers


class _Missing:
    def __repr__(self):
        return "<missing>"


_MISSING = _Missing()


def _decision_from_cache(data: dict) -> Decision | None:
    dec = data.get("decision")
    if dec in ("ACCEPT", "REJECT", "REVIEW"):
        return Decision(dec)
    # cached ABSTAIN or explicit null → unlabeled
    return None


def _coerce_misfire(value) -> str:
    v = str(value or "none").lower().strip()
    return v if v in _VALID_MISFIRE else "none"


def _dedup_ids(seq) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_judge_response(text: str) -> dict:
    """Extract judge fields from a raw model response.

    Always returns a dict with keys `decision`, `rationale`, `closest_candidate`,
    `misfire`, `misfire_note`. `decision` is a `Decision`, the literal string
    "ABSTAIN", or None if the response was unparseable.
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

    def _pack(decision, rationale: str) -> dict:
        return {
            "decision": decision,
            "rationale": rationale[:400],
            "closest_candidate": "none",
            "misfire": "none",
            "misfire_note": "",
        }

    if isinstance(obj, dict):
        dec_str = str(obj.get("decision", "")).upper().strip()
        rationale = str(obj.get("rationale", "") or "")[:400]
        closest = str(obj.get("closest_candidate", "") or "none")[:80]
        misfire = _coerce_misfire(obj.get("misfire"))
        misfire_note = str(obj.get("misfire_note", "") or "")[:400]
        decision: Decision | str | None
        if dec_str == "ACCEPT":
            decision = Decision.ACCEPT
        elif dec_str == "REJECT":
            decision = Decision.REJECT
        elif dec_str == "ABSTAIN":
            decision = "ABSTAIN"
        else:
            decision = None
        return {
            "decision": decision,
            "rationale": rationale,
            "closest_candidate": closest or "none",
            "misfire": misfire,
            "misfire_note": misfire_note,
        }
    upper = text.upper()
    if "REJECT" in upper and "ACCEPT" not in upper:
        return _pack(Decision.REJECT, text[:200])
    if "ACCEPT" in upper and "REJECT" not in upper:
        return _pack(Decision.ACCEPT, text[:200])
    return _pack(None, text[:200])
