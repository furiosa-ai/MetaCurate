"""Pairwise trajectory adjudication.

The loop has no gold ordering: pairs every candidate orders the same way
are free internal pseudo-truth, and the pairs candidates disagree on come
here — an LLM reads both trajectories and says which one is better SFT
training data. Pairwise choice is a far easier task for a judge than
absolute scoring, which is exactly why the ranking loop is supervised on
pairs rather than on judge-produced scores.

Every pair is adjudicated TWICE — once with each trajectory presented as A —
and only a pair both orders name the same winner for is resolved. The rest
are TIE and leave the truth set entirely. Position bias in a pairwise LLM
judge is large enough that a single-order verdict is partly a measurement of
which slot a trajectory landed in, and that noise entered the objective as
signal.

Verdicts are cached per ordered presentation, model, and view signature, so
re-running an iteration or resuming a run re-pays nothing. A cached record is
only reused for the presentation it actually answered: every record names the
trajectory that sat in slot A, and one that does not — every file written
before the two-order rule — is not a verdict about either order. Those files
remain readable for their prose (`rationale_for`) and are re-judged for both
orders, which is the honest price of not knowing how they were asked.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from meta_filter.rendering import render_submission, render_turns, view_signature
from meta_filter.schemas import Trajectory
from meta_filter.view_policy import pair_view_signature

logger = logging.getLogger(__name__)

# Two full transcripts share one prompt, so each side gets a smaller tool
# budget than the single-trajectory judge's 60K.
PAIR_TOOL_BUDGET = 20_000

# How many times one presentation is asked before the pair is abandoned for
# this run. Only unparseable answers are retried; a transport error is not.
PARSE_ATTEMPTS = 2

PAIR_PROMPT = """\
You are comparing two SWE agent trajectories as candidate SFT training data.
Both solved (or attempted) their own task; the question is which one a strong
SWE model would learn MORE good behavior from imitating — diagnosis quality,
patch quality, verification discipline, efficiency. Judge the process, not
just the outcome.

=== TRAJECTORY A ({id_a}) ===
Task:
{task_a}

Harness metadata: {meta_a}

Transcript (long tool output elided in the middle):
{turns_a}

Final submitted patch:
{submission_a}

=== TRAJECTORY B ({id_b}) ===
Task:
{task_b}

Harness metadata: {meta_b}

Transcript (long tool output elided in the middle):
{turns_b}

Final submitted patch:
{submission_b}

Which trajectory is better SFT training data?
Return exactly this JSON (no other text):
{{"winner": "A" | "B" | "TIE", "rationale": "one sentence"}}"""


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", name)[:120]


# The "Harness metadata:" line of each trajectory block in `PAIR_PROMPT`: a
# fixed line, the same for every trajectory, kept for prompt-format stability
# (it is part of the prompt every cached verdict was formed on). It carries no
# information about the row.
PAIR_METADATA_LINE = '{"resolved": null, "f2p_pass": null, "p2p_pass": null, "exit_status": null}'


def _meta_line(traj: Trajectory) -> str:
    """The metadata line for one trajectory's block (always `PAIR_METADATA_LINE`)."""

    _ = traj
    return PAIR_METADATA_LINE


def _rationale_from_raw(raw: str) -> str:
    """The judge's one-line rationale out of a raw answer, or ''."""

    m = re.search(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"', raw or "")
    return m.group(1).replace('\\"', '"') if m else ""


def _parse_winner(raw: str) -> str | None:
    m = re.search(r'"winner"\s*:\s*"(A|B|TIE)"', raw)
    return m.group(1) if m else None


# Which trajectory of the sorted pair was presented as A. Part of the cache
# key, so the two passes cannot overwrite each other.
ORDER_FORWARD = "ab"
ORDER_REVERSED = "ba"

# What `judge_both_orders` concluded. TIE and UNRESOLVED both leave the truth
# set, but they mean different things and only one of them is a judge opinion.
RESOLVED = "resolved"
TIE = "tie"
UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class PairResolution:
    """The outcome of adjudicating a batch of unordered pairs.

    `resolved` are ordered ``(winner, loser)`` pairs both presentation orders
    agreed on. `tied` are pairs the judge answered in both orders without
    agreeing — a verdict, and the thing concordance must exclude rather than
    score at 0.5. `unresolved` are pairs a call failed on or an answer could
    not be parsed for: no verdict at all.
    """

    resolved: list[tuple[str, str]] = field(default_factory=list)
    tied: list[tuple[str, str]] = field(default_factory=list)
    unresolved: list[tuple[str, str]] = field(default_factory=list)

    @property
    def tie_share(self) -> float:
        judged = len(self.resolved) + len(self.tied)
        return len(self.tied) / judged if judged else 0.0


@dataclass
class PairJudge:
    """LLM adjudicator over unordered trajectory pairs, cached on disk."""

    client: object
    cache_dir: Path = Path("data/pair_judge_cache")

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Answers this judge could not parse even after a retry. Counted
        # rather than cached, so the number is per-run and a later run gets
        # to ask again. `resolve_pairs_detail` logs it.
        self.parse_failures = 0
        self._counter_lock = threading.Lock()

    def _cache_path(
        self,
        id_a: str,
        id_b: str,
        label_view: str = "",
        order: str = "",
    ) -> Path:
        """Where one ordered presentation of one pair lives.

        `label_view` records which grading fields the rows lacked, and is
        empty only when every field is present, so a key without the
        component and a key with it can never collide. See
        `view_policy.pair_view_signature` for why the key needs it at all.

        `order` records which of the two trajectories was presented as A.
        Empty is the single-order key: verdicts written before the two-order
        rule, which are read but never written.
        """

        lo, hi = sorted((id_a, id_b))
        model = getattr(self.client, "model", "unknown")
        view = view_signature(PAIR_TOOL_BUDGET)
        lv = f"__lv_{label_view}" if label_view else ""
        od = f"__ord_{order}" if order else ""
        return self.cache_dir / (
            f"{_safe_name(lo)}__vs__{_safe_name(hi)}__{_safe_name(model)}__{view}{lv}{od}.json"
        )

    @staticmethod
    def _read_verdict(cache: Path, expected_first: str) -> tuple[str | None, bool] | None:
        """`(winner_id, decided)` from a cache file, or None if unusable.

        `decided` separates "the judge said TIE" from "the call never
        produced an answer".

        `expected_first` is the trajectory that must have been in slot A for
        this record to answer the question being asked. A record that does not
        say which slot each side occupied cannot answer it at all, so it is
        rejected — see `_judge_one_order` for what that costs and why.
        """

        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("shown_as_a") != expected_first:
            return None
        winner = data.get("winner_id")
        letter = data.get("letter")
        decided = letter in ("A", "B", "TIE") or winner is not None
        return winner, decided

    def _judge_one_order(
        self,
        first: Trajectory,
        second: Trajectory,
        *,
        label_view: str,
        order: str,
    ) -> tuple[str | None, bool]:
        """Adjudicate one presentation: `first` is A, `second` is B.

        Returns `(winner_id, decided)`. `winner_id` is None both for a TIE
        and for an answer that could not be parsed; `decided` tells them
        apart.
        """

        cache = self._cache_path(first.trajectory_id, second.trajectory_id, label_view, order)
        if cache.exists():
            hit = self._read_verdict(cache, first.trajectory_id)
            if hit is not None:
                return hit
        # Cache files written before the two-order rule are NOT reused here,
        # even for the forward pass. They record a winner and not the slot
        # each side was shown in, and the order they were asked in is not
        # recoverable from anything: consensus and audit pairs were appended
        # to the judge queue in the candidates' score order (`ranking.py`),
        # so a single-order record is as likely to be a reverse presentation as a
        # forward one. Seeding the forward slot with one would let a
        # position-biased judge agree with itself across two different
        # presentations of the same pair and report "resolved" where the
        # honest answer is TIE — the exact failure the two-order rule exists
        # to catch. They stay readable for `rationale_for`, which wants prose
        # rather than a verdict, and both orders are paid for fresh.

        prompt = PAIR_PROMPT.format(
            id_a=first.trajectory_id,
            task_a=(first.task or "(no task)")[:2000],
            meta_a=_meta_line(first),
            turns_a=render_turns(first, tool_output_budget=PAIR_TOOL_BUDGET),
            submission_a=render_submission(first),
            id_b=second.trajectory_id,
            task_b=(second.task or "(no task)")[:2000],
            meta_b=_meta_line(second),
            turns_b=render_turns(second, tool_output_budget=PAIR_TOOL_BUDGET),
            submission_b=render_submission(second),
        )
        # One retry on an unparseable answer, same prompt. An unparseable
        # response is a transport-shaped failure, not a verdict, and writing
        # it to the cache would freeze it: every later run sharing this cache
        # reads the record, finds no letter, and calls the pair UNRESOLVED
        # forever without ever asking again.
        raw = ""
        letter = None
        for attempt in range(PARSE_ATTEMPTS):
            try:
                raw = self.client.generate(prompt).text
            except Exception as exc:  # noqa: BLE001 — a lost pair is a skipped pair
                logger.warning(
                    "pair judge failed for %s vs %s (order %s): %s",
                    first.trajectory_id,
                    second.trajectory_id,
                    order,
                    type(exc).__name__,
                )
                return None, False
            letter = _parse_winner(raw)
            if letter is not None:
                break
            logger.warning(
                "pair judge returned an unparseable verdict for %s vs %s "
                "(order %s), attempt %d of %d",
                first.trajectory_id,
                second.trajectory_id,
                order,
                attempt + 1,
                PARSE_ATTEMPTS,
            )
        if letter is None:
            with self._counter_lock:
                self.parse_failures += 1
            logger.warning(
                "pair judge gave up on %s vs %s (order %s) after %d "
                "unparseable answers; nothing cached, the pair stays askable",
                first.trajectory_id,
                second.trajectory_id,
                order,
                PARSE_ATTEMPTS,
            )
            return None, False
        winner_id = {
            "A": first.trajectory_id,
            "B": second.trajectory_id,
        }.get(letter or "")
        cache.write_text(
            json.dumps(
                {
                    "pair": sorted((first.trajectory_id, second.trajectory_id)),
                    # Which slot each side was shown in, so a cached record
                    # can be audited for position bias after the fact.
                    "shown_as_a": first.trajectory_id,
                    "shown_as_b": second.trajectory_id,
                    "order": order,
                    "winner_id": winner_id,
                    "letter": letter,
                    "model": getattr(self.client, "model", "unknown"),
                    # Recorded as well as keyed: without this field, verdicts
                    # formed on a label-free view would be unidentifiable
                    # once they were in a shared cache.
                    "label_view": label_view,
                    "raw": raw[:2000],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return winner_id, True

    def judge_both_orders(self, traj_a: Trajectory, traj_b: Trajectory) -> tuple[str | None, str]:
        """Adjudicate one pair in both A/B orders.

        Returns `(winner_id, status)`. The winner is non-None only when both
        presentations named the same trajectory; otherwise the status says
        whether the judge disagreed with itself (`TIE`) or never answered
        (`UNRESOLVED`).
        """

        label_view = pair_view_signature(traj_a, traj_b)
        lo_id, _hi_id = sorted((traj_a.trajectory_id, traj_b.trajectory_id))
        lo, hi = (traj_a, traj_b) if traj_a.trajectory_id == lo_id else (traj_b, traj_a)

        forward, forward_decided = self._judge_one_order(
            lo, hi, label_view=label_view, order=ORDER_FORWARD
        )
        reverse, reverse_decided = self._judge_one_order(
            hi, lo, label_view=label_view, order=ORDER_REVERSED
        )
        if not (forward_decided and reverse_decided):
            return None, UNRESOLVED
        if forward is not None and forward == reverse:
            return forward, RESOLVED
        return None, TIE

    def judge(self, traj_a: Trajectory, traj_b: Trajectory) -> str | None:
        """The winning trajectory_id, or None for a tie / failure.

        Order-agnostic by construction: both presentations are asked and only
        an agreement counts.
        """

        return self.judge_both_orders(traj_a, traj_b)[0]

    def rationale_for(self, id_a: str, id_b: str, label_view: str = "") -> str:
        """The judge's one-line rationale for a cached pair, or ''.

        Read-only: never triggers a judgment. Used by the pair-evidence
        channel, which only ever asks about pairs already adjudicated. The
        forward pass is preferred, then the reverse one, then a
        single-order record — any of the three is the same judge on the same
        pair, and the caller wants prose, not a verdict.
        """

        for order in (ORDER_FORWARD, ORDER_REVERSED, ""):
            cache = self._cache_path(id_a, id_b, label_view, order)
            if not cache.exists():
                continue
            try:
                raw = json.loads(cache.read_text(encoding="utf-8")).get("raw", "")
            except (OSError, json.JSONDecodeError):
                continue
            prose = _rationale_from_raw(raw)
            if prose:
                return prose
        return ""

    def presentations_for(self, id_a: str, id_b: str, label_view: str = "") -> list[dict]:
        """Both cached presentations of one pair, read-only.

        One entry per presentation that has a cache file, forward (`ab`, the
        alphabetically lower id in slot A) first. Each names which trajectory
        sat in which slot, who won it, whether the call was decided at all,
        and the rationale that presentation gave.

        Nothing is judged here and no call is made: a presentation with no
        cache file is simply absent, which is what an unanswered pair and a
        never-asked pair both look like. This exists because the verdicts
        themselves live outside the run directory — the proposer's workspace
        can hold what the judge decided only if something copies it in.
        """

        out: list[dict] = []
        for order in (ORDER_FORWARD, ORDER_REVERSED):
            cache = self._cache_path(id_a, id_b, label_view, order)
            if not cache.exists():
                continue
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            letter = data.get("letter")
            out.append(
                {
                    "order": order,
                    "shown_as_a": data.get("shown_as_a"),
                    "shown_as_b": data.get("shown_as_b"),
                    "letter": letter,
                    "winner": data.get("winner_id"),
                    "decided": letter in ("A", "B", "TIE") or data.get("winner_id") is not None,
                    "rationale": _rationale_from_raw(data.get("raw", "")),
                }
            )
        return out

    def resolve_pairs_detail(
        self,
        pairs: list[tuple[str, str]],
        traj_by_id: dict[str, Trajectory],
        *,
        max_workers: int = 8,
    ) -> PairResolution:
        """Adjudicate unordered pairs in both orders; report all three outcomes.

        Missing trajectory ids are skipped entirely — they are not a judge
        opinion and must not dilute the TIE share.
        """

        usable = [(a, b) for a, b in pairs if a in traj_by_id and b in traj_by_id]
        # Say it at the moment it matters: how many of this batch were judged
        # on a view other than "all three grading fields present".
        non_baseline = sum(
            1 for a, b in usable if pair_view_signature(traj_by_id[a], traj_by_id[b])
        )
        if non_baseline:
            logger.info(
                "pair judge: %d of %d pairs render on a non-baseline label view "
                "and are keyed separately (see view_policy.pair_view_signature)",
                non_baseline,
                len(usable),
            )
        out = PairResolution()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.judge_both_orders, traj_by_id[a], traj_by_id[b]): (a, b)
                for a, b in usable
            }
            for fut in as_completed(futures):
                a, b = futures[fut]
                try:
                    winner, status = fut.result()
                except Exception as exc:  # noqa: BLE001 — defensive
                    logger.warning("pair resolution error for %s vs %s: %s", a, b, exc)
                    out.unresolved.append((a, b))
                    continue
                if status == RESOLVED and winner == a:
                    out.resolved.append((a, b))
                elif status == RESOLVED and winner == b:
                    out.resolved.append((b, a))
                elif status == TIE:
                    out.tied.append((a, b))
                else:
                    out.unresolved.append((a, b))
        if usable:
            logger.info(
                "pair judge: %d resolved, %d TIE (both orders asked, no "
                "agreement), %d unanswered of %d pairs",
                len(out.resolved),
                len(out.tied),
                len(out.unresolved),
                len(usable),
            )
        if self.parse_failures:
            logger.warning(
                "pair judge: %d presentation(s) produced no parseable verdict "
                "after %d attempts each; nothing was cached for them, so a "
                "later run asks again",
                self.parse_failures,
                PARSE_ATTEMPTS,
            )
        return out

    def resolve_pairs(
        self,
        pairs: list[tuple[str, str]],
        traj_by_id: dict[str, Trajectory],
        *,
        max_workers: int = 8,
    ) -> list[tuple[str, str]]:
        """Adjudicate unordered pairs; returns ordered (winner, loser) pairs.

        Ties, order disagreements and failures are dropped — see
        `resolve_pairs_detail` when the caller needs to tell them apart.
        """

        return self.resolve_pairs_detail(pairs, traj_by_id, max_workers=max_workers).resolved
