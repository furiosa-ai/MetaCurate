"""Deterministic mock LLM client.

The mock is *not* a stub — it emits real proposer JSON that meaningfully edits
`rubric.md`, `rules.py`, and `policy.py` in response to the FN/FP counts, so
the optimizer produces observable diffs across iterations without any API key.

The mock is intentionally aware only of aggregate FN/FP counts and the raw
sources; it does NOT contain any knowledge about "promotion" or "rule
promotion". Emergent structural changes come from the proposer's incentive
mix (loss weights against FN/FP/cost/complexity/review), never from mock
instructions that say "convert rubric to rules".
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from meta_filter.llm.base import LLMResponse


@dataclass
class MockLLMClient:
    """Deterministic response generator."""

    model: str = "mock-1"
    seed: int = 17
    _call_index: int = field(default=0, init=False)

    def _tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> LLMResponse:
        _ = cache_system  # mock ignores caching hints
        self._call_index += 1
        text = self._respond(prompt=prompt, system=system or "")
        return LLMResponse(
            text=text,
            prompt_tokens=self._tokens(prompt) + self._tokens(system or ""),
            completion_tokens=self._tokens(text),
            model=self.model,
            calls=1,
            # A mock call really is free, and saying so with a source keeps an
            # offline run's zero total distinguishable from an unpriced one.
            cost_usd=0.0,
            cost_source="mock",
        )

    # ----- response dispatch -----

    def _respond(self, prompt: str, system: str) -> str:
        # The proposer prompt carries a small marker so we can dispatch reliably.
        if "META_FILTER::PROPOSER" in prompt:
            return self._respond_proposer(prompt)
        if "META_FILTER::SOLVER" in prompt:
            return self._respond_solver(prompt)
        if "=== TRAJECTORY A (" in prompt and "=== TRAJECTORY B (" in prompt:
            return self._respond_pair_judge(prompt)
        if "### RUBRIC (verbatim)" in prompt and _SCORER_CRITERIA_LINE in prompt:
            return self._respond_rubric_scorer(prompt)
        return json.dumps({"note": "mock echo", "prompt_len": len(prompt)})

    # ----- per-criterion rubric scorer -----

    def _respond_rubric_scorer(self, prompt: str) -> str:
        """A readable score for each criterion the prompt asks for.

        Deterministic in (task, criterion) and nothing else — not the seed,
        not the call index — so a cell scored by the proposer's scoring tool
        and the same cell scored by the full pass agree, which is the cache
        property `meta_filter.score_tool` exists to exercise offline. The
        value is spread over the whole -2..+2 scale so a mock ranking run's
        scores are not degenerate.
        """

        task = ""
        if "Task:\n" in prompt:
            task = prompt.split("Task:\n", 1)[1].split("\n\nMetadata:", 1)[0]
        line = next(line for line in prompt.splitlines() if line.startswith(_SCORER_CRITERIA_LINE))
        criteria = [c.strip() for c in line[len(_SCORER_CRITERIA_LINE) :].split(",") if c.strip()]
        scores = {
            criterion: {
                "score": _hash_int("mock-scorer", task, criterion) % 5 - 2,
                "evidence": f"mock evidence for {criterion}",
            }
            for criterion in criteria
        }
        return json.dumps({"scores": scores, "summary": "mock rubric scoring"})

    # ----- proposer -----

    def _respond_proposer(self, prompt: str) -> str:
        state = _parse_proposer_state(prompt)
        candidates = _generate_candidates(state, seed=self.seed + self._call_index)
        return json.dumps({"candidates": candidates}, indent=2)

    # ----- pair judge -----

    def _respond_pair_judge(self, prompt: str) -> str:
        """A verdict that depends only on the two ids, whichever is shown as A.

        The pair judge asks every pair in both presentation orders and keeps
        only agreements, so a mock that echoed `{"note": ...}` left every pair
        UNRESOLVED and a mock ranking run had no truth pairs and no selection
        set at all. Hashing the *sorted* ids makes the two orders agree, one
        pair in ten is a TIE so the exclusion path is exercised too, and the
        seed is deliberately absent: the verdict has to be the same across
        the two calls of one pair and across a resumed run.
        """

        id_a = prompt.split("=== TRAJECTORY A (", 1)[1].split(")", 1)[0]
        id_b = prompt.split("=== TRAJECTORY B (", 1)[1].split(")", 1)[0]
        lo, hi = sorted((id_a, id_b))
        bucket = _hash_int("pair-judge", lo, hi) % 10
        if bucket == 0:
            letter = "TIE"
        else:
            preferred = lo if bucket % 2 else hi
            letter = "A" if preferred == id_a else "B"
        # A rationale in the judge's register rather than a placeholder, so
        # the rationale-theme block (`rationale_themes`) has something to
        # group in a mock run. Chosen by the same sorted-id hash as the
        # verdict, so both presentation orders say the same thing.
        rationale = _MOCK_RATIONALES[_hash_int("pair-rationale", lo, hi) % len(_MOCK_RATIONALES)]
        return json.dumps({"winner": letter, "rationale": rationale})

    # ----- solver (unused for the mock; solver has its own deterministic path) -----

    def _respond_solver(self, prompt: str) -> str:
        return json.dumps(
            {
                "decision": "REVIEW",
                "confidence": 0.5,
                "rubric_scores": {},
                "triggered_rules": [],
                "evidence": [],
                "reasoning_summary": "mock solver fallback",
            }
        )


# ---------------------------------------------------------------------------
# Helpers below drive the mock proposer. They are intentionally not exposed as
# public API — treat them as inline heuristics used to make the offline demo
# meaningful, not as part of the algorithm being studied.
# ---------------------------------------------------------------------------


# The line `llm_scorer.SCORER_PROMPT` lists the criteria to score on.
_SCORER_CRITERIA_LINE = "Criteria to score (in order): "


def _parse_proposer_state(prompt: str) -> dict:
    """Extract the embedded JSON state block from the proposer prompt."""

    match = re.search(r"<STATE_JSON>(.*?)</STATE_JSON>", prompt, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


# What a mock pair judge says about the pair it just ordered: one sentence per
# common failure pattern, in the vocabulary `rationale_themes.THEMES` groups.
_MOCK_RATIONALES: tuple[str, ...] = (
    "The preferred trajectory reproduced the bug before fixing it and ran the "
    "test suite afterwards; the other never ran a test.",
    "The preferred trajectory fixed the root cause; the other papers over the "
    "symptom with a special case.",
    "The other trajectory touches unrelated files and rewrote a helper it did "
    "not need to; the preferred one kept a minimal diff.",
    "The other trajectory claims the fix works without evidence; the preferred "
    "one read the test output.",
    "The other trajectory misread the problem statement and solved a different task.",
    "The other trajectory repeated the same command over and over after a "
    "traceback; the preferred one recovered and corrected course.",
    "The other trajectory's change risks a regression in existing behavior.",
)


def _hash_int(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _rewrite_rubric_shrink(rubric: str) -> str:
    """Drop the last criterion block to trade rubric weight for deterministic rules."""

    lines = rubric.splitlines()
    keep: list[str] = []
    dropped = False
    for line in reversed(lines):
        if not dropped and (
            line.strip().lower().startswith("criterion:") or line.strip().startswith("## Criterion")
        ):
            dropped = True
            continue
        if dropped:
            keep.append(line)
    if not dropped:
        return rubric
    return "\n".join(reversed(keep)).rstrip() + "\n"


def _append_rule(rules_src: str, rule_name: str, rule_body: str) -> str:
    if f"def {rule_name}(" in rules_src:
        return rules_src
    return rules_src.rstrip() + "\n\n\n" + rule_body.rstrip() + "\n"


def _remove_rule(rules_src: str, rule_name: str) -> str:
    """Delete a top-level `def rule_<name>` function if present."""

    pattern = re.compile(
        rf"^def {re.escape(rule_name)}\(.*?(?=^def |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    return pattern.sub("", rules_src)


LOOP_RULE = '''\
def rule_repeated_action_loop(trajectory) -> dict | None:
    """Reject trajectories where the same action fires 4+ times back-to-back."""

    counts: dict[str, int] = {}
    last_action: str | None = None
    streak = 0
    max_streak = 0
    culprit: str | None = None
    for turn in trajectory.turns:
        action = getattr(turn, "action", None) or turn.role
        if action == last_action:
            streak += 1
        else:
            streak = 1
            last_action = action
        if streak > max_streak:
            max_streak = streak
            culprit = action
        counts[action] = counts.get(action, 0) + 1
    if max_streak >= 4:
        return {
            "name": "repeated_action_loop",
            "verdict": "REJECT",
            "evidence": [f"action {culprit!r} repeated {max_streak}x consecutively"],
        }
    return None
'''

EDIT_TEST_THRASH_RULE = '''\
def rule_edit_test_thrash(trajectory) -> dict | None:
    """Reject trajectories that alternate edit/test 6+ times without progress."""

    toggles = 0
    prev: str | None = None
    for turn in trajectory.turns:
        action = getattr(turn, "action", None)
        if action in {"edit", "test"} and prev in {"edit", "test"} and action != prev:
            toggles += 1
        prev = action or prev
    if toggles >= 6:
        return {
            "name": "edit_test_thrash",
            "verdict": "REJECT",
            "evidence": [f"edit/test toggled {toggles} times"],
        }
    return None
'''

VERIFICATION_RULE = '''\
def rule_missing_verification(trajectory) -> dict | None:
    """Flag trajectories that claim completion but never ran a test/lint/build."""

    saw_verification = False
    claimed_done = False
    for turn in trajectory.turns:
        action = getattr(turn, "action", None) or ""
        content = (turn.content or "").lower()
        if action in {"test", "lint", "build", "run_tests"}:
            saw_verification = True
        if "task complete" in content or "all done" in content or "finished" in content:
            claimed_done = True
    if claimed_done and not saw_verification:
        return {
            "name": "missing_verification",
            "verdict": "REVIEW",
            "evidence": ["completion claimed without verification step"],
        }
    return None
'''


NEW_RUBRIC_CRITERION_DIFF_QUALITY = """
## Criterion: Diff Quality

Does the final submitted patch look plausible for the stated task?

+2: Small, targeted hunks that touch the file/symbol the task described.
+1: Reasonable diff with some unrelated formatting churn.
 0: Diff exists but its scope is unclear.
-1: Diff spans many unrelated files.
-2: Diff is empty, trivial (<200 bytes), or contradicts the task.
"""


def _replace_anchor_verification(rubric: str) -> str:
    """Rewrite the Verification Completeness anchors to add more test-command
    keywords (`pytest`, `python -m pytest`, project's build/lint)."""

    marker = "## Criterion: Verification Completeness"
    if marker not in rubric:
        return rubric
    # Find the block and replace it with a strengthened version.
    lines = rubric.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == marker:
            out.append(lines[i])
            i += 1
            # skip until next criterion or EOF
            while i < len(lines) and not lines[i].strip().startswith("## Criterion:"):
                i += 1
            out.append("")
            out.append("Does the trajectory actually verify that the change works?")
            out.append("")
            out.append("+2: Runs `pytest`, `python -m pytest`, `python -m unittest`, or")
            out.append("    the project's declared test runner AFTER the final edit")
            out.append("    and observes a passing exit code.")
            out.append("+1: Runs partial verification (single test file, smoke test,")
            out.append("    linter such as `ruff`/`flake8`) after the final edit.")
            out.append(" 0: Verification exists but not clearly after the final edit.")
            out.append("-1: Verification ran early but not after the final edit,")
            out.append("    OR only `python -c` inline snippets were used.")
            out.append("-2: Completion declared with no test/lint/build invocation at any")
            out.append("    point in the trajectory.")
            out.append("")
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).rstrip() + "\n"


def _append_criterion(rubric: str, new_block: str) -> str:
    if new_block.strip().splitlines()[0] in rubric:
        return rubric
    return rubric.rstrip() + "\n\n" + new_block.strip() + "\n"


def _tweak_policy_thresholds(policy_src: str, direction: str) -> tuple[str, str]:
    """Return (new_src, note). direction ∈ {'less_review', 'more_review',
    'stricter_reject', 'looser_reject'}. If nothing matches, return unchanged."""

    def _replace_number(pattern: str, new_value: str) -> str | None:
        m = re.search(pattern, policy_src)
        if not m:
            return None
        cur = m.group(1)
        try:
            if float(cur) == float(new_value):
                return None
        except ValueError:
            return None
        return re.sub(pattern, lambda mm: mm.group(0).replace(cur, new_value), policy_src)

    if direction == "less_review":
        # Lower REVIEW_LOWER makes more decisions confident.
        new_src = _replace_number(r"REVIEW_LOWER\s*=\s*([\d.]+)", "0.20")
        note = "policy: lower REVIEW_LOWER band to reduce hedging."
    elif direction == "more_review":
        new_src = _replace_number(r"REVIEW_LOWER\s*=\s*([\d.]+)", "0.35")
        note = "policy: raise REVIEW_LOWER to push borderline decisions into REVIEW."
    elif direction == "stricter_reject":
        new_src = _replace_number(r"REJECT_MARGIN\s*=\s*(\d+)", "1")
        note = "policy: tighten REJECT_MARGIN to reject more aggressively."
    elif direction == "looser_reject":
        new_src = _replace_number(r"REJECT_MARGIN\s*=\s*(\d+)", "3")
        note = "policy: relax REJECT_MARGIN so borderline rubric doesn't reject."
    else:
        new_src = None
        note = ""
    if new_src is None:
        return policy_src, ""
    return new_src, note


def _generate_candidates(state: dict, seed: int) -> list[dict]:
    """Deterministically produce N candidates using an edit catalog.

    Each iteration picks up to N distinct edit ops from the catalog based on
    a seeded hash and the current FN/FP pressure. The catalog covers every
    filter surface — rubric anchor rewrite, criterion add, rule add/remove,
    and policy threshold tweaks — so we observe genuine diversity across
    iterations without needing a real LLM.
    """

    n = int(state.get("n_candidates", 3))
    rubric = str(state.get("rubric_md", ""))
    rules_src = str(state.get("rules_py", ""))
    policy_src = str(state.get("policy_py", ""))
    fn_count = int(state.get("fn_count", 0))
    fp_count = int(state.get("fp_count", 0))
    iteration = int(state.get("iteration", 0))

    # Each entry is (applicable_bool, note, transform_fn) where transform_fn
    # returns (rubric', rules', policy'). Applicability filters obvious no-ops.
    catalog: list[tuple[bool, str, callable]] = []

    catalog.append(
        (
            "rule_repeated_action_loop" not in rules_src,
            "rules: add rule_repeated_action_loop for streaks of identical actions",
            lambda: (
                rubric,
                _append_rule(rules_src, "rule_repeated_action_loop", LOOP_RULE),
                policy_src,
            ),
        )
    )
    catalog.append(
        (
            "rule_missing_verification" not in rules_src,
            "rules: add rule_missing_verification (REVIEW verdict)",
            lambda: (
                rubric,
                _append_rule(rules_src, "rule_missing_verification", VERIFICATION_RULE),
                policy_src,
            ),
        )
    )
    catalog.append(
        (
            "rule_edit_test_thrash" not in rules_src,
            "rules: add rule_edit_test_thrash for repeated edit/test toggles",
            lambda: (
                rubric,
                _append_rule(rules_src, "rule_edit_test_thrash", EDIT_TEST_THRASH_RULE),
                policy_src,
            ),
        )
    )
    catalog.append(
        (
            "rule_trivially_short" in rules_src and fp_count > 0,
            "rules: remove rule_trivially_short (was over-rejecting brief valid tasks)",
            lambda: (
                rubric,
                _remove_rule(rules_src, "rule_trivially_short"),
                policy_src,
            ),
        )
    )
    catalog.append(
        (
            "## Criterion: Verification Completeness" in rubric,
            "rubric: rewrite Verification anchors to include pytest / python -m pytest",
            lambda: (
                _replace_anchor_verification(rubric),
                rules_src,
                policy_src,
            ),
        )
    )
    catalog.append(
        (
            "## Criterion: Diff Quality" not in rubric,
            "rubric: add a new Diff Quality criterion",
            lambda: (
                _append_criterion(rubric, NEW_RUBRIC_CRITERION_DIFF_QUALITY),
                rules_src,
                policy_src,
            ),
        )
    )
    catalog.append(
        (
            rubric.count("## Criterion:") > 3,
            "rubric: drop the trailing criterion to reduce dimensionality",
            lambda: (
                _rewrite_rubric_shrink(rubric),
                rules_src,
                policy_src,
            ),
        )
    )
    for direction in ("less_review", "more_review", "stricter_reject", "looser_reject"):
        new_policy, note = _tweak_policy_thresholds(policy_src, direction)
        catalog.append(
            (
                bool(note),
                note or f"policy: {direction}",
                lambda new_policy=new_policy: (rubric, rules_src, new_policy),
            )
        )

    applicable = [(note, fn) for ok, note, fn in catalog if ok]
    if not applicable:
        # Everything already applied — emit a single no-op so the pool is valid.
        applicable = [("no-op: nothing left to change", lambda: (rubric, rules_src, policy_src))]

    # Pick N distinct ops deterministically, biased by current pressure.
    def _op_priority(idx: int, note: str) -> int:
        boost = 0
        if fp_count > fn_count and "add rule_" in note:
            boost += 5  # adding new rules risks more FP
        if fn_count > fp_count and "remove" in note:
            boost += 5
        if fp_count == 0 and fn_count == 0 and "rubric:" in note:
            boost -= 2
        return (_hash_int(seed, iteration, idx) & 0xFFFF) + boost

    ordered = sorted(range(len(applicable)), key=lambda i: _op_priority(i, applicable[i][0]))
    picks = ordered[:n]

    candidates: list[dict] = []
    for order_idx, cat_idx in enumerate(picks):
        note, fn = applicable[cat_idx]
        rub2, rules2, policy2 = fn()
        candidates.append(
            {
                "name": f"cand_iter{iteration}_{chr(ord('a') + order_idx)}",
                "notes": note,
                "changes": {"rubric.md": rub2, "rules.py": rules2, "policy.py": policy2},
            }
        )

    # Pad if catalog gave fewer than N (rare — only when everything is applied).
    while len(candidates) < n:
        candidates.append(
            {
                "name": f"cand_iter{iteration}_x{len(candidates)}",
                "notes": "no-op filler",
                "changes": {"rubric.md": rubric, "rules.py": rules_src, "policy.py": policy_src},
            }
        )

    return candidates
