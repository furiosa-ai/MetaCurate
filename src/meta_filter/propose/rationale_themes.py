"""What the judge values on the pairs the current filter gets wrong, by theme.

The pair judge writes a rationale for every pair it orders, and those
sentences otherwise reach the proposer one at a time: five misordered pairs
per prompt, each with its judge quote, rotating with the pair sample. This is
the aggregate — over ALL the pairs the winner misorders, what does the judge
keep saying? Without it a proposer has no view of which qualities the judge
rewards that no criterion asks about, so it rarely touches `rubric.md` and
keeps editing rules against the pairs in front of it.

This module groups the rationales of the misordered pairs into a fixed
vocabulary of software-engineering evaluation themes (`THEMES`) by keyword
match — deterministic, free, and deliberately crude: a rationale lands in
every theme whose keyword set it hits, counts are per pair, and a theme
nobody mentioned renders nothing. The same keyword match run over the
current rubric's criterion text says whether any criterion already covers
the theme, which is the line the proposer can act on: "root-cause-vs-symptom:
11 misordered pairs; NOT covered by any criterion" is a rubric edit with its
evidence attached, in a form the per-pair quotes never took.

The rationale text itself is never rendered — only theme names, counts and
criterion names — so nothing here needs the pseudonymizer, and only train
pairs are ever read (the caller hands over the misordered pairs of the
accumulated train truth set; the selection set is never consulted).
"""

from __future__ import annotations

import ast
import inspect
import logging
import re
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

# theme -> keyword / phrase set. Lower-case; matched on word boundaries
# (spaces and punctuation inside a phrase are matched literally). Phrases are
# what a judge writes about a trajectory, not rubric jargon, because the
# rationales are the input; the rubric side is matched with the same sets so
# a criterion that speaks the judge's language counts as covering the theme.
THEMES: dict[str, tuple[str, ...]] = {
    "reproduction-before-fix": (
        "reproduc",
        "repro ",
        "reproduce the",
        "reproduced",
        "minimal example",
        "before fixing",
        "confirm the bug",
        "triggered the bug",
    ),
    "test-execution-evidence": (
        "ran the test",
        "runs the test",
        "running the test",
        "ran tests",
        "ran pytest",
        "test run",
        "test output",
        "tests pass",
        "tests passed",
        "test suite",
        "executed the test",
        "actually ran",
        "ran the suite",
    ),
    "wrote-new-tests": (
        "added a test",
        "adds a test",
        "new test",
        "wrote a test",
        "regression test",
        "test case",
        "added tests",
        "test coverage",
    ),
    "root-cause-vs-symptom": (
        "root cause",
        "underlying cause",
        "symptom",
        "band-aid",
        "papers over",
        "actual cause",
        "real cause",
        "masks the",
    ),
    "scope-discipline": (
        "unrelated change",
        "unrelated file",
        "out of scope",
        "scope creep",
        "touches only",
        "minimal change",
        "beyond the",
        "unnecessary change",
        "unrelated edit",
        "focused change",
        "narrow change",
    ),
    "minimal-diff": (
        "minimal diff",
        "small diff",
        "surgical",
        "concise patch",
        "large diff",
        "sprawling",
        "bloated",
        "hundreds of lines",
        "rewrote",
    ),
    "regression-risk": (
        "regression",
        "breaks existing",
        "break other",
        "side effect",
        "backward compat",
        "existing behavior",
        "existing behaviour",
        "could break",
    ),
    "claimed-vs-observed": (
        "claims",
        "claimed",
        "asserts that",
        "without evidence",
        "unverified",
        "did not verify",
        "never verified",
        "no evidence",
        "states that it",
        "says it",
        "assumes",
    ),
    "reading-the-issue": (
        "problem statement",
        "issue description",
        "read the issue",
        "misread",
        "misunderst",
        "the task asks",
        "what was asked",
        "requirements",
        "understood the",
    ),
    "codebase-exploration": (
        "explored",
        "exploration",
        "searched the",
        "grep",
        "located the",
        "found the relevant",
        "navigat",
        "call site",
        "callers",
        "read the code",
        "inspected",
    ),
    "env-setup-churn": (
        "install",
        "dependency",
        "dependencies",
        "environment",
        "setup",
        "set up",
        "pip ",
        "virtualenv",
        "conda",
        "build fail",
    ),
    "repetition-thrash": (
        "repeated",
        "repeating",
        "loop",
        "thrash",
        "same command",
        "over and over",
        "trial and error",
        "flailing",
        "back and forth",
    ),
    "error-recovery": (
        "recover",
        "after the error",
        "traceback",
        "handled the failure",
        "adapted",
        "corrected course",
        "fixed the mistake",
        "retried",
    ),
    "edit-justification": (
        "explains why",
        "explained the change",
        "justif",
        "rationale for",
        "reasoning behind",
        "no explanation",
        "without explaining",
        "commented",
    ),
    "task-completion": (
        "incomplete",
        "did not finish",
        "unfinished",
        "gave up",
        "abandoned",
        "completed the task",
        "solved the",
        "resolves the",
        "fixes the issue",
        "addresses the issue",
        "never submitted",
    ),
    "correctness-of-fix": (
        "incorrect fix",
        "wrong fix",
        "correct fix",
        "correctly fixes",
        "does not fix",
        "doesn't fix",
        "buggy",
        "introduces a bug",
        "logic error",
        "off-by-one",
    ),
    "edge-cases": (
        "edge case",
        "corner case",
        "boundary",
        "empty input",
        "none case",
        "null",
        "special case",
    ),
    "code-quality": (
        "readab",
        "idiomatic",
        "clean code",
        "style",
        "naming",
        "duplicat",
        "hacky",
        "dead code",
        "maintainab",
    ),
    "verification-completeness": (
        "verif",
        "validated",
        "validation",
        "double-check",
        "sanity check",
        "confirmed that",
        "checked that",
    ),
    "hallucination-fabrication": (
        "hallucinat",
        "fabricat",
        "invented",
        "made up",
        "nonexistent",
        "non-existent",
        "does not exist",
        "imaginary",
    ),
    "tool-use-discipline": (
        "tool call",
        "tool use",
        "malformed",
        "wrong tool",
        "misused",
        "command failed",
        "syntax error",
        "invalid command",
    ),
    "final-answer-quality": (
        "summary",
        "final message",
        "final answer",
        "reported",
        "communicat",
        "explanation to",
        "described what",
    ),
    "gold-patch-alignment": (
        "gold patch",
        "reference patch",
        "reference fix",
        "same approach as",
        "matches the reference",
        "differs from the reference",
        "upstream fix",
    ),
    "efficiency": (
        "efficient",
        "wasted",
        "too many steps",
        "fewer steps",
        "quickly",
        "took longer",
        "excessive",
        "redundant steps",
    ),
    "documentation-updates": (
        "docstring",
        "documentation",
        "changelog",
        "readme",
        "comment explaining",
    ),
}

# A theme block renders only when the winner misorders at least this many
# pairs: below it the counts are single digits and read as anecdotes.
MIN_MISORDERED_PAIRS = 5
THEMES_BLOCK_BUDGET = 900


def _compile(keywords: Iterable[str]) -> re.Pattern[str]:
    parts = []
    for keyword in keywords:
        key = keyword.lower()
        trailing_space = key.endswith(" ")
        body = re.escape(key.strip())
        # A keyword ending in a space (`pip `, `repro `) is a whole word; one
        # ending in a letter may be a stem (`reproduc`, `verif`).
        parts.append(rf"\b{body}" + (r"\b" if trailing_space else ""))
    return re.compile("|".join(parts), re.IGNORECASE)


_PATTERNS: dict[str, re.Pattern[str]] = {name: _compile(keys) for name, keys in THEMES.items()}


def themes_of(text: str) -> set[str]:
    """Every theme whose keyword set `text` hits. Empty for empty text."""

    if not text:
        return set()
    return {name for name, pattern in _PATTERNS.items() if pattern.search(text)}


def theme_counts(rationales_by_pair: Mapping[tuple[str, str], Sequence[str]]) -> dict[str, int]:
    """Misordered pairs per theme.

    `rationales_by_pair` maps each misordered pair to the judge's rationales
    for it — both presentation orders when both are cached. A pair counts
    once per theme however many of its rationales mention it.
    """

    counts: dict[str, int] = {}
    for rationales in rationales_by_pair.values():
        hit: set[str] = set()
        for text in rationales:
            hit |= themes_of(text or "")
        for name in hit:
            counts[name] = counts.get(name, 0) + 1
    return counts


def criterion_coverage(rubric_md: str, criteria: Sequence[str]) -> dict[str, list[str]]:
    """theme -> the criteria whose own section hits the theme's keywords.

    Uses the scorer's section splitter so "the criterion's text" is the same
    text the scorer is shown for it; the preamble is not attributed to any
    criterion.
    """

    from meta_filter.core.llm_scorer import _split_rubric_sections

    criteria = list(criteria)
    if not criteria:
        return {}
    sections = _split_rubric_sections(rubric_md, criteria)
    coverage: dict[str, list[str]] = {}
    for criterion in criteria:
        for theme in themes_of(sections.get(criterion) or ""):
            coverage.setdefault(theme, []).append(criterion)
    return coverage


def rule_coverage(rule_functions: Sequence[Callable]) -> dict[str, list[str]]:
    """theme -> the rules whose prose hits the theme's keywords.

    A rule catches a theme as surely as a criterion does, and until this
    existed the readout could only ever say "no CRITERION asks about it" —
    so a theme a rule already handled read as uncovered, and a theme best
    caught deterministically had no way of being named as rule work.

    "The rule's prose" is deliberately narrow: its name with the underscores
    opened out, its docstring, and the string literals in its body — which is
    where a rule says, in the judge's own vocabulary, what it found ("no test
    command appears in the transcript"). Matching the whole source would drag
    in identifiers and marker tuples and hit on everything.
    """

    coverage: dict[str, list[str]] = {}
    for rule in rule_functions or []:
        name = getattr(rule, "__name__", "")
        parts = [name.removeprefix("rule_").replace("_", " "), inspect.getdoc(rule) or ""]
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(rule)))
        except (OSError, TypeError, SyntaxError):
            tree = None  # a rule built at runtime has no readable source
        if tree is not None:
            parts.extend(
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            )
        for theme in themes_of(" \n".join(parts)):
            coverage.setdefault(theme, []).append(name)
    return coverage


def rationale_themes(
    rationales_by_pair: Mapping[tuple[str, str], Sequence[str]],
    rubric_md: str,
    criteria: Sequence[str],
    rule_functions: Sequence[Callable] = (),
) -> dict:
    """The full readout: per-theme misordered-pair counts and coverage.

    JSON-safe: ``{"n_pairs", "n_with_rationale", "themes": [{"theme",
    "pairs", "covered_by": [...]}, ...]}`` sorted by count, highest first,
    uncovered themes ahead of covered ones at equal count.

    `covered_by` names criteria and rules together, because what the caller
    needs is whether ANYTHING in the filter asks about the theme — which
    mechanism should catch it is the proposer's call, not this module's.
    """

    counts = theme_counts(rationales_by_pair)
    coverage = criterion_coverage(rubric_md, criteria)
    for theme, names in rule_coverage(rule_functions).items():
        coverage.setdefault(theme, []).extend(names)
    with_text = sum(1 for texts in rationales_by_pair.values() if any(t for t in texts))
    themes = [
        {"theme": name, "pairs": n, "covered_by": coverage.get(name, [])}
        for name, n in counts.items()
    ]
    themes.sort(key=lambda t: (-t["pairs"], bool(t["covered_by"]), t["theme"]))
    return {"n_pairs": len(rationales_by_pair), "n_with_rationale": with_text, "themes": themes}


def render_rationale_themes(
    report: Mapping | None,
    *,
    min_pairs: int = MIN_MISORDERED_PAIRS,
    budget: int = THEMES_BLOCK_BUDGET,
) -> str:
    """The prompt block, or "" when there is too little to say.

    One line per theme — "theme: N misordered pairs; covered by: X, Y" or
    "... NOT covered by any criterion" — highest count first, cut at
    `budget` characters with a "+N more themes" line. Uncovered themes are
    the actionable ones and are listed first among equals.
    """

    if not report or int(report.get("n_pairs", 0)) < min_pairs or not report.get("themes"):
        return ""
    head = (
        f"judge-rationale themes over the {report['n_pairs']} judged train pairs the last "
        f"winner orders WRONGLY ({report['n_with_rationale']} with a rationale; keyword "
        "grouping, counts are pairs — a theme NOT covered by any criterion is a quality the "
        "judge rewards that the rubric never asks about):"
    )
    lines = [head]
    used = len(head) + 1
    themes = list(report["themes"])
    for i, entry in enumerate(themes):
        covered = entry.get("covered_by") or []
        tail = f"covered by: {', '.join(covered)}" if covered else "NOT covered by any criterion"
        line = f"  * {entry['theme']}: {entry['pairs']} misordered pairs; {tail}"
        remaining = len(themes) - i
        more = f"  ... +{remaining} more themes" if remaining > 1 else ""
        if used + len(line) + 1 + (len(more) + 1 if more else 0) > budget:
            lines.append(more or f"  ... +{remaining} more themes")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
