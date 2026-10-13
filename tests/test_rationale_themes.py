"""`rationale_themes`: the judge's rationales on misordered pairs, by theme.

Synthetic rationales, written the way the pair judge writes them, so each
theme's keyword set is exercised on prose rather than on its own keywords.
"""

from __future__ import annotations

from meta_filter.propose.rationale_themes import (
    MIN_MISORDERED_PAIRS,
    THEMES,
    THEMES_BLOCK_BUDGET,
    criterion_coverage,
    rationale_themes,
    render_rationale_themes,
    theme_counts,
    themes_of,
)

RUBRIC = """# Rubric

## Criterion: Task Grounding

Does the agent read the problem statement before editing?

## Criterion: Verification Completeness

Did the agent run the test suite after its change and read the test output?

## Criterion: Error Recovery

After a traceback, does the agent recover and correct course?
"""
CRITERIA = ["Task Grounding", "Verification Completeness", "Error Recovery"]

RATIONALES = {
    ("a1", "b1"): [
        "A reproduced the bug with a minimal example before fixing it; B edited "
        "the file straight away and never ran the test suite.",
        "B claims the fix works without evidence; A ran pytest and read the output.",
    ],
    ("a2", "b2"): [
        "A addressed the root cause in the parser; B papers over the symptom with a special case."
    ],
    ("a3", "b3"): [
        "B's patch touches unrelated files and rewrote a helper it did not need to; "
        "A kept a minimal diff."
    ],
    ("a4", "b4"): ["A fixed the root cause. B's change risks a regression in existing behavior."],
    ("a5", "b5"): ["B misread the problem statement and solved a different task."],
    ("a6", "b6"): ["A reproduced the failure first, then fixed the root cause."],
    ("a7", "b7"): [""],
}


def test_the_vocabulary_is_about_twenty_five_themes_with_keyword_sets():
    assert 20 <= len(THEMES) <= 30
    assert all(keys for keys in THEMES.values())


def test_a_rationale_lands_in_every_theme_it_mentions():
    hit = themes_of(RATIONALES[("a1", "b1")][0])
    assert {"reproduction-before-fix", "test-execution-evidence"} <= hit
    assert "root-cause-vs-symptom" not in hit
    assert themes_of("") == set()


def test_stems_match_inflections_but_whole_word_keywords_do_not_match_inside_words():
    assert "reproduction-before-fix" in themes_of("the agent reproduces it")
    assert "env-setup-churn" in themes_of("ran pip install twice")
    # `pip ` is a whole word: "pipeline" is not an install.
    assert "env-setup-churn" not in themes_of("the data pipeline was rewritten")


def test_counts_are_per_pair_not_per_rationale():
    counts = theme_counts(RATIONALES)
    # Pair a1/b1 mentions test execution in both presentations: counted once.
    assert counts["test-execution-evidence"] == 1
    assert counts["root-cause-vs-symptom"] == 3
    assert counts["reproduction-before-fix"] == 2
    assert counts["reading-the-issue"] == 1


def test_criterion_coverage_uses_the_criterions_own_section():
    coverage = criterion_coverage(RUBRIC, CRITERIA)
    assert coverage["test-execution-evidence"] == ["Verification Completeness"]
    assert coverage["reading-the-issue"] == ["Task Grounding"]
    assert coverage["error-recovery"] == ["Error Recovery"]
    assert "root-cause-vs-symptom" not in coverage
    assert criterion_coverage(RUBRIC, []) == {}


def test_the_report_orders_by_count_with_uncovered_themes_first_among_equals():
    report = rationale_themes(RATIONALES, RUBRIC, CRITERIA)
    assert report["n_pairs"] == 7
    assert report["n_with_rationale"] == 6
    first = report["themes"][0]
    assert first["theme"] == "root-cause-vs-symptom"
    assert first["pairs"] == 3
    assert first["covered_by"] == []
    by_name = {t["theme"]: t for t in report["themes"]}
    assert by_name["test-execution-evidence"]["covered_by"] == ["Verification Completeness"]
    # Equal counts: the uncovered theme precedes the covered one.
    ones = [t for t in report["themes"] if t["pairs"] == 1]
    uncovered_positions = [i for i, t in enumerate(ones) if not t["covered_by"]]
    covered_positions = [i for i, t in enumerate(ones) if t["covered_by"]]
    assert max(uncovered_positions) < min(covered_positions)


def test_the_block_names_coverage_per_theme_and_counts_pairs():
    text = render_rationale_themes(rationale_themes(RATIONALES, RUBRIC, CRITERIA))
    assert text.startswith("judge-rationale themes over the 7 judged train pairs")
    assert "  * root-cause-vs-symptom: 3 misordered pairs; NOT covered by any criterion" in text
    assert "  * reading-the-issue: 1 misordered pairs; covered by: Task Grounding" in text
    assert len(text) <= THEMES_BLOCK_BUDGET
    # No rationale text reaches the block.
    assert "papers over" not in text and "pytest" not in text
    # With no cap every theme is listed, coverage named per theme.
    full = render_rationale_themes(rationale_themes(RATIONALES, RUBRIC, CRITERIA), budget=10_000)
    assert (
        "  * test-execution-evidence: 1 misordered pairs; covered by: Verification Completeness"
        in full
    )
    assert "more themes" not in full


def test_below_five_misordered_pairs_nothing_renders():
    few = dict(list(RATIONALES.items())[: MIN_MISORDERED_PAIRS - 1])
    assert render_rationale_themes(rationale_themes(few, RUBRIC, CRITERIA)) == ""
    assert render_rationale_themes(None) == ""
    empty = {(f"a{i}", f"b{i}"): [""] for i in range(6)}
    assert render_rationale_themes(rationale_themes(empty, RUBRIC, CRITERIA)) == ""


def test_the_block_is_capped_and_says_how_many_themes_were_cut():
    many = {
        (f"a{i}", f"b{i}"): [" ".join(" ".join(keys[:2]) for keys in THEMES.values())]
        for i in range(8)
    }
    report = rationale_themes(many, RUBRIC, CRITERIA)
    assert len(report["themes"]) == len(THEMES)
    text = render_rationale_themes(report)
    assert len(text) <= THEMES_BLOCK_BUDGET
    assert "more themes" in text


# --- rules cover themes too -------------------------------------------------
#
# Read off the rubric alone, a theme a RULE already handles would report as
# uncovered: a proposer told its largest theme is covered by nothing, while a
# rule in the filter does exactly that, may well delete the rule.


def _rule(name: str, doc: str, evidence: str):
    src = (
        f"def rule_{name}(trajectory):\n"
        f'    """{doc}"""\n'
        f'    return {{"name": "{name}", "verdict": "REJECT", "evidence": ["{evidence}"]}}\n'
    )
    ns: dict = {}
    exec(compile(src, f"<rule {name}>", "exec"), ns)  # noqa: S102 — fixture source
    return ns[f"rule_{name}"]


def test_a_rule_covers_the_theme_its_prose_names():
    from meta_filter.propose.rationale_themes import rule_coverage

    fn = _rule(
        "no_verification",
        "Substantive patch with no test command.",
        "no test command appears in the transcript",
    )
    coverage = rule_coverage([fn])

    assert "verification-completeness" in coverage
    assert coverage["verification-completeness"] == ["rule_no_verification"]


def test_a_rule_that_names_no_theme_covers_nothing():
    """The wired blank seed's example rule carries no domain opinion."""

    from meta_filter.propose.rationale_themes import rule_coverage

    fn = _rule(
        "no_turns",
        "A trajectory with zero turns is not evidence.",
        "trajectory has 0 turns",
    )

    assert rule_coverage([fn]) == {}


def test_covered_by_names_criteria_and_rules_together():
    from meta_filter.propose.rationale_themes import rationale_themes

    fn = _rule(
        "no_verification",
        "Substantive patch with no test command.",
        "no test command appears in the transcript",
    )
    rubric = "# R\n\n## Criterion: Alpha\n\nDid it reproduce the bug before fixing?\n"
    # A rationale in the theme the rule speaks to, so the theme is in the
    # report at all — only themes the judge actually raised are listed.
    pairs = {("a", "b"): ["the agent never verified the fix; no test was run"]}

    with_rules = rationale_themes(pairs, rubric, ["Alpha"], [fn])
    without = rationale_themes(pairs, rubric, ["Alpha"])

    covered = {t["theme"]: t["covered_by"] for t in with_rules["themes"]}
    bare = {t["theme"]: t["covered_by"] for t in without["themes"]}

    assert "rule_no_verification" in covered["verification-completeness"]
    assert bare["verification-completeness"] == []
    # The rubric side is untouched by the addition.
    assert covered.keys() == bare.keys()


def test_a_rule_with_no_readable_source_is_skipped_not_fatal():
    from meta_filter.propose.rationale_themes import rule_coverage

    fn = eval("lambda trajectory: None")  # noqa: S307 — no source for inspect
    fn.__name__ = "rule_runtime_built"

    assert rule_coverage([fn]) == {}
