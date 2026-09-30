"""`--freeze rubric` has to close the rubric channel, not just the file.

`RUBRIC_CRITERIA` lives in policy.py, so freezing rubric.md alone leaves the
channel open: a candidate can put a criterion back, leaving the scorer billing
a criterion rubric.md does not define.
"""

from pathlib import Path

from meta_filter.constraints import frozen_changed

POLICY = """
RUBRIC_CRITERIA = {criteria!r}


def decide(trajectory, rubric_scores, triggered_rules):
    return {{"decision": "ACCEPT", "score": 50.0, "confidence": 0.5,
             "reasoning_summary": "", "evidence": []}}
"""


def _filter(root: Path, name: str, criteria: list[str], rubric: str = "# r\n") -> Path:
    d = root / name
    d.mkdir()
    (d / "rubric.md").write_text(rubric, encoding="utf-8")
    (d / "rules.py").write_text("", encoding="utf-8")
    (d / "policy.py").write_text(POLICY.format(criteria=criteria), encoding="utf-8")
    return d


def test_adding_a_criterion_through_policy_is_caught_when_the_rubric_is_frozen(tmp_path):
    base = _filter(tmp_path, "base", [])
    cand = _filter(tmp_path, "cand", ["Overall Training Value"])
    assert frozen_changed(cand, base, ["rubric"]) == ["policy.py:RUBRIC_CRITERIA"]


def test_an_unchanged_declaration_passes(tmp_path):
    base = _filter(tmp_path, "base", ["A"])
    cand = _filter(tmp_path, "cand", ["A"])
    assert frozen_changed(cand, base, ["rubric"]) == []


def test_the_pin_does_not_apply_when_the_rubric_is_not_frozen(tmp_path):
    base = _filter(tmp_path, "base", [])
    cand = _filter(tmp_path, "cand", ["A"])
    assert frozen_changed(cand, base, ["rules"]) == []


def test_policy_already_frozen_needs_no_extra_pin(tmp_path):
    """Both frozen: policy.py's own byte check already covers the list."""

    base = _filter(tmp_path, "base", [])
    cand = _filter(tmp_path, "cand", ["A"])
    assert frozen_changed(cand, base, ["rubric", "policy"]) == ["policy.py"]


def test_a_changed_rubric_file_is_still_reported(tmp_path):
    base = _filter(tmp_path, "base", ["A"], rubric="# one\n")
    cand = _filter(tmp_path, "cand", ["A"], rubric="# two\n")
    assert frozen_changed(cand, base, ["rubric"]) == ["rubric.md"]


def test_an_unloadable_policy_is_left_to_the_checks_that_own_it(tmp_path):
    """Not silence: the loader and `rubric_criteria_defect` refuse it already,
    and a second reason naming RUBRIC_CRITERIA would misdescribe the fault."""

    base = _filter(tmp_path, "base", [])
    cand = tmp_path / "broken"
    cand.mkdir()
    (cand / "rubric.md").write_text("# r\n", encoding="utf-8")
    (cand / "rules.py").write_text("", encoding="utf-8")
    (cand / "policy.py").write_text("this is not python(", encoding="utf-8")
    assert frozen_changed(cand, base, ["rubric"]) == []
