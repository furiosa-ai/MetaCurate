"""A rule that fires and moves no score is a wiring fault, not a weak rule.

`Solver` substitutes the Python function name for whatever label a rule
declares, so a policy keyed by the declared label misses and takes its
fallback. `dict.get` does not raise, so nothing in a run says so: the only
signal is the rank-lift line reporting no shift, whose advice is to prune —
which is how a mis-keyed rule gets deleted for being useless. These cover the
readout that separates the two.
"""

from types import SimpleNamespace

from meta_filter.criterion_lift import render_rule_pricing, rule_pricing

ROW = SimpleNamespace(trajectory_id="t0")


def _decision(rank_score, fired, scores=None):
    """`FilterDecision` carries rule NAMES, not the dicts decide() was given."""

    return SimpleNamespace(
        trajectory_id="t0",
        rank_score=rank_score,
        triggered_rules=[f["name"] for f in fired],
        rubric_scores=scores or {"Overall": 1.0},
    )


def _rule(name, verdict="REJECT"):
    """A rule function, named the way `Solver` will label its firing."""

    def fn(trajectory):
        return {"name": "declared_label", "verdict": verdict, "evidence": []}

    fn.__name__ = name
    return fn


def _filter(decide, rules=()):
    return SimpleNamespace(
        policy_module=SimpleNamespace(decide=decide),
        rule_functions=list(rules),
    )


def _mis_keyed_policy(trajectory, rubric_scores, triggered_rules):
    """The bug: the table is keyed by the declared label, not the function."""

    weights = {"no_turns": -15.0}
    score = 50.0
    for rule in triggered_rules or []:
        score += weights.get(rule.get("name"), 0.0)
    return {"score": score}


def _correct_policy(trajectory, rubric_scores, triggered_rules):
    weights = {"rule_no_turns": -15.0}
    score = 50.0
    for rule in triggered_rules or []:
        score += weights.get(rule.get("name"), 0.0)
    return {"score": score}


FIRED = [{"name": "rule_no_turns", "verdict": "REJECT", "evidence": []}]
RULES = (_rule("rule_no_turns"),)


def test_a_mis_keyed_rule_is_reported_as_inert():
    report = rule_pricing(_filter(_mis_keyed_policy, RULES), [_decision(50.0, FIRED)], [ROW])
    assert report["rule_no_turns"]["fires"] == 1
    assert report["rule_no_turns"]["inert"] is True


def test_a_correctly_keyed_rule_is_not_inert():
    report = rule_pricing(_filter(_correct_policy, RULES), [_decision(35.0, FIRED)], [ROW])
    assert report["rule_no_turns"]["inert"] is False
    assert report["rule_no_turns"]["deltas"] == {-15.0: 1}


def test_the_readout_names_the_wiring_and_says_not_to_delete():
    text = render_rule_pricing(
        rule_pricing(_filter(_mis_keyed_policy, RULES), [_decision(50.0, FIRED)], [ROW])
    )
    assert "RULE WIRING BROKEN" in text
    assert "rule_no_turns" in text
    assert "do not delete the rule" in text


def test_a_healthy_filter_reports_prices_without_the_warning():
    text = render_rule_pricing(
        rule_pricing(_filter(_correct_policy, RULES), [_decision(35.0, FIRED)], [ROW])
    )
    assert "RULE WIRING BROKEN" not in text
    assert "rule_no_turns: 1 fires" in text


def test_one_rule_of_several_is_attributed_by_leave_one_out():
    """Two rules fire; only the mis-keyed one reads as inert."""

    def policy(trajectory, rubric_scores, triggered_rules):
        weights = {"rule_live": -9.0}
        return {"score": 50.0 + sum(weights.get(r.get("name"), 0.0) for r in triggered_rules or [])}

    fired = [
        {"name": "rule_live", "verdict": "REJECT", "evidence": []},
        {"name": "rule_dead", "verdict": "REJECT", "evidence": []},
    ]
    report = rule_pricing(
        _filter(policy, (_rule("rule_live"), _rule("rule_dead"))),
        [_decision(41.0, fired)],
        [ROW],
    )
    assert report["rule_live"]["deltas"] == {-9.0: 1}
    assert report["rule_dead"]["inert"] is True


def test_no_rules_and_no_policy_are_both_survivable():
    assert rule_pricing(_filter(_correct_policy, ()), [_decision(50.0, [])], [ROW]) == {}
    assert rule_pricing(SimpleNamespace(policy_module=None), [_decision(50.0, FIRED)], [ROW]) == {}
    assert render_rule_pricing({}) == ""


def test_a_rule_whose_price_is_wrong_but_non_zero_is_not_called_inert():
    """The lookup misses and a NUMERIC fallback applies.

    The rule works, so it is not inert — but it does not do what policy.py
    says it does, and the readout must report the price it actually applied
    rather than accusing it of doing nothing.
    """

    def fallback_policy(trajectory, rubric_scores, triggered_rules):
        weights = {"runaway_workflow": -14.0}  # declared label, never matched
        score = 50.0
        for rule in triggered_rules or []:
            score += weights.get(rule.get("name"), -8.0)
        return {"score": score}

    fired = [{"name": "rule_runaway_workflow", "verdict": "REJECT", "evidence": []}]
    report = rule_pricing(
        _filter(fallback_policy, (_rule("rule_runaway_workflow"),)),
        [_decision(42.0, fired)],
        [ROW],
    )
    assert report["rule_runaway_workflow"]["inert"] is False
    assert report["rule_runaway_workflow"]["deltas"] == {-8.0: 1}
    text = render_rule_pricing(report)
    assert "RULE WIRING BROKEN" not in text
    assert "-8x1" in text
