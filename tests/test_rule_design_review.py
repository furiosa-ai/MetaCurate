"""The design review that closes the delete-until-perfect loophole.

The deterministic `rule_quality` is a mean over *firing* rules, so an
optimizer can reach 1.0 by deleting a 95%-precision rule and an 81%-precision
rule — both drag the mean down — until a single rule remains. These tests
pin the arithmetic that makes that strategy stop paying, and the evidence the
model needs in order to notice it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from meta_filter.core.schemas import Decision, Trajectory
from meta_filter.critics import rule_design
from meta_filter.critics.rule_critic import RuleCritic


def test_deleting_every_imperfect_rule_stops_paying():
    """The delete-until-perfect collapse, priced through both halves of the fix.

    The deterministic figures below are what `_aggregate_quality` now returns
    for those two filters — 0.558 for the one-firing-rule set covering 82 of
    200, 0.685 for the three-rule set covering ~140 — not the 1.0 and 0.6358
    the old plain mean produced. The design review then reinforces the same
    ordering rather than being the only thing holding it up.
    """

    lonely_but_perfect = rule_design.combine(
        0.558,
        rule_design.DesignReview(
            scores={
                "coverage": -2,
                "factoring": -2,
                "precision_discipline": 0,
                "division_of_labor": 0,
                "robustness": 0,
                "policy_coherence": 0,
            }
        ).quality,
    )
    messy_but_covering = rule_design.combine(
        0.685,
        rule_design.DesignReview(
            scores={
                "coverage": 2,
                "factoring": 1,
                "precision_discipline": 1,
                "division_of_labor": 1,
                "robustness": 0,
                "policy_coherence": 0,
            }
        ).quality,
    )

    assert lonely_but_perfect < messy_but_covering


def test_the_design_term_can_never_zero_out_a_filter():
    """A critic disliking the factoring is not grounds for scoring a working
    filter at nothing, so the multiplier floors at half."""

    worst = rule_design.DesignReview(scores=dict.fromkeys(rule_design.DIMENSIONS, -2))
    assert worst.quality == 0.0
    assert rule_design.combine(1.0, worst.quality) == rule_design.DESIGN_FLOOR


def test_an_absent_review_leaves_the_measured_score_untouched():
    assert rule_design.combine(0.6358, None) == 0.6358


def test_a_neutral_review_sits_at_the_midpoint():
    assert rule_design.DesignReview(scores=dict.fromkeys(rule_design.DIMENSIONS, 0)).quality == 0.5
    assert rule_design.DesignReview().quality == 0.5


def test_uncovered_decisions_are_what_the_stats_cannot_say():
    """Precision says nothing about the trajectories no rule touched — which is
    exactly the blindness the mean rewards."""

    uncovered, misfires, n_uncovered, n_labelled = rule_design.build_evidence(
        trajectory_ids=["a", "b", "c", "d"],
        fired_ids={"a"},
        truth_map={"a": Decision.ACCEPT, "b": Decision.REJECT, "c": Decision.REJECT, "d": None},
        misfires=[("a", "rule_x", Decision.ACCEPT, Decision.REJECT)],
    )

    assert (n_uncovered, n_labelled) == (2, 3)  # d is unlabelled, a fired
    assert "b" in uncovered and "c" in uncovered and "a" not in uncovered
    assert "rule_x" in misfires


def test_full_coverage_says_so_rather_than_rendering_an_empty_block():
    uncovered, misfires, n_uncovered, _ = rule_design.build_evidence(
        trajectory_ids=["a"],
        fired_ids={"a"},
        truth_map={"a": Decision.ACCEPT},
        misfires=[],
    )
    assert n_uncovered == 0
    assert "none" in uncovered and "none" in misfires


def test_scores_are_clamped_and_junk_degrades_to_neutral():
    review = rule_design.parse(
        json.dumps(
            {
                "scores": {"coverage": 99, "factoring": "nonsense", "robustness": -7},
                "evidence": {"coverage": "rule_x never fires"},
                "circularity_warnings": ["rule_resolved reads metadata.resolved"],
            }
        )
    )

    assert review.scores["coverage"] == 2
    assert review.scores["factoring"] == 0
    assert review.scores["robustness"] == -2
    assert review.circularity_warnings

    assert rule_design.parse("the model apologised instead").quality == 0.5


# ------------------------------------------------------------------ integration


@dataclass
class _Resp:
    text: str


class _StubClient:
    model = "stub"

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate(self, prompt, system="", temperature=0.0):  # noqa: ARG002
        self.prompts.append(prompt)
        return _Resp(self.payload)


def _rule_always_accept(trajectory):  # noqa: ARG001
    return {"verdict": Decision.ACCEPT}


def _rule_never(trajectory):  # noqa: ARG001
    return None


DESIGN_PAYLOAD = json.dumps(
    {
        "scores": {
            "coverage": -2,
            "precision_discipline": 0,
            "factoring": -2,
            "division_of_labor": 0,
            "robustness": 0,
        },
        "evidence": {"coverage": "no rule fires on the rejected trajectories"},
        "proposals": [
            {
                "action": "add",
                "target": "rule_thrashing",
                "change": "flag repeated failed edits",
                "expected_effect": "covers b and c",
            }
        ],
        "circularity_warnings": [],
    }
)


def _trajectories():
    return [Trajectory(trajectory_id=t, turns=[]) for t in ("a", "b", "c")]


def test_the_review_lowers_a_precise_but_blind_rule_set(tmp_path):
    client = _StubClient(DESIGN_PAYLOAD)
    critic = RuleCritic(client=client, cache_dir=tmp_path, model_tag="stub", score_design=True)

    report = critic.review(
        rules_source="def rule_always_accept(t): ...",
        rule_functions=[_rule_always_accept, _rule_never],
        trajectories=_trajectories(),
        truth_map={"a": Decision.ACCEPT, "b": Decision.REJECT, "c": Decision.REJECT},
        rubric_md="# Rubric",
    )

    assert report.design is not None
    assert report.deterministic_quality is not None
    assert report.rule_quality < report.deterministic_quality
    # The uncovered trajectories have to actually reach the model, or it is
    # scoring coverage from the source alone.
    design_prompt = client.prompts[-1]
    assert "b" in design_prompt and "c" in design_prompt
    assert "uncovered decisions" in design_prompt


def test_the_review_is_off_unless_asked_for(tmp_path):
    client = _StubClient(DESIGN_PAYLOAD)
    critic = RuleCritic(client=client, cache_dir=tmp_path, model_tag="stub")

    report = critic.review(
        rules_source="def rule_always_accept(t): ...",
        rule_functions=[_rule_always_accept, _rule_never],
        trajectories=_trajectories(),
        truth_map={"a": Decision.ACCEPT, "b": Decision.REJECT, "c": Decision.REJECT},
    )

    assert report.design is None
    assert "deterministic_quality" not in report.to_dict()


def test_a_failing_reviewer_leaves_the_measured_score_alone(tmp_path):
    """A critic outage must cost the run its extra signal, not its iteration."""

    class _Broken(_StubClient):
        def generate(self, prompt, system="", temperature=0.0):
            if "Score the design" in prompt:
                raise RuntimeError("claude is down")
            return _Resp(self.payload)

    critic = RuleCritic(
        client=_Broken('{"rules": [], "signals_missing_from_rules": []}'),
        cache_dir=tmp_path,
        model_tag="stub",
        score_design=True,
    )
    report = critic.review(
        rules_source="def rule_always_accept(t): ...",
        rule_functions=[_rule_always_accept, _rule_never],
        trajectories=_trajectories(),
        truth_map={"a": Decision.ACCEPT, "b": Decision.REJECT, "c": Decision.REJECT},
    )

    assert report.design is None
    assert 0.0 <= report.rule_quality <= 1.0
