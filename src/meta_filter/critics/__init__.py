"""Critics: symmetric recall, precision, and boundary auditors.

The three critics share a `CriticBackend` abstraction, so the source of the
adjudicating opinion — LLM judge, ensemble, human audit — is swappable and
always explicit. None of them is ground truth.
"""

from meta_filter.critics.backend import CriticBackend, StaticVerdictBackend
from meta_filter.critics.boundary import BoundaryCritic
from meta_filter.critics.precision import PrecisionCritic
from meta_filter.critics.recall import RecallCritic
from meta_filter.critics.rubric_critic import RubricCritic, RubricCriticReport
from meta_filter.critics.rule_critic import RuleCritic, RuleCriticReport, RuleStats
from meta_filter.critics.trajectory_critic import TrajectoryCritic, TrajectoryVerdict

__all__ = [
    "CriticBackend",
    "StaticVerdictBackend",
    "TrajectoryCritic",
    "TrajectoryVerdict",
    "BoundaryCritic",
    "PrecisionCritic",
    "RecallCritic",
    "RubricCritic",
    "RubricCriticReport",
    "RuleCritic",
    "RuleCriticReport",
    "RuleStats",
]
