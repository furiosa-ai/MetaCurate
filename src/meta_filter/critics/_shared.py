"""Shared context that every LLM-backed critic prepends to its own prompt.

A common preamble so the judge, RubricCritic and
RuleCritic all speak the same frame: this is a SWE-agent training-data
project, there are no gold labels, and the critic's job is to audit whether
the *filters' reasoning* is looking at the right signals — not just to
approve or reject in a vacuum.

Keeping the text in one place means every critic upgrade lands in every
critic, and prompt drift between them is impossible.
"""

from __future__ import annotations

CRITIC_PREAMBLE = (
    "You are a data-quality auditor for a project that trains SWE (software "
    "engineering) coding agents. The team is filtering agent rollouts to "
    "keep only trajectories that a general-purpose LLM could learn good "
    "SWE behavior from. We do NOT have gold labels. Each trajectory has "
    "been evaluated by candidate filters; each filter uses a shared rubric "
    "prompt (rubric.md) and a set of deterministic checks (rules.py). Your "
    "job varies by role, but the common frame is: audit whether the filters "
    "are looking at the right things, and if they are wrong, say clearly "
    "whether the fix belongs in the rubric prompt, in the rules, or in the "
    "aggregation policy."
)
