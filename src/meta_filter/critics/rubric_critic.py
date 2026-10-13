"""RubricCritic — LLM audit of the rubric prompt itself.

Purpose: read `rubric.md` critically as a prompt-
engineering artifact, paired with a small curated evidence set (trajectory
summary + filter's rubric_scores + filter verdict + judge verdict). Return a
scalar `rubric_quality ∈ [0,1]`, a list of concrete issues, and a list of
missing criteria the rubric should name but doesn't.

The scalar feeds `FilterMetrics.rubric_quality`; the issue / missing lists
land in the proposer prompt as targeted advice.

One LLM call per iteration. Output cached by content hash so re-running the
same iteration is free.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meta_filter.critics._shared import CRITIC_PREAMBLE, load_cache
from meta_filter.llm.base import LLMClient

logger = logging.getLogger(__name__)


RUBRIC_CRITIC_SYSTEM = (
    CRITIC_PREAMBLE + " Your specific role: audit the rubric.md prompt. Decide whether the "
    "criteria and their anchor descriptions name the signals a senior SWE "
    "reviewer would actually check. Return exactly one JSON object; no prose "
    "outside it."
)

RUBRIC_CRITIC_PROMPT = """\
Audit the rubric prompt below and the paired evidence.

=== rubric.md ===
{rubric_md}

=== Paired evidence ({n_cases} cases) ===
{evidence_block}

For each case, ask:
  1. Does the filter's rubric_scores match what a senior SWE reviewer would
     give, based on the trajectory summary and the judge's verdict?
  2. If not, which criterion is under- or over-firing? Is its anchor text
     vague, missing a case, or blaming the wrong signal?

Then step back and look at the rubric as a whole:
  - Are there SWE-specific signals (e.g. "regression risk", "root cause vs
    symptom", "test-was-added-for-the-fix", "patch touches unrelated files")
    that a senior reviewer would check but the rubric never names?

Return exactly this JSON (no other text):
{{
  "rubric_quality": <float in [0.0, 1.0]>,
  "issues": [
    {{"criterion": "<name>", "problem": "<one sentence>",
      "suggested_edit": "<one short sentence describing the edit>"}}
  ],
  "missing_criteria": [
    "<short name of a signal the rubric should name but doesn't>"
  ]
}}"""


@dataclass
class RubricCriticReport:
    """RubricCritic's structured output for a single iteration."""

    rubric_quality: float
    issues: list[dict[str, str]] = field(default_factory=list)
    missing_criteria: list[str] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_quality": self.rubric_quality,
            "issues": list(self.issues),
            "missing_criteria": list(self.missing_criteria),
        }


@dataclass
class RubricCritic:
    """LLM-backed rubric auditor with a filesystem cache."""

    client: LLMClient
    cache_dir: Path = Path("data/rubric_critic_cache")
    model_tag: str = ""

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.model_tag:
            self.model_tag = getattr(self.client, "model", "unknown")

    def review(
        self,
        rubric_md: str,
        curated_evidence: list[dict[str, Any]],
    ) -> RubricCriticReport:
        """Produce a report for one iteration.

        `curated_evidence` is a list of case dicts. Every case is treated as
        opaque text: caller decides how much detail to put in (histogram,
        turns, rubric_scores, verdicts, judge misfire). We only demand a
        stable serialization so caching works.
        """

        evidence_block = _format_evidence(curated_evidence)
        prompt = RUBRIC_CRITIC_PROMPT.format(
            rubric_md=rubric_md.strip(),
            n_cases=len(curated_evidence),
            evidence_block=evidence_block or "(none)",
        )
        cache_key = _hash_inputs(rubric_md, curated_evidence, self.model_tag)
        cache_path = self.cache_dir / f"{cache_key}.json"
        cached = load_cache(cache_path)
        if cached is not None:
            return RubricCriticReport(
                rubric_quality=cached["rubric_quality"],
                issues=cached.get("issues", []),
                missing_criteria=cached.get("missing_criteria", []),
                from_cache=True,
            )

        raw = self._ask(prompt)
        parsed = _parse_report(raw)
        cache_path.write_text(
            json.dumps(
                {
                    **parsed,
                    "model": self.model_tag,
                    "raw": raw,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return RubricCriticReport(
            rubric_quality=parsed["rubric_quality"],
            issues=parsed["issues"],
            missing_criteria=parsed["missing_criteria"],
            from_cache=False,
        )

    def _ask(self, prompt: str) -> str:
        """The single LLM call. Overridden by the agentic critic, which lets
        the model read the run's actual solver output instead."""

        return self.client.generate(prompt, system=RUBRIC_CRITIC_SYSTEM, temperature=0.0).text


def _format_evidence(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return ""
    parts: list[str] = []
    for i, case in enumerate(cases):
        parts.append(f"--- case {i:02d} ---")
        parts.append(json.dumps(case, indent=2, ensure_ascii=False))
    return "\n".join(parts)


def _hash_inputs(
    rubric_md: str,
    curated_evidence: list[dict[str, Any]],
    model_tag: str,
) -> str:
    h = hashlib.sha256()
    h.update(model_tag.encode())
    h.update(b"\0")
    h.update(rubric_md.encode())
    h.update(b"\0")
    h.update(json.dumps(curated_evidence, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()[:16]


def _parse_report(text: str) -> dict[str, Any]:
    """Extract fields from the LLM's JSON. Missing fields fall back to safe
    defaults so a single malformed response doesn't crash the optimizer.
    """

    default = {
        "rubric_quality": 0.5,
        "issues": [],
        "missing_criteria": [],
    }
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    obj: Any = None
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
    if not isinstance(obj, dict):
        return default

    try:
        raw_quality = float(obj.get("rubric_quality", 0.5))
    except (TypeError, ValueError):
        raw_quality = 0.5
    quality = max(0.0, min(1.0, raw_quality))

    issues: list[dict[str, str]] = []
    for item in obj.get("issues", []) or []:
        if not isinstance(item, dict):
            continue
        issues.append(
            {
                "criterion": str(item.get("criterion", ""))[:120],
                "problem": str(item.get("problem", ""))[:400],
                "suggested_edit": str(item.get("suggested_edit", ""))[:400],
            }
        )

    missing: list[str] = []
    for item in obj.get("missing_criteria", []) or []:
        text_item = str(item).strip()
        if text_item:
            missing.append(text_item[:200])

    return {
        "rubric_quality": quality,
        "issues": issues,
        "missing_criteria": missing,
    }
