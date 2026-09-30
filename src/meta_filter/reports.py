"""Rendering helpers: filter diffs and per-iteration Markdown reports."""

from __future__ import annotations

import difflib
from pathlib import Path

from meta_filter.schemas import CandidateEvaluation

# Written verbatim into diff.md when the baseline survived an iteration. The
# viewer matches on it to say "nothing changed" without rendering an empty card.
NO_CHANGES_MD = "_No changes across rubric.md / rules.py / policy.py._\n"


def unified_diff(before_dir: Path, after_dir: Path) -> str:
    """Return a Markdown-formatted diff across the three canonical files."""

    chunks: list[str] = []
    for fname in ("rubric.md", "rules.py", "policy.py"):
        before = _read_or_empty(before_dir / fname)
        after = _read_or_empty(after_dir / fname)
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"before/{fname}",
                tofile=f"after/{fname}",
                n=3,
            )
        )
        if not diff.strip():
            continue
        chunks.append(f"### {fname}\n```diff\n{diff}```\n")
    if not chunks:
        return NO_CHANGES_MD
    return "\n".join(chunks)


def _read_or_empty(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def render_iteration_report(
    *,
    iteration: int,
    baseline_id: str,
    selected_id: str,
    selection_reason: str,
    evaluations: list[CandidateEvaluation],
    diff_md: str,
    new_fn_ids: list[str],
    new_fp_ids: list[str],
    new_boundary_ids: list[str],
    pareto_ids_complexity: list[str] | None = None,
    pareto_ids_cost: list[str] | None = None,
    winner_pareto_safe: bool = False,
) -> str:
    lines: list[str] = []
    lines.append(f"# Iteration {iteration}\n")
    lines.append(f"**Baseline candidate:** `{baseline_id}`\n")
    lines.append(f"**Selected candidate:** `{selected_id}`\n")
    lines.append("\n## Candidate comparison\n")

    for ev in evaluations:
        m = ev.metrics
        mark = " (baseline)" if ev.is_baseline else ""
        mark += " (SELECTED)" if ev.candidate_id == selected_id else ""
        lines.append(f"### {ev.candidate_id}{mark}\n")
        if ev.invalid_reason:
            lines.append(f"- **Invalid**: {ev.invalid_reason}\n")
            continue
        lines.append(
            "\n".join(
                [
                    f"- critic_reject_recall = {m.critic_reject_recall:.3f}",
                    f"- critic_accept_retention = {m.critic_accept_retention:.3f}",
                    f"- disputed_accept_rate = {m.disputed_accept_rate:.3f}",
                    f"- disputed_reject_rate = {m.disputed_reject_rate:.3f}",
                    f"- review_rate = {m.review_rate:.3f}",
                    f"- filter_accept_critic_reject (FN) = {m.filter_accept_critic_reject}",
                    f"- filter_reject_critic_accept (FP) = {m.filter_reject_critic_accept}",
                    f"- reviewed = {m.reviewed}",
                    f"- llm_calls = {m.llm_calls}",
                    f"- estimated_filter_cost = {m.estimated_filter_cost:.3f}",
                    f"- complexity_score = {m.complexity_score:.2f}",
                    f"- complexity_breakdown = {m.complexity}",
                    f"- regression_must_accept_violations = {m.regression_must_accept_violations}",
                    f"- regression_must_reject_violations = {m.regression_must_reject_violations}",
                    f"- **loss** = {m.loss:.3f}",
                ]
            )
        )
        lines.append("")

    lines.append("## Selection\n")
    lines.append(f"{selection_reason}\n")
    safe_str = "on Pareto frontier" if winner_pareto_safe else "not on Pareto frontier"
    lines.append(f"- Winner status: **{safe_str}**\n")
    if pareto_ids_complexity or pareto_ids_cost:
        lines.append("\n## Pareto frontiers\n")
        if pareto_ids_complexity:
            lines.append(f"- (loss, complexity_score): {pareto_ids_complexity}")
        if pareto_ids_cost:
            lines.append(f"- (loss, estimated_filter_cost): {pareto_ids_cost}")

    lines.append("## New counterexamples committed to memory\n")
    lines.append(f"- new false negatives: {new_fn_ids}\n")
    lines.append(f"- new false positives: {new_fp_ids}\n")
    lines.append(f"- new boundary cases : {new_boundary_ids}\n")

    lines.append("\n## Filter diff (previous → selected)\n")
    lines.append(diff_md)

    return "\n".join(lines) + "\n"
